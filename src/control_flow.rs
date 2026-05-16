use crate::ffi;
use crate::graph::Tensor;
use crate::types::{collect_owned_tensors, Operation};
use core::ffi::{c_char, c_void};
use core::ptr;
use std::ffi::CString;
use std::mem;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn optional_cstring(name: Option<&str>) -> Option<CString> {
    name.and_then(|value| CString::new(value).ok())
}

#[allow(clippy::ref_option)]
fn cstring_ptr(value: &Option<CString>) -> *const c_char {
    value.as_ref().map_or(ptr::null(), |value| value.as_ptr())
}

fn tensor_array_box_from_tensors(tensors: &[Tensor]) -> *mut c_void {
    let handles = tensors.iter().map(Tensor::as_ptr).collect::<Vec<_>>();
    let handles_ptr = if handles.is_empty() {
        ptr::null()
    } else {
        handles.as_ptr()
    };
    // SAFETY: the handles stay valid for the duration of the bridge call and the Swift array retains them.
    unsafe { ffi::mpsgraph_tensor_array_box_new(handles_ptr, handles.len()) }
}

fn wrap_tensor_array(box_handle: *mut c_void) -> Option<Vec<Tensor>> {
    if box_handle.is_null() {
        None
    } else {
        Some(collect_owned_tensors(box_handle))
    }
}

fn into_owned_tensor_handle(tensor: Tensor) -> *mut c_void {
    let ptr = tensor.as_ptr();
    mem::forget(tensor);
    ptr
}

pub struct WhileBeforeResult {
    pub predicate: Tensor,
    pub results: Vec<Tensor>,
}

struct ZeroArgCallbackContext<'a, F> {
    callback: &'a mut F,
}

unsafe extern "C" fn zero_arg_tensor_array_trampoline<F>(context: *mut c_void) -> *mut c_void
where
    F: FnMut() -> Vec<Tensor>,
{
    let context = unsafe { &mut *context.cast::<ZeroArgCallbackContext<'_, F>>() };
    let tensors = catch_unwind(AssertUnwindSafe(|| (context.callback)()))
        .unwrap_or_else(|_| std::process::abort());
    tensor_array_box_from_tensors(&tensors)
}

struct WhileBeforeCallbackContext<'a, F> {
    callback: &'a mut F,
}

unsafe extern "C" fn while_before_trampoline<F>(
    context: *mut c_void,
    input_box_handle: *mut c_void,
    out_result_box_handle: *mut *mut c_void,
) -> *mut c_void
where
    F: FnMut(&[Tensor]) -> WhileBeforeResult,
{
    let context = unsafe { &mut *context.cast::<WhileBeforeCallbackContext<'_, F>>() };
    let inputs = collect_owned_tensors(input_box_handle);
    match catch_unwind(AssertUnwindSafe(|| (context.callback)(&inputs))) {
        Ok(result) => {
            if !out_result_box_handle.is_null() {
                // SAFETY: caller provided a valid output slot for the tensor-array box.
                unsafe { *out_result_box_handle = tensor_array_box_from_tensors(&result.results) };
            }
            into_owned_tensor_handle(result.predicate)
        }
        Err(_) => std::process::abort(),
    }
}

struct TensorArrayInputCallbackContext<'a, F> {
    callback: &'a mut F,
}

unsafe extern "C" fn tensor_array_input_trampoline<F>(
    context: *mut c_void,
    input_box_handle: *mut c_void,
) -> *mut c_void
where
    F: FnMut(&[Tensor]) -> Vec<Tensor>,
{
    let context = unsafe { &mut *context.cast::<TensorArrayInputCallbackContext<'_, F>>() };
    let inputs = collect_owned_tensors(input_box_handle);
    let tensors = catch_unwind(AssertUnwindSafe(|| (context.callback)(&inputs)))
        .unwrap_or_else(|_| std::process::abort());
    tensor_array_box_from_tensors(&tensors)
}

struct ForBodyCallbackContext<'a, F> {
    callback: &'a mut F,
}

unsafe extern "C" fn for_body_trampoline<F>(
    context: *mut c_void,
    index_handle: *mut c_void,
    input_box_handle: *mut c_void,
) -> *mut c_void
where
    F: FnMut(&Tensor, &[Tensor]) -> Vec<Tensor>,
{
    if index_handle.is_null() {
        return ptr::null_mut();
    }
    let context = unsafe { &mut *context.cast::<ForBodyCallbackContext<'_, F>>() };
    let index = Tensor::from_raw(index_handle);
    let inputs = collect_owned_tensors(input_box_handle);
    let tensors = catch_unwind(AssertUnwindSafe(|| (context.callback)(&index, &inputs)))
        .unwrap_or_else(|_| std::process::abort());
    tensor_array_box_from_tensors(&tensors)
}

impl crate::graph::Graph {
    pub fn control_dependency<F>(
        &self,
        operations: &[&Operation],
        mut dependent_block: F,
        name: Option<&str>,
    ) -> Option<Vec<Tensor>>
    where
        F: FnMut() -> Vec<Tensor>,
    {
        let name = optional_cstring(name);
        let operation_handles = operations.iter().map(|operation| operation.as_ptr()).collect::<Vec<_>>();
        let operation_ptr = if operation_handles.is_empty() {
            ptr::null()
        } else {
            operation_handles.as_ptr()
        };
        let mut context = ZeroArgCallbackContext {
            callback: &mut dependent_block,
        };
        // SAFETY: the callback contexts and handles remain valid for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_graph_control_dependency(
                self.as_ptr(),
                operation_ptr,
                operation_handles.len(),
                Some(zero_arg_tensor_array_trampoline::<F>),
                ptr::from_mut(&mut context).cast(),
                cstring_ptr(&name),
            )
        };
        wrap_tensor_array(box_handle)
    }

    pub fn if_then<Then>(
        &self,
        predicate: &Tensor,
        mut then_block: Then,
        name: Option<&str>,
    ) -> Option<Vec<Tensor>>
    where
        Then: FnMut() -> Vec<Tensor>,
    {
        let name = optional_cstring(name);
        let mut then_context = ZeroArgCallbackContext {
            callback: &mut then_block,
        };
        // SAFETY: the callback context and handles remain valid for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_graph_if_then_else(
                self.as_ptr(),
                predicate.as_ptr(),
                Some(zero_arg_tensor_array_trampoline::<Then>),
                ptr::from_mut(&mut then_context).cast(),
                None,
                ptr::null_mut(),
                cstring_ptr(&name),
            )
        };
        wrap_tensor_array(box_handle)
    }

    pub fn if_then_else<Then, Else>(
        &self,
        predicate: &Tensor,
        mut then_block: Then,
        mut else_block: Else,
        name: Option<&str>,
    ) -> Option<Vec<Tensor>>
    where
        Then: FnMut() -> Vec<Tensor>,
        Else: FnMut() -> Vec<Tensor>,
    {
        let name = optional_cstring(name);
        let mut then_context = ZeroArgCallbackContext {
            callback: &mut then_block,
        };
        let mut else_context = ZeroArgCallbackContext {
            callback: &mut else_block,
        };
        // SAFETY: the callback contexts and handles remain valid for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_graph_if_then_else(
                self.as_ptr(),
                predicate.as_ptr(),
                Some(zero_arg_tensor_array_trampoline::<Then>),
                ptr::from_mut(&mut then_context).cast(),
                Some(zero_arg_tensor_array_trampoline::<Else>),
                ptr::from_mut(&mut else_context).cast(),
                cstring_ptr(&name),
            )
        };
        wrap_tensor_array(box_handle)
    }

    pub fn while_loop<Before, After>(
        &self,
        initial_inputs: &[&Tensor],
        mut before: Before,
        mut after: After,
        name: Option<&str>,
    ) -> Option<Vec<Tensor>>
    where
        Before: FnMut(&[Tensor]) -> WhileBeforeResult,
        After: FnMut(&[Tensor]) -> Vec<Tensor>,
    {
        let name = optional_cstring(name);
        let input_handles = initial_inputs.iter().map(|tensor| tensor.as_ptr()).collect::<Vec<_>>();
        let input_ptr = if input_handles.is_empty() {
            ptr::null()
        } else {
            input_handles.as_ptr()
        };
        let mut before_context = WhileBeforeCallbackContext {
            callback: &mut before,
        };
        let mut after_context = TensorArrayInputCallbackContext {
            callback: &mut after,
        };
        // SAFETY: the callback contexts and handles remain valid for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_graph_while_loop(
                self.as_ptr(),
                input_ptr,
                input_handles.len(),
                Some(while_before_trampoline::<Before>),
                ptr::from_mut(&mut before_context).cast(),
                Some(tensor_array_input_trampoline::<After>),
                ptr::from_mut(&mut after_context).cast(),
                cstring_ptr(&name),
            )
        };
        wrap_tensor_array(box_handle)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn for_loop<Body>(
        &self,
        lower_bound: &Tensor,
        upper_bound: &Tensor,
        step: &Tensor,
        initial_body_arguments: &[&Tensor],
        mut body: Body,
        name: Option<&str>,
    ) -> Option<Vec<Tensor>>
    where
        Body: FnMut(&Tensor, &[Tensor]) -> Vec<Tensor>,
    {
        let name = optional_cstring(name);
        let argument_handles = initial_body_arguments
            .iter()
            .map(|tensor| tensor.as_ptr())
            .collect::<Vec<_>>();
        let argument_ptr = if argument_handles.is_empty() {
            ptr::null()
        } else {
            argument_handles.as_ptr()
        };
        let mut context = ForBodyCallbackContext {
            callback: &mut body,
        };
        // SAFETY: the callback context and handles remain valid for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_graph_for_loop(
                self.as_ptr(),
                lower_bound.as_ptr(),
                upper_bound.as_ptr(),
                step.as_ptr(),
                argument_ptr,
                argument_handles.len(),
                Some(for_body_trampoline::<Body>),
                ptr::from_mut(&mut context).cast(),
                cstring_ptr(&name),
            )
        };
        wrap_tensor_array(box_handle)
    }

    pub fn for_loop_iterations<Body>(
        &self,
        number_of_iterations: &Tensor,
        initial_body_arguments: &[&Tensor],
        mut body: Body,
        name: Option<&str>,
    ) -> Option<Vec<Tensor>>
    where
        Body: FnMut(&Tensor, &[Tensor]) -> Vec<Tensor>,
    {
        let name = optional_cstring(name);
        let argument_handles = initial_body_arguments
            .iter()
            .map(|tensor| tensor.as_ptr())
            .collect::<Vec<_>>();
        let argument_ptr = if argument_handles.is_empty() {
            ptr::null()
        } else {
            argument_handles.as_ptr()
        };
        let mut context = ForBodyCallbackContext {
            callback: &mut body,
        };
        // SAFETY: the callback context and handles remain valid for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_graph_for_loop_iterations(
                self.as_ptr(),
                number_of_iterations.as_ptr(),
                argument_ptr,
                argument_handles.len(),
                Some(for_body_trampoline::<Body>),
                ptr::from_mut(&mut context).cast(),
                cstring_ptr(&name),
            )
        };
        wrap_tensor_array(box_handle)
    }
}
