use crate::checks;
use crate::error::{Error, Result};
use crate::ffi;
use crate::graph::{handles, release_handles, take_tensor_handles, Graph, Tensor};
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

fn into_owned_tensor_handle(tensor: Tensor) -> *mut c_void {
    let ptr = tensor.as_ptr();
    mem::forget(tensor);
    ptr
}

fn tensor_handles(tensors: &[&Tensor]) -> Vec<*mut c_void> {
    tensors.iter().map(|tensor| tensor.as_ptr()).collect()
}

fn array_ptr(handles: &[*mut c_void]) -> *const *mut c_void {
    if handles.is_empty() {
        ptr::null()
    } else {
        handles.as_ptr()
    }
}

/// Mirrors the `MPSGraph` framework counterpart for `WhileBeforeResult`.
pub struct WhileBeforeResult {
/// Mirrors the `MPSGraph` framework property for `predicate`.
    pub predicate: Tensor,
/// Mirrors the `MPSGraph` framework property for `results`.
    pub results: Vec<Tensor>,
}

struct Block<'a, F> {
    graph: &'a Graph,
    callback: &'a mut F,
    error: Option<Error>,
    deps: Vec<usize>,
}

impl<'a, F> Block<'a, F> {
    fn new(graph: &'a Graph, callback: &'a mut F) -> Self {
        Self {
            graph,
            callback,
            error: None,
            deps: Vec::new(),
        }
    }

    fn finish(&mut self, scope: Option<u64>, results: &[&Tensor]) -> bool {
        let checked = self.graph.check(results);
        if let Some(scope) = scope {
            self.deps.extend(self.graph.close_block(scope));
        }
        self.deps.extend(handles(results));
        match checked {
            Ok(()) => true,
            Err(error) => {
                self.error.get_or_insert(error);
                false
            }
        }
    }
}

unsafe extern "C" fn tensor_block<F, const SCOPED: bool>(context: *mut c_void) -> *mut c_void
where
    F: FnMut() -> Vec<Tensor>,
{
    // SAFETY: `context` is a pointer to `Block<'_, F>` set up by the safe wrapper at the callsite.
    let block = unsafe { &mut *context.cast::<Block<'_, F>>() };
    let scope = SCOPED.then(|| block.graph.open_block());
    let tensors = catch_unwind(AssertUnwindSafe(|| (block.callback)()))
        .unwrap_or_else(|_| std::process::abort());
    if block.finish(scope, &tensors.iter().collect::<Vec<_>>()) {
        tensor_array_box_from_tensors(&tensors)
    } else {
        ptr::null_mut()
    }
}

unsafe extern "C" fn while_before_block<F>(
    context: *mut c_void,
    input_box_handle: *mut c_void,
    out_result_box_handle: *mut *mut c_void,
) -> *mut c_void
where
    F: FnMut(&[Tensor]) -> WhileBeforeResult,
{
    // SAFETY: `context` is a pointer to `Block<'_, F>` set up by the safe wrapper at the callsite.
    let block = unsafe { &mut *context.cast::<Block<'_, F>>() };
    let scope = block.graph.open_block();
    let inputs = collect_owned_tensors(input_box_handle, scope);
    let result = catch_unwind(AssertUnwindSafe(|| (block.callback)(&inputs)))
        .unwrap_or_else(|_| std::process::abort());
    let produced = result
        .results
        .iter()
        .chain(core::iter::once(&result.predicate))
        .collect::<Vec<_>>();
    if !block.finish(Some(scope), &produced) || out_result_box_handle.is_null() {
        return ptr::null_mut();
    }
    // SAFETY: the caller provided a valid output slot for the tensor-array box.
    unsafe { *out_result_box_handle = tensor_array_box_from_tensors(&result.results) };
    into_owned_tensor_handle(result.predicate)
}

unsafe extern "C" fn tensor_input_block<F>(
    context: *mut c_void,
    input_box_handle: *mut c_void,
) -> *mut c_void
where
    F: FnMut(&[Tensor]) -> Vec<Tensor>,
{
    // SAFETY: `context` is a pointer to `Block<'_, F>` set up by the safe wrapper at the callsite.
    let block = unsafe { &mut *context.cast::<Block<'_, F>>() };
    let scope = block.graph.open_block();
    let inputs = collect_owned_tensors(input_box_handle, scope);
    let tensors = catch_unwind(AssertUnwindSafe(|| (block.callback)(&inputs)))
        .unwrap_or_else(|_| std::process::abort());
    if block.finish(Some(scope), &tensors.iter().collect::<Vec<_>>()) {
        tensor_array_box_from_tensors(&tensors)
    } else {
        ptr::null_mut()
    }
}

unsafe extern "C" fn for_body_block<F>(
    context: *mut c_void,
    index_handle: *mut c_void,
    input_box_handle: *mut c_void,
) -> *mut c_void
where
    F: FnMut(&Tensor, &[Tensor]) -> Vec<Tensor>,
{
    // SAFETY: `context` is a pointer to `Block<'_, F>` set up by the safe wrapper at the callsite.
    let block = unsafe { &mut *context.cast::<Block<'_, F>>() };
    let scope = block.graph.open_block();
    let inputs = collect_owned_tensors(input_box_handle, scope);
    if index_handle.is_null() {
        block.finish(Some(scope), &[]);
        return ptr::null_mut();
    }
    let index = Tensor::from_raw(index_handle, scope);
    let tensors = catch_unwind(AssertUnwindSafe(|| (block.callback)(&index, &inputs)))
        .unwrap_or_else(|_| std::process::abort());
    if block.finish(Some(scope), &tensors.iter().collect::<Vec<_>>()) {
        tensor_array_box_from_tensors(&tensors)
    } else {
        ptr::null_mut()
    }
}

fn finish_control_flow(
    graph: &Graph,
    box_handle: *mut c_void,
    failed: bool,
    deps: &[usize],
    error: Option<Error>,
    mismatch: &'static str,
) -> Result<Vec<Tensor>> {
    if failed || box_handle.is_null() {
        release_handles(take_tensor_handles(box_handle));
        return Err(error.unwrap_or(Error::InvalidShape(mismatch)));
    }
    graph.outputs(
        box_handle,
        deps,
        None,
        "MPSGraph did not create the control-flow operation",
    )
}

const IF_MISMATCH: &str =
    "the then and else blocks must each return at least one tensor, with matching shapes and data types";
const WHILE_MISMATCH: &str = "the before block must return a rank-0 bool predicate and at least one tensor, and the after block tensors matching the initial inputs' shapes and data types";
const FOR_MISMATCH: &str =
    "the loop body must return tensors matching the body arguments' shapes and data types";

impl crate::graph::Graph {
/// Calls the `MPSGraph` framework counterpart for `control_dependency`.
    pub fn control_dependency<F>(
        &self,
        operations: &[&Operation],
        mut dependent_block: F,
        name: Option<&str>,
    ) -> Result<Vec<Tensor>>
    where
        F: FnMut() -> Vec<Tensor>,
    {
        self.check_operations(operations)?;
        let name = optional_cstring(name);
        let operation_handles = operations
            .iter()
            .map(|operation| operation.as_ptr())
            .collect::<Vec<_>>();
        let mark = self.created_mark();
        let mut block = Block::new(self, &mut dependent_block);
        let mut failed = false;
        // SAFETY: the callback contexts and handles remain valid for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_graph_control_dependency(
                self.as_ptr(),
                array_ptr(&operation_handles),
                operation_handles.len(),
                Some(tensor_block::<F, false>),
                ptr::from_mut(&mut block).cast(),
                cstring_ptr(&name),
                &raw mut failed,
            )
        };
        let mut deps = self.created_since(mark);
        deps.extend(block.deps);
        deps.extend(
            operations
                .iter()
                .flat_map(|operation| operation.inputs().iter().copied()),
        );
        finish_control_flow(
            self,
            box_handle,
            failed,
            &deps,
            block.error,
            "the dependent block's tensors must belong to this graph",
        )
    }

/// Calls the `MPSGraph` framework counterpart for `if_then_else`.
    pub fn if_then_else<Then, Else>(
        &self,
        predicate: &Tensor,
        mut then_block: Then,
        mut else_block: Else,
        name: Option<&str>,
    ) -> Result<Vec<Tensor>>
    where
        Then: FnMut() -> Vec<Tensor>,
        Else: FnMut() -> Vec<Tensor>,
    {
        self.check(&[predicate])?;
        checks::single_element(predicate, "the predicate must hold exactly one element")?;
        let name = optional_cstring(name);
        let mut then_context = Block::new(self, &mut then_block);
        let mut else_context = Block::new(self, &mut else_block);
        let mut failed = false;
        // SAFETY: the callback contexts and handles remain valid for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_graph_if_then_else(
                self.as_ptr(),
                predicate.as_ptr(),
                Some(tensor_block::<Then, true>),
                ptr::from_mut(&mut then_context).cast(),
                Some(tensor_block::<Else, true>),
                ptr::from_mut(&mut else_context).cast(),
                cstring_ptr(&name),
                &raw mut failed,
            )
        };
        let mut deps = handles(&[predicate]);
        deps.extend(then_context.deps);
        deps.extend(else_context.deps);
        finish_control_flow(
            self,
            box_handle,
            failed,
            &deps,
            then_context.error.or(else_context.error),
            IF_MISMATCH,
        )
    }

/// Calls the `MPSGraph` framework counterpart for `while_loop`.
    pub fn while_loop<Before, After>(
        &self,
        initial_inputs: &[&Tensor],
        mut before: Before,
        mut after: After,
        name: Option<&str>,
    ) -> Result<Vec<Tensor>>
    where
        Before: FnMut(&[Tensor]) -> WhileBeforeResult,
        After: FnMut(&[Tensor]) -> Vec<Tensor>,
    {
        self.check(initial_inputs)?;
        let name = optional_cstring(name);
        let input_handles = tensor_handles(initial_inputs);
        let mut before_context = Block::new(self, &mut before);
        let mut after_context = Block::new(self, &mut after);
        let mut failed = false;
        // SAFETY: the callback contexts and handles remain valid for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_graph_while_loop(
                self.as_ptr(),
                array_ptr(&input_handles),
                input_handles.len(),
                Some(while_before_block::<Before>),
                ptr::from_mut(&mut before_context).cast(),
                Some(tensor_input_block::<After>),
                ptr::from_mut(&mut after_context).cast(),
                cstring_ptr(&name),
                &raw mut failed,
            )
        };
        let mut deps = handles(initial_inputs);
        deps.extend(before_context.deps);
        deps.extend(after_context.deps);
        finish_control_flow(
            self,
            box_handle,
            failed,
            &deps,
            before_context.error.or(after_context.error),
            WHILE_MISMATCH,
        )
    }

/// Calls the `MPSGraph` framework counterpart for `for_loop`.
    #[allow(clippy::too_many_arguments)]
    pub fn for_loop<Body>(
        &self,
        lower_bound: &Tensor,
        upper_bound: &Tensor,
        step: &Tensor,
        initial_body_arguments: &[&Tensor],
        mut body: Body,
        name: Option<&str>,
    ) -> Result<Vec<Tensor>>
    where
        Body: FnMut(&Tensor, &[Tensor]) -> Vec<Tensor>,
    {
        self.check(&[lower_bound, upper_bound, step])?;
        self.check(initial_body_arguments)?;
        for bound in [lower_bound, upper_bound, step] {
            checks::single_element(bound, "loop bounds must hold exactly one element")?;
        }
        check_body_arguments(initial_body_arguments)?;
        let name = optional_cstring(name);
        let argument_handles = tensor_handles(initial_body_arguments);
        let mut context = Block::new(self, &mut body);
        let mut failed = false;
        // SAFETY: the callback context and handles remain valid for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_graph_for_loop(
                self.as_ptr(),
                lower_bound.as_ptr(),
                upper_bound.as_ptr(),
                step.as_ptr(),
                array_ptr(&argument_handles),
                argument_handles.len(),
                Some(for_body_block::<Body>),
                ptr::from_mut(&mut context).cast(),
                cstring_ptr(&name),
                &raw mut failed,
            )
        };
        let mut deps = handles(&[lower_bound, upper_bound, step]);
        deps.extend(handles(initial_body_arguments));
        deps.extend(context.deps);
        finish_control_flow(self, box_handle, failed, &deps, context.error, FOR_MISMATCH)
    }

/// Calls the `MPSGraph` framework counterpart for `for_loop_iterations`.
    pub fn for_loop_iterations<Body>(
        &self,
        number_of_iterations: &Tensor,
        initial_body_arguments: &[&Tensor],
        mut body: Body,
        name: Option<&str>,
    ) -> Result<Vec<Tensor>>
    where
        Body: FnMut(&Tensor, &[Tensor]) -> Vec<Tensor>,
    {
        self.check(&[number_of_iterations])?;
        self.check(initial_body_arguments)?;
        checks::single_element(
            number_of_iterations,
            "the iteration count must hold exactly one element",
        )?;
        check_body_arguments(initial_body_arguments)?;
        let name = optional_cstring(name);
        let argument_handles = tensor_handles(initial_body_arguments);
        let mut context = Block::new(self, &mut body);
        let mut failed = false;
        // SAFETY: the callback context and handles remain valid for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_graph_for_loop_iterations(
                self.as_ptr(),
                number_of_iterations.as_ptr(),
                array_ptr(&argument_handles),
                argument_handles.len(),
                Some(for_body_block::<Body>),
                ptr::from_mut(&mut context).cast(),
                cstring_ptr(&name),
                &raw mut failed,
            )
        };
        let mut deps = handles(&[number_of_iterations]);
        deps.extend(handles(initial_body_arguments));
        deps.extend(context.deps);
        finish_control_flow(self, box_handle, failed, &deps, context.error, FOR_MISMATCH)
    }
}

fn check_body_arguments(arguments: &[&Tensor]) -> Result<()> {
    if arguments.is_empty() {
        Err(Error::InvalidArgument(
            "for loops need at least one body argument; MPSGraph crashes without one",
        ))
    } else {
        Ok(())
    }
}
