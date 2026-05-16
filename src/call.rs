use crate::error::{Error, Result};
use crate::ffi;
use crate::graph::Tensor;
use crate::types::{collect_owned_tensors, ShapedType};
use core::ffi::{c_char, c_void};
use core::ptr;
use std::ffi::CString;

fn cstring_ptr(value: &CString) -> *const c_char {
    value.as_ptr()
}

fn optional_cstring(name: Option<&str>) -> Option<CString> {
    name.and_then(|value| CString::new(value).ok())
}

#[allow(clippy::ref_option)]
fn optional_name_ptr(value: &Option<CString>) -> *const c_char {
    value.as_ref().map_or(ptr::null(), |value| value.as_ptr())
}

fn wrap_tensor_array(box_handle: *mut c_void) -> Option<Vec<Tensor>> {
    if box_handle.is_null() {
        None
    } else {
        Some(collect_owned_tensors(box_handle))
    }
}

impl crate::graph::Graph {
    pub fn call(
        &self,
        symbol_name: &str,
        input_tensors: &[&Tensor],
        output_types: &[&ShapedType],
        name: Option<&str>,
    ) -> Result<Vec<Tensor>> {
        let symbol_name =
            CString::new(symbol_name).map_err(|_| Error::OperationFailed("call symbol name contained NUL"))?;
        let name = optional_cstring(name);
        let input_handles = input_tensors.iter().map(|tensor| tensor.as_ptr()).collect::<Vec<_>>();
        let output_type_handles = output_types
            .iter()
            .map(|output_type| output_type.as_ptr())
            .collect::<Vec<_>>();
        let input_ptr = if input_handles.is_empty() {
            ptr::null()
        } else {
            input_handles.as_ptr()
        };
        let output_type_ptr = if output_type_handles.is_empty() {
            ptr::null()
        } else {
            output_type_handles.as_ptr()
        };
        // SAFETY: all handles remain valid for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_graph_call_symbol(
                self.as_ptr(),
                cstring_ptr(&symbol_name),
                input_ptr,
                input_handles.len(),
                output_type_ptr,
                output_type_handles.len(),
                optional_name_ptr(&name),
            )
        };
        wrap_tensor_array(box_handle).ok_or(Error::OperationFailed("failed to create call op"))
    }
}
