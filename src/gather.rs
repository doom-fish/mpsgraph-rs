use crate::ffi;
use crate::graph::Tensor;
use core::ffi::{c_char, c_void};
use core::ptr;
use std::ffi::CString;

fn optional_cstring(name: Option<&str>) -> Option<CString> {
    name.and_then(|value| CString::new(value).ok())
}

#[allow(clippy::ref_option)]
fn cstring_ptr(value: &Option<CString>) -> *const c_char {
    value.as_ref().map_or(ptr::null(), |value| value.as_ptr())
}

fn wrap_tensor(ptr: *mut c_void) -> Option<Tensor> {
    if ptr.is_null() {
        None
    } else {
        Some(Tensor::from_raw(ptr))
    }
}

impl crate::graph::Graph {
    #[must_use]
    pub fn gather_nd(
        &self,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        batch_dimensions: usize,
        name: Option<&str>,
    ) -> Option<Tensor> {
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_gather_nd(
                self.as_ptr(),
                updates_tensor.as_ptr(),
                indices_tensor.as_ptr(),
                batch_dimensions,
                cstring_ptr(&name),
            )
        };
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn gather(
        &self,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        axis: usize,
        batch_dimensions: usize,
        name: Option<&str>,
    ) -> Option<Tensor> {
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_gather(
                self.as_ptr(),
                updates_tensor.as_ptr(),
                indices_tensor.as_ptr(),
                axis,
                batch_dimensions,
                cstring_ptr(&name),
            )
        };
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn gather_along_axis(
        &self,
        axis: isize,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        name: Option<&str>,
    ) -> Option<Tensor> {
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_gather_along_axis(
                self.as_ptr(),
                axis,
                updates_tensor.as_ptr(),
                indices_tensor.as_ptr(),
                cstring_ptr(&name),
            )
        };
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn gather_along_axis_tensor(
        &self,
        axis_tensor: &Tensor,
        updates_tensor: &Tensor,
        indices_tensor: &Tensor,
        name: Option<&str>,
    ) -> Option<Tensor> {
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_gather_along_axis_tensor(
                self.as_ptr(),
                axis_tensor.as_ptr(),
                updates_tensor.as_ptr(),
                indices_tensor.as_ptr(),
                cstring_ptr(&name),
            )
        };
        wrap_tensor(ptr)
    }
}
