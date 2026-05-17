use crate::error::{Error, Result};
use crate::ffi;
use crate::graph::Tensor;
use crate::types::collect_owned_tensors;
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

fn wrap_tensor_pair(box_handle: *mut c_void) -> Option<(Tensor, Tensor)> {
    let mut values = collect_owned_tensors(box_handle);
    if values.len() != 2 {
        return None;
    }
    let second = values.pop()?;
    let first = values.pop()?;
    Some((first, second))
}

/// `MPSGraphRandomDistribution` constants.
pub mod random_distribution {
    pub const UNIFORM: u64 = 0;
    pub const NORMAL: u64 = 1;
    pub const TRUNCATED_NORMAL: u64 = 2;
}

/// `MPSGraphRandomNormalSamplingMethod` constants.
pub mod random_normal_sampling_method {
    pub const INV_CDF: u64 = 0;
    pub const BOX_MULLER: u64 = 1;
}

/// Safe owner for `MPSGraphRandomOpDescriptor`.
pub struct RandomOpDescriptor {
    ptr: *mut c_void,
}

unsafe impl Send for RandomOpDescriptor {}
unsafe impl Sync for RandomOpDescriptor {}

impl Drop for RandomOpDescriptor {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `ptr` is a +1 retained Swift/ObjC object pointer owned by this wrapper.
            unsafe { ffi::mpsgraph_object_release(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

impl RandomOpDescriptor {
    #[must_use]
    pub fn new(distribution: u64, data_type: u32) -> Option<Self> {
        // SAFETY: pure constructor with POD arguments.
        let ptr = unsafe { ffi::mpsgraph_random_op_descriptor_new(distribution, data_type) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub(crate) const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    #[must_use]
    pub fn distribution(&self) -> u64 {
        // SAFETY: `self.ptr` is a live descriptor handle.
        unsafe { ffi::mpsgraph_random_op_descriptor_distribution(self.ptr) }
    }

    pub fn set_distribution(&self, value: u64) -> Result<()> {
        // SAFETY: `self.ptr` is a live descriptor handle.
        let ok = unsafe { ffi::mpsgraph_random_op_descriptor_set_distribution(self.ptr, value) };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed("failed to set random distribution"))
        }
    }

    #[must_use]
    pub fn data_type(&self) -> u32 {
        // SAFETY: `self.ptr` is a live descriptor handle.
        unsafe { ffi::mpsgraph_random_op_descriptor_data_type(self.ptr) }
    }

    pub fn set_data_type(&self, value: u32) -> Result<()> {
        // SAFETY: `self.ptr` is a live descriptor handle.
        let ok = unsafe { ffi::mpsgraph_random_op_descriptor_set_data_type(self.ptr, value) };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed("failed to set random data type"))
        }
    }

    #[must_use]
    pub fn min(&self) -> f32 {
        // SAFETY: `self.ptr` is a live descriptor handle.
        unsafe { ffi::mpsgraph_random_op_descriptor_min(self.ptr) }
    }

    pub fn set_min(&self, value: f32) -> Result<()> {
        // SAFETY: `self.ptr` is a live descriptor handle.
        let ok = unsafe { ffi::mpsgraph_random_op_descriptor_set_min(self.ptr, value) };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed("failed to set random min"))
        }
    }

    #[must_use]
    pub fn max(&self) -> f32 {
        // SAFETY: `self.ptr` is a live descriptor handle.
        unsafe { ffi::mpsgraph_random_op_descriptor_max(self.ptr) }
    }

    pub fn set_max(&self, value: f32) -> Result<()> {
        // SAFETY: `self.ptr` is a live descriptor handle.
        let ok = unsafe { ffi::mpsgraph_random_op_descriptor_set_max(self.ptr, value) };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed("failed to set random max"))
        }
    }

    #[must_use]
    pub fn min_integer(&self) -> isize {
        // SAFETY: `self.ptr` is a live descriptor handle.
        unsafe { ffi::mpsgraph_random_op_descriptor_min_integer(self.ptr) }
    }

    pub fn set_min_integer(&self, value: isize) -> Result<()> {
        // SAFETY: `self.ptr` is a live descriptor handle.
        let ok = unsafe { ffi::mpsgraph_random_op_descriptor_set_min_integer(self.ptr, value) };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed("failed to set random minInteger"))
        }
    }

    #[must_use]
    pub fn max_integer(&self) -> isize {
        // SAFETY: `self.ptr` is a live descriptor handle.
        unsafe { ffi::mpsgraph_random_op_descriptor_max_integer(self.ptr) }
    }

    pub fn set_max_integer(&self, value: isize) -> Result<()> {
        // SAFETY: `self.ptr` is a live descriptor handle.
        let ok = unsafe { ffi::mpsgraph_random_op_descriptor_set_max_integer(self.ptr, value) };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed("failed to set random maxInteger"))
        }
    }

    #[must_use]
    pub fn mean(&self) -> f32 {
        // SAFETY: `self.ptr` is a live descriptor handle.
        unsafe { ffi::mpsgraph_random_op_descriptor_mean(self.ptr) }
    }

    pub fn set_mean(&self, value: f32) -> Result<()> {
        // SAFETY: `self.ptr` is a live descriptor handle.
        let ok = unsafe { ffi::mpsgraph_random_op_descriptor_set_mean(self.ptr, value) };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed("failed to set random mean"))
        }
    }

    #[must_use]
    pub fn standard_deviation(&self) -> f32 {
        // SAFETY: `self.ptr` is a live descriptor handle.
        unsafe { ffi::mpsgraph_random_op_descriptor_standard_deviation(self.ptr) }
    }

    pub fn set_standard_deviation(&self, value: f32) -> Result<()> {
        // SAFETY: `self.ptr` is a live descriptor handle.
        let ok =
            unsafe { ffi::mpsgraph_random_op_descriptor_set_standard_deviation(self.ptr, value) };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed(
                "failed to set random standardDeviation",
            ))
        }
    }

    #[must_use]
    pub fn sampling_method(&self) -> u64 {
        // SAFETY: `self.ptr` is a live descriptor handle.
        unsafe { ffi::mpsgraph_random_op_descriptor_sampling_method(self.ptr) }
    }

    pub fn set_sampling_method(&self, value: u64) -> Result<()> {
        // SAFETY: `self.ptr` is a live descriptor handle.
        let ok = unsafe { ffi::mpsgraph_random_op_descriptor_set_sampling_method(self.ptr, value) };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed(
                "failed to set random sampling method",
            ))
        }
    }
}

impl crate::graph::Graph {
    #[must_use]
    pub fn random_philox_state_seed(&self, seed: usize, name: Option<&str>) -> Option<Tensor> {
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_random_philox_state_seed(self.as_ptr(), seed, cstring_ptr(&name))
        };
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn random_philox_state_counter(
        &self,
        counter_low: usize,
        counter_high: usize,
        key: usize,
        name: Option<&str>,
    ) -> Option<Tensor> {
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_random_philox_state_counter(
                self.as_ptr(),
                counter_low,
                counter_high,
                key,
                cstring_ptr(&name),
            )
        };
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn random_tensor(
        &self,
        shape: &[usize],
        descriptor: &RandomOpDescriptor,
        name: Option<&str>,
    ) -> Option<Tensor> {
        let name = optional_cstring(name);
        let shape_ptr = if shape.is_empty() {
            ptr::null()
        } else {
            shape.as_ptr()
        };
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_random_tensor(
                self.as_ptr(),
                shape_ptr,
                shape.len(),
                descriptor.as_ptr(),
                cstring_ptr(&name),
            )
        };
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn random_tensor_shape_tensor(
        &self,
        shape_tensor: &Tensor,
        descriptor: &RandomOpDescriptor,
        name: Option<&str>,
    ) -> Option<Tensor> {
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_random_tensor_shape_tensor(
                self.as_ptr(),
                shape_tensor.as_ptr(),
                descriptor.as_ptr(),
                cstring_ptr(&name),
            )
        };
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn random_tensor_seed(
        &self,
        shape: &[usize],
        descriptor: &RandomOpDescriptor,
        seed: usize,
        name: Option<&str>,
    ) -> Option<Tensor> {
        let name = optional_cstring(name);
        let shape_ptr = if shape.is_empty() {
            ptr::null()
        } else {
            shape.as_ptr()
        };
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_random_tensor_seed(
                self.as_ptr(),
                shape_ptr,
                shape.len(),
                descriptor.as_ptr(),
                seed,
                cstring_ptr(&name),
            )
        };
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn random_tensor_shape_tensor_seed(
        &self,
        shape_tensor: &Tensor,
        descriptor: &RandomOpDescriptor,
        seed: usize,
        name: Option<&str>,
    ) -> Option<Tensor> {
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_random_tensor_shape_tensor_seed(
                self.as_ptr(),
                shape_tensor.as_ptr(),
                descriptor.as_ptr(),
                seed,
                cstring_ptr(&name),
            )
        };
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn random_tensor_state(
        &self,
        shape: &[usize],
        descriptor: &RandomOpDescriptor,
        state: &Tensor,
        name: Option<&str>,
    ) -> Option<(Tensor, Tensor)> {
        let name = optional_cstring(name);
        let shape_ptr = if shape.is_empty() {
            ptr::null()
        } else {
            shape.as_ptr()
        };
        // SAFETY: all handles remain valid for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_graph_random_tensor_state(
                self.as_ptr(),
                shape_ptr,
                shape.len(),
                descriptor.as_ptr(),
                state.as_ptr(),
                cstring_ptr(&name),
            )
        };
        wrap_tensor_pair(box_handle)
    }

    #[must_use]
    pub fn random_tensor_shape_tensor_state(
        &self,
        shape_tensor: &Tensor,
        descriptor: &RandomOpDescriptor,
        state: &Tensor,
        name: Option<&str>,
    ) -> Option<(Tensor, Tensor)> {
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_graph_random_tensor_shape_tensor_state(
                self.as_ptr(),
                shape_tensor.as_ptr(),
                descriptor.as_ptr(),
                state.as_ptr(),
                cstring_ptr(&name),
            )
        };
        wrap_tensor_pair(box_handle)
    }

    #[must_use]
    pub fn dropout(&self, tensor: &Tensor, rate: f64, name: Option<&str>) -> Option<Tensor> {
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_dropout(self.as_ptr(), tensor.as_ptr(), rate, cstring_ptr(&name))
        };
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn dropout_tensor(
        &self,
        tensor: &Tensor,
        rate_tensor: &Tensor,
        name: Option<&str>,
    ) -> Option<Tensor> {
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_dropout_tensor(
                self.as_ptr(),
                tensor.as_ptr(),
                rate_tensor.as_ptr(),
                cstring_ptr(&name),
            )
        };
        wrap_tensor(ptr)
    }
}
