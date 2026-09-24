use crate::checks;
use crate::error::{Error, Result};
use crate::ffi;
use crate::graph::{data_type, Tensor};
use core::cell::Cell;
use core::ffi::{c_char, c_void};
use core::marker::PhantomData;
use core::ptr;
use std::ffi::CString;

fn optional_cstring(name: Option<&str>) -> Option<CString> {
    name.and_then(|value| CString::new(value).ok())
}

#[allow(clippy::ref_option)]
fn cstring_ptr(value: &Option<CString>) -> *const c_char {
    value.as_ref().map_or(ptr::null(), |value| value.as_ptr())
}

fn check_distribution(distribution: u64, data_type: u32) -> Result<()> {
    if distribution > random_distribution::TRUNCATED_NORMAL {
        return Err(Error::InvalidArgument("unknown MPSGraphRandomDistribution"));
    }
    let supported = if distribution == random_distribution::UNIFORM {
        matches!(
            data_type,
            data_type::FLOAT16 | data_type::FLOAT32 | data_type::INT32
        )
    } else {
        matches!(data_type, data_type::FLOAT16 | data_type::FLOAT32)
    };
    if supported {
        Ok(())
    } else {
        Err(Error::InvalidDataType(
            "uniform distributions generate float16, float32 or int32 values and normal distributions float16 or float32",
        ))
    }
}

fn check_state(state: &Tensor) -> Result<()> {
    if state.data_type() != data_type::INT32 {
        return Err(Error::InvalidDataType(
            "the Philox state must be an int32 tensor",
        ));
    }
    checks::dims_match(
        state,
        &[PHILOX_STATE_LENGTH],
        "the Philox state must have shape [7]",
    )
}

const PHILOX_STATE_LENGTH: isize = 7;

fn check_shape_tensor(shape_tensor: &Tensor) -> Result<()> {
    checks::index_vector(
        shape_tensor,
        None,
        "the shape tensor must be a rank-1 int32 or int64 tensor",
    )
}

/// `MPSGraphRandomDistribution` constants.
pub mod random_distribution {
/// Mirrors the `MPSGraph` framework constant `UNIFORM`.
    pub const UNIFORM: u64 = 0;
/// Mirrors the `MPSGraph` framework constant `NORMAL`.
    pub const NORMAL: u64 = 1;
/// Mirrors the `MPSGraph` framework constant `TRUNCATED_NORMAL`.
    pub const TRUNCATED_NORMAL: u64 = 2;
}

/// `MPSGraphRandomNormalSamplingMethod` constants.
pub mod random_normal_sampling_method {
/// Mirrors the `MPSGraph` framework constant `INV_CDF`.
    pub const INV_CDF: u64 = 0;
/// Mirrors the `MPSGraph` framework constant `BOX_MULLER`.
    pub const BOX_MULLER: u64 = 1;
}

/// Safe owner for `MPSGraphRandomOpDescriptor`.
pub struct RandomOpDescriptor {
    ptr: *mut c_void,
    _not_sync: PhantomData<Cell<()>>,
}

unsafe impl Send for RandomOpDescriptor {}

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
/// Calls the `MPSGraph` framework counterpart for `new`.
    pub fn new(distribution: u64, data_type: u32) -> Result<Self> {
        check_distribution(distribution, data_type)?;
        // SAFETY: pure constructor with POD arguments.
        let ptr = unsafe { ffi::mpsgraph_random_op_descriptor_new(distribution, data_type) };
        if ptr.is_null() {
            Err(Error::OperationFailed(
                "MPSGraph did not create the random descriptor",
            ))
        } else {
            Ok(Self {
                ptr,
                _not_sync: PhantomData,
            })
        }
    }

    #[must_use]
    pub(crate) const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

/// Calls the `MPSGraph` framework counterpart for `distribution`.
    #[must_use]
    pub fn distribution(&self) -> u64 {
        // SAFETY: `self.ptr` is a live descriptor handle.
        unsafe { ffi::mpsgraph_random_op_descriptor_distribution(self.ptr) }
    }

/// Calls the `MPSGraph` framework counterpart for `set_distribution`.
    pub fn set_distribution(&self, value: u64) -> Result<()> {
        check_distribution(value, self.data_type())?;
        // SAFETY: `self.ptr` is a live descriptor handle.
        let ok = unsafe { ffi::mpsgraph_random_op_descriptor_set_distribution(self.ptr, value) };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed("failed to set random distribution"))
        }
    }

/// Calls the `MPSGraph` framework counterpart for `data_type`.
    #[must_use]
    pub fn data_type(&self) -> u32 {
        // SAFETY: `self.ptr` is a live descriptor handle.
        unsafe { ffi::mpsgraph_random_op_descriptor_data_type(self.ptr) }
    }

/// Calls the `MPSGraph` framework counterpart for `set_data_type`.
    pub fn set_data_type(&self, value: u32) -> Result<()> {
        check_distribution(self.distribution(), value)?;
        // SAFETY: `self.ptr` is a live descriptor handle.
        let ok = unsafe { ffi::mpsgraph_random_op_descriptor_set_data_type(self.ptr, value) };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed("failed to set random data type"))
        }
    }

/// Calls the `MPSGraph` framework counterpart for `min`.
    #[must_use]
    pub fn min(&self) -> f32 {
        // SAFETY: `self.ptr` is a live descriptor handle.
        unsafe { ffi::mpsgraph_random_op_descriptor_min(self.ptr) }
    }

/// Calls the `MPSGraph` framework counterpart for `set_min`.
    pub fn set_min(&self, value: f32) -> Result<()> {
        // SAFETY: `self.ptr` is a live descriptor handle.
        let ok = unsafe { ffi::mpsgraph_random_op_descriptor_set_min(self.ptr, value) };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed("failed to set random min"))
        }
    }

/// Calls the `MPSGraph` framework counterpart for `max`.
    #[must_use]
    pub fn max(&self) -> f32 {
        // SAFETY: `self.ptr` is a live descriptor handle.
        unsafe { ffi::mpsgraph_random_op_descriptor_max(self.ptr) }
    }

/// Calls the `MPSGraph` framework counterpart for `set_max`.
    pub fn set_max(&self, value: f32) -> Result<()> {
        // SAFETY: `self.ptr` is a live descriptor handle.
        let ok = unsafe { ffi::mpsgraph_random_op_descriptor_set_max(self.ptr, value) };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed("failed to set random max"))
        }
    }

/// Calls the `MPSGraph` framework counterpart for `min_integer`.
    #[must_use]
    pub fn min_integer(&self) -> isize {
        // SAFETY: `self.ptr` is a live descriptor handle.
        unsafe { ffi::mpsgraph_random_op_descriptor_min_integer(self.ptr) }
    }

/// Calls the `MPSGraph` framework counterpart for `set_min_integer`.
    pub fn set_min_integer(&self, value: isize) -> Result<()> {
        // SAFETY: `self.ptr` is a live descriptor handle.
        let ok = unsafe { ffi::mpsgraph_random_op_descriptor_set_min_integer(self.ptr, value) };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed("failed to set random minInteger"))
        }
    }

/// Calls the `MPSGraph` framework counterpart for `max_integer`.
    #[must_use]
    pub fn max_integer(&self) -> isize {
        // SAFETY: `self.ptr` is a live descriptor handle.
        unsafe { ffi::mpsgraph_random_op_descriptor_max_integer(self.ptr) }
    }

/// Calls the `MPSGraph` framework counterpart for `set_max_integer`.
    pub fn set_max_integer(&self, value: isize) -> Result<()> {
        // SAFETY: `self.ptr` is a live descriptor handle.
        let ok = unsafe { ffi::mpsgraph_random_op_descriptor_set_max_integer(self.ptr, value) };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed("failed to set random maxInteger"))
        }
    }

/// Calls the `MPSGraph` framework counterpart for `mean`.
    #[must_use]
    pub fn mean(&self) -> f32 {
        // SAFETY: `self.ptr` is a live descriptor handle.
        unsafe { ffi::mpsgraph_random_op_descriptor_mean(self.ptr) }
    }

/// Calls the `MPSGraph` framework counterpart for `set_mean`.
    pub fn set_mean(&self, value: f32) -> Result<()> {
        // SAFETY: `self.ptr` is a live descriptor handle.
        let ok = unsafe { ffi::mpsgraph_random_op_descriptor_set_mean(self.ptr, value) };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed("failed to set random mean"))
        }
    }

/// Calls the `MPSGraph` framework counterpart for `standard_deviation`.
    #[must_use]
    pub fn standard_deviation(&self) -> f32 {
        // SAFETY: `self.ptr` is a live descriptor handle.
        unsafe { ffi::mpsgraph_random_op_descriptor_standard_deviation(self.ptr) }
    }

/// Calls the `MPSGraph` framework counterpart for `set_standard_deviation`.
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

/// Calls the `MPSGraph` framework counterpart for `sampling_method`.
    #[must_use]
    pub fn sampling_method(&self) -> u64 {
        // SAFETY: `self.ptr` is a live descriptor handle.
        unsafe { ffi::mpsgraph_random_op_descriptor_sampling_method(self.ptr) }
    }

/// Calls the `MPSGraph` framework counterpart for `set_sampling_method`.
    pub fn set_sampling_method(&self, value: u64) -> Result<()> {
        if value > random_normal_sampling_method::BOX_MULLER {
            return Err(Error::InvalidArgument(
                "unknown MPSGraphRandomNormalSamplingMethod",
            ));
        }
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


fn shape_pointer(shape: &[usize]) -> Result<*const usize> {
    checks::shape_to_dims(shape)?;
    Ok(if shape.is_empty() {
        ptr::null()
    } else {
        shape.as_ptr()
    })
}

impl crate::graph::Graph {
/// Calls the `MPSGraph` framework counterpart for `random_philox_state_seed`.
    pub fn random_philox_state_seed(&self, seed: usize, name: Option<&str>) -> Result<Tensor> {
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_random_philox_state_seed(self.as_ptr(), seed, cstring_ptr(&name))
        };
        self.output(ptr, &[], "MPSGraph did not create the Philox state")
    }

/// Calls the `MPSGraph` framework counterpart for `random_philox_state_counter`.
    pub fn random_philox_state_counter(
        &self,
        counter_low: usize,
        counter_high: usize,
        key: usize,
        name: Option<&str>,
    ) -> Result<Tensor> {
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
        self.output(ptr, &[], "MPSGraph did not create the Philox state")
    }

/// Calls the `MPSGraph` framework counterpart for `random_tensor`.
    pub fn random_tensor(
        &self,
        shape: &[usize],
        descriptor: &RandomOpDescriptor,
        name: Option<&str>,
    ) -> Result<Tensor> {
        let shape_ptr = shape_pointer(shape)?;
        let name = optional_cstring(name);
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
        self.output(ptr, &[], "MPSGraph did not create the random tensor")
    }

/// Calls the `MPSGraph` framework counterpart for `random_tensor_shape_tensor`.
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn random_tensor_shape_tensor(
        &self,
        shape_tensor: &Tensor,
        descriptor: &RandomOpDescriptor,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[shape_tensor])?;
        check_shape_tensor(shape_tensor)?;
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
        self.output(ptr, &[shape_tensor], "MPSGraph did not create the random tensor")
    }

/// Calls the `MPSGraph` framework counterpart for `random_tensor_seed`.
    pub fn random_tensor_seed(
        &self,
        shape: &[usize],
        descriptor: &RandomOpDescriptor,
        seed: usize,
        name: Option<&str>,
    ) -> Result<Tensor> {
        let shape_ptr = shape_pointer(shape)?;
        let name = optional_cstring(name);
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
        self.output(ptr, &[], "MPSGraph did not create the random tensor")
    }

/// Calls the `MPSGraph` framework counterpart for `random_tensor_shape_tensor_seed`.
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn random_tensor_shape_tensor_seed(
        &self,
        shape_tensor: &Tensor,
        descriptor: &RandomOpDescriptor,
        seed: usize,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[shape_tensor])?;
        check_shape_tensor(shape_tensor)?;
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
        self.output(ptr, &[shape_tensor], "MPSGraph did not create the random tensor")
    }

/// Calls the `MPSGraph` framework counterpart for `random_tensor_state`.
    pub fn random_tensor_state(
        &self,
        shape: &[usize],
        descriptor: &RandomOpDescriptor,
        state: &Tensor,
        name: Option<&str>,
    ) -> Result<(Tensor, Tensor)> {
        self.check(&[state])?;
        check_state(state)?;
        let shape_ptr = shape_pointer(shape)?;
        let name = optional_cstring(name);
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
        self.output_pair(box_handle, &[state], "MPSGraph did not create the random tensor")
    }

/// Calls the `MPSGraph` framework counterpart for `random_tensor_shape_tensor_state`.
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn random_tensor_shape_tensor_state(
        &self,
        shape_tensor: &Tensor,
        descriptor: &RandomOpDescriptor,
        state: &Tensor,
        name: Option<&str>,
    ) -> Result<(Tensor, Tensor)> {
        self.check(&[shape_tensor, state])?;
        check_shape_tensor(shape_tensor)?;
        check_state(state)?;
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
        self.output_pair(
            box_handle,
            &[shape_tensor, state],
            "MPSGraph did not create the random tensor",
        )
    }

/// Calls the `MPSGraph` framework counterpart for `dropout`.
    pub fn dropout(&self, tensor: &Tensor, rate: f64, name: Option<&str>) -> Result<Tensor> {
        self.check(&[tensor])?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_dropout(self.as_ptr(), tensor.as_ptr(), rate, cstring_ptr(&name))
        };
        self.output(ptr, &[tensor], "MPSGraph did not create the dropout")
    }

/// Calls the `MPSGraph` framework counterpart for `dropout_tensor`.
    pub fn dropout_tensor(
        &self,
        tensor: &Tensor,
        rate_tensor: &Tensor,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[tensor, rate_tensor])?;
        checks::elementwise(tensor, rate_tensor)?;
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
        self.output(
            ptr,
            &[tensor, rate_tensor],
            "MPSGraph did not create the dropout",
        )
    }
}
