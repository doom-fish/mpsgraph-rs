use crate::error::{Error, Result};
use crate::ffi;
use crate::graph::{data_type, data_type_size};
use apple_metal::{MetalBuffer, MetalDevice};
use core::ffi::c_void;
use core::ptr;

fn checked_byte_len(shape: &[usize], data_type: u32) -> Option<usize> {
    let element_size = data_type_size(data_type)?;
    shape
        .iter()
        .try_fold(element_size, |acc, dimension| acc.checked_mul(*dimension))
}

/// Safe owner for an Objective-C `MPSGraphTensorData`.
pub struct TensorData {
    ptr: *mut c_void,
}

unsafe impl Send for TensorData {}
unsafe impl Sync for TensorData {}

impl Drop for TensorData {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `ptr` is a +1 retained Swift/ObjC object pointer owned by this wrapper.
            unsafe { ffi::mpsgraph_object_release(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

impl TensorData {
    pub(crate) const fn from_raw(ptr: *mut c_void) -> Self {
        Self { ptr }
    }

    /// Build tensor data by copying CPU bytes onto the given Metal device.
    #[must_use]
    pub fn from_bytes(
        device: &MetalDevice,
        bytes: &[u8],
        shape: &[usize],
        data_type: u32,
    ) -> Option<Self> {
        let expected = checked_byte_len(shape, data_type)?;
        if bytes.len() != expected {
            return None;
        }

        // SAFETY: The device handle and byte slice stay valid for the duration of the FFI call.
        let ptr = unsafe {
            ffi::mpsgraph_tensor_data_new_with_bytes(
                device.as_ptr(),
                bytes.as_ptr().cast(),
                bytes.len(),
                shape.as_ptr(),
                shape.len(),
                data_type,
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Build tensor data from a contiguous `f32` slice.
    #[must_use]
    pub fn from_f32_slice(device: &MetalDevice, values: &[f32], shape: &[usize]) -> Option<Self> {
        // SAFETY: `values` is a contiguous slice of `f32` that may be viewed as bytes.
        let bytes = unsafe {
            core::slice::from_raw_parts(
                values.as_ptr().cast::<u8>(),
                core::mem::size_of_val(values),
            )
        };
        Self::from_bytes(device, bytes, shape, data_type::FLOAT32)
    }

    /// Alias an existing `MTLBuffer` as tensor data.
    #[must_use]
    pub fn from_buffer(buffer: &MetalBuffer, shape: &[usize], data_type: u32) -> Option<Self> {
        // SAFETY: The buffer handle remains valid for the duration of the FFI call.
        let ptr = unsafe {
            ffi::mpsgraph_tensor_data_new_with_buffer(
                buffer.as_ptr(),
                shape.as_ptr(),
                shape.len(),
                data_type,
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    #[must_use]
    pub fn data_type(&self) -> u32 {
        // SAFETY: `self.ptr` is a valid `MPSGraphTensorData` while `self` is alive.
        unsafe { ffi::mpsgraph_tensor_data_data_type(self.ptr) }
    }

    #[must_use]
    pub fn shape(&self) -> Vec<usize> {
        // SAFETY: `self.ptr` is a valid `MPSGraphTensorData` while `self` is alive.
        let len = unsafe { ffi::mpsgraph_tensor_data_shape_len(self.ptr) };
        let mut shape = vec![0_usize; len];
        if len > 0 {
            // SAFETY: `shape` has capacity for exactly `len` values and the tensor data outlives the call.
            unsafe { ffi::mpsgraph_tensor_data_copy_shape(self.ptr, shape.as_mut_ptr()) };
        }
        shape
    }

    #[must_use]
    pub fn element_count(&self) -> usize {
        self.shape().iter().product()
    }

    pub fn byte_len(&self) -> Result<usize> {
        checked_byte_len(&self.shape(), self.data_type())
            .ok_or_else(|| Error::UnsupportedDataType(self.data_type()))
    }

    pub fn read_bytes(&self) -> Result<Vec<u8>> {
        let byte_len = self.byte_len()?;
        let mut bytes = vec![0_u8; byte_len];
        // SAFETY: `bytes` is valid for writes of `byte_len` bytes and the tensor data outlives the call.
        let ok = unsafe {
            ffi::mpsgraph_tensor_data_read_bytes(self.ptr, bytes.as_mut_ptr().cast(), byte_len)
        };
        if ok {
            Ok(bytes)
        } else {
            Err(Error::OperationFailed("failed to read tensor data"))
        }
    }

    pub fn read_f32(&self) -> Result<Vec<f32>> {
        if self.data_type() != data_type::FLOAT32 {
            return Err(Error::UnsupportedDataType(self.data_type()));
        }

        let byte_len = self.byte_len()?;
        let mut values = vec![0.0_f32; byte_len / core::mem::size_of::<f32>()];
        // SAFETY: `values` is a contiguous `Vec<f32>` with `byte_len` bytes of backing storage.
        let ok = unsafe {
            ffi::mpsgraph_tensor_data_read_bytes(self.ptr, values.as_mut_ptr().cast(), byte_len)
        };
        if ok {
            Ok(values)
        } else {
            Err(Error::OperationFailed("failed to read tensor data"))
        }
    }
}
