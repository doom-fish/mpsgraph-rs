use crate::data::TensorData;
use crate::error::{Error, Result};
use crate::ffi;
use crate::graph::{take_tensor_handles, Tensor};
use apple_metal::MetalDevice;
use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;
use core::ptr;

fn release_handle(ptr: &mut *mut c_void) {
    if !ptr.is_null() {
        // SAFETY: `ptr` is a +1 retained Swift/ObjC object pointer owned by this wrapper.
        unsafe { ffi::mpsgraph_object_release(*ptr) };
        *ptr = ptr::null_mut();
    }
}

fn copy_optional_signed_shape(
    handle: *mut c_void,
    copy_shape: unsafe extern "C" fn(*mut c_void, *mut isize, usize) -> isize,
) -> Option<Vec<isize>> {
    let mut shape = Vec::new();
    loop {
        // SAFETY: `shape` has space for `shape.len()` elements, and the Swift shim writes only a shape that fits.
        let count = unsafe { copy_shape(handle, shape.as_mut_ptr(), shape.len()) };
        let count = usize::try_from(count).ok()?;
        if count <= shape.len() {
            shape.truncate(count);
            return Some(shape);
        }
        shape.resize(count, 0);
    }
}

pub(crate) fn collect_tensor_data_array_box(handle: *mut c_void) -> Vec<TensorData> {
    if handle.is_null() {
        return Vec::new();
    }

    // SAFETY: `handle` is a retained tensor-data-array box created by the Swift bridge.
    let len = unsafe { ffi::mpsgraph_tensor_data_array_box_len(handle) };
    let mut values = Vec::with_capacity(len);
    for index in 0..len {
        // SAFETY: indices are bounded by the just-read length.
        let value = unsafe { ffi::mpsgraph_tensor_data_array_box_get(handle, index) };
        if !value.is_null() {
            values.push(TensorData::from_raw(value));
        }
    }
    let mut box_handle = handle;
    release_handle(&mut box_handle);
    values
}

pub(crate) fn collect_shaped_type_array_box(handle: *mut c_void) -> Vec<ShapedType> {
    if handle.is_null() {
        return Vec::new();
    }

    // SAFETY: `handle` is a retained shaped-type-array box created by the Swift bridge.
    let len = unsafe { ffi::mpsgraph_shaped_type_array_box_len(handle) };
    let mut values = Vec::with_capacity(len);
    for index in 0..len {
        // SAFETY: indices are bounded by the just-read length.
        let value = unsafe { ffi::mpsgraph_shaped_type_array_box_get(handle, index) };
        if !value.is_null() {
            values.push(ShapedType {
                ptr: value,
                _not_sync: PhantomData,
            });
        }
    }
    let mut box_handle = handle;
    release_handle(&mut box_handle);
    values
}

/// `MPSGraphDeviceType` constants.
pub mod graph_device_type {
/// Mirrors the `MPSGraph` framework constant `METAL`.
    pub const METAL: u32 = 0;
}

/// Owned wrapper for `MPSGraphDevice`.
pub struct GraphDevice {
    ptr: *mut c_void,
}

unsafe impl Send for GraphDevice {}
unsafe impl Sync for GraphDevice {}

impl Drop for GraphDevice {
    fn drop(&mut self) {
        release_handle(&mut self.ptr);
    }
}

impl GraphDevice {
    /// Create a graph device from an existing Metal device.
    #[must_use]
    pub fn from_metal_device(device: &MetalDevice) -> Option<Self> {
        // SAFETY: `device` remains valid for the duration of the bridge call.
        let ptr = unsafe { ffi::mpsgraph_device_new_with_metal_device(device.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

/// Mirrors the `MPSGraph` framework constant `fn`.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Return the underlying `MPSGraphDeviceType` raw value.
    #[must_use]
    pub fn device_type(&self) -> u32 {
        // SAFETY: `self.ptr` is a live graph-device handle.
        unsafe { ffi::mpsgraph_device_type(self.ptr) }
    }
}

/// Owned wrapper for `MPSGraphShapedType`.
pub struct ShapedType {
    ptr: *mut c_void,
    _not_sync: PhantomData<Cell<()>>,
}

unsafe impl Send for ShapedType {}

impl Drop for ShapedType {
    fn drop(&mut self) {
        release_handle(&mut self.ptr);
    }
}

impl ShapedType {
    /// Create a shaped type from an optional shape and `MPSDataType` raw value.
    #[must_use]
    pub fn new(shape: Option<&[isize]>, data_type: u32) -> Option<Self> {
        let (shape_ptr, shape_len) =
            shape.map_or((ptr::null(), 0), |shape| (shape.as_ptr(), shape.len()));
        // SAFETY: the optional slice lives for the duration of the call.
        let ptr = unsafe { ffi::mpsgraph_shaped_type_new(shape_ptr, shape_len, data_type) };
        if ptr.is_null() {
            None
        } else {
            Some(Self {
                ptr,
                _not_sync: PhantomData,
            })
        }
    }

    #[must_use]
    pub(crate) const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Return the optional tensor shape. `None` corresponds to an unranked shape.
    #[must_use]
    pub fn shape(&self) -> Option<Vec<isize>> {
        copy_optional_signed_shape(self.ptr, ffi::mpsgraph_shaped_type_copy_shape)
    }

    /// Return the underlying `MPSDataType` raw value.
    #[must_use]
    pub fn data_type(&self) -> u32 {
        // SAFETY: `self.ptr` is a live shaped-type handle.
        unsafe { ffi::mpsgraph_shaped_type_data_type(self.ptr) }
    }

    /// Replace the shape metadata for this shaped type.
    pub fn set_shape(&self, shape: Option<&[isize]>) -> Result<()> {
        let (shape_ptr, shape_len) =
            shape.map_or((ptr::null(), 0), |shape| (shape.as_ptr(), shape.len()));
        // SAFETY: the optional slice lives for the duration of the call.
        let ok = unsafe { ffi::mpsgraph_shaped_type_set_shape(self.ptr, shape_ptr, shape_len) };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed("failed to set shaped type shape"))
        }
    }

    /// Replace the data-type metadata for this shaped type.
    pub fn set_data_type(&self, data_type: u32) -> Result<()> {
        // SAFETY: `self.ptr` is a live shaped-type handle.
        let ok = unsafe { ffi::mpsgraph_shaped_type_set_data_type(self.ptr, data_type) };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed(
                "failed to set shaped type data type",
            ))
        }
    }

    /// Compare two shaped types using `MPSGraphShapedType.isEqual(to:)`.
    #[must_use]
    pub fn is_equal(&self, other: Option<&Self>) -> bool {
        let other_ptr = other.map_or(ptr::null_mut(), Self::as_ptr);
        // SAFETY: all handles stay alive for the duration of the call.
        unsafe { ffi::mpsgraph_shaped_type_is_equal(self.ptr, other_ptr) }
    }
}

/// Owned wrapper for `MPSGraphOperation`.
pub struct Operation {
    ptr: *mut c_void,
    scope: u64,
    inputs: Vec<usize>,
}

unsafe impl Send for Operation {}
unsafe impl Sync for Operation {}

impl Drop for Operation {
    fn drop(&mut self) {
        release_handle(&mut self.ptr);
    }
}

impl Operation {
    #[must_use]
    pub(crate) const fn from_raw(ptr: *mut c_void, scope: u64, inputs: Vec<usize>) -> Self {
        Self { ptr, scope, inputs }
    }

    pub(crate) const fn scope(&self) -> u64 {
        self.scope
    }

    pub(crate) fn inputs(&self) -> &[usize] {
        &self.inputs
    }

/// Mirrors the `MPSGraph` framework constant `fn`.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Tensor {
    /// Return the optional symbolic tensor shape.
    #[must_use]
    pub fn shape(&self) -> Option<Vec<isize>> {
        copy_optional_signed_shape(self.as_ptr(), ffi::mpsgraph_tensor_copy_shape)
    }

    /// Return the tensor's `MPSDataType` raw value.
    #[must_use]
    pub fn data_type(&self) -> u32 {
        // SAFETY: `self` owns a live tensor handle.
        unsafe { ffi::mpsgraph_tensor_data_type(self.as_ptr()) }
    }

    /// Return the operation that produced this tensor.
    #[must_use]
    pub fn operation(&self) -> Option<Operation> {
        // SAFETY: `self` owns a live tensor handle.
        let ptr = unsafe { ffi::mpsgraph_tensor_operation(self.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Operation {
                ptr,
                scope: self.scope(),
                inputs: vec![self.as_ptr() as usize],
            })
        }
    }
}

impl TensorData {
    /// Return the graph-device type that backs this tensor data.
    #[must_use]
    pub fn graph_device_type(&self) -> Option<u32> {
        // SAFETY: `self` owns a live tensor-data handle.
        let ptr = unsafe { ffi::mpsgraph_tensor_data_device(self.as_ptr()) };
        if ptr.is_null() {
            return None;
        }
        let device = GraphDevice { ptr };
        Some(device.device_type())
    }
}

pub(crate) fn collect_owned_tensors(handle: *mut c_void, scope: u64) -> Vec<Tensor> {
    take_tensor_handles(handle)
        .into_iter()
        .filter(|tensor| !tensor.is_null())
        .map(|tensor| Tensor::from_raw(tensor, scope))
        .collect()
}
