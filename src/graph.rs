use crate::data::TensorData;
use crate::error::{Error, Result};
use crate::ffi;
use apple_metal::{CommandQueue, MetalDevice};
use core::ffi::{c_char, c_void};
use core::ptr;
use std::ffi::CString;

/// Selected `MPSDataType` constants useful for graph inputs and outputs.
pub mod data_type {
/// Mirrors the `MPSGraph` framework constant `INVALID`.
    pub const INVALID: u32 = 0;
/// Mirrors the `MPSGraph` framework constant `FLOAT32`.
    pub const FLOAT32: u32 = 0x1000_0020;
/// Mirrors the `MPSGraph` framework constant `FLOAT16`.
    pub const FLOAT16: u32 = 0x1000_0010;
/// Mirrors the `MPSGraph` framework constant `INT8`.
    pub const INT8: u32 = 0x2000_0008;
/// Mirrors the `MPSGraph` framework constant `INT16`.
    pub const INT16: u32 = 0x2000_0010;
/// Mirrors the `MPSGraph` framework constant `INT32`.
    pub const INT32: u32 = 0x2000_0020;
/// Mirrors the `MPSGraph` framework constant `INT64`.
    pub const INT64: u32 = 0x2000_0040;
/// Mirrors the `MPSGraph` framework constant `UINT8`.
    pub const UINT8: u32 = 0x0000_0008;
/// Mirrors the `MPSGraph` framework constant `UINT16`.
    pub const UINT16: u32 = 0x0000_0010;
/// Mirrors the `MPSGraph` framework constant `UINT32`.
    pub const UINT32: u32 = 0x0000_0020;
/// Mirrors the `MPSGraph` framework constant `UINT64`.
    pub const UINT64: u32 = 0x0000_0040;
/// Mirrors the `MPSGraph` framework constant `BOOL`.
    pub const BOOL: u32 = 0x8000_0008;
/// Mirrors the `MPSGraph` framework constant `UNORM8`.
    pub const UNORM8: u32 = 0x4000_0008;
}

/// Return the byte width of a supported `MPSDataType`.
#[must_use]
pub const fn data_type_size(data_type: u32) -> Option<usize> {
    match data_type {
        data_type::FLOAT16 | data_type::INT16 | data_type::UINT16 => Some(2),
        data_type::FLOAT32 | data_type::INT32 | data_type::UINT32 => Some(4),
        data_type::INT64 | data_type::UINT64 => Some(8),
        data_type::INT8 | data_type::UINT8 | data_type::BOOL | data_type::UNORM8 => Some(1),
        _ => None,
    }
}

/// `MPSGraphTensorNamedDataLayout` constants.
pub mod tensor_named_data_layout {
/// Mirrors the `MPSGraph` framework constant `NCHW`.
    pub const NCHW: usize = 0;
/// Mirrors the `MPSGraph` framework constant `NHWC`.
    pub const NHWC: usize = 1;
/// Mirrors the `MPSGraph` framework constant `OIHW`.
    pub const OIHW: usize = 2;
/// Mirrors the `MPSGraph` framework constant `HWIO`.
    pub const HWIO: usize = 3;
/// Mirrors the `MPSGraph` framework constant `CHW`.
    pub const CHW: usize = 4;
/// Mirrors the `MPSGraph` framework constant `HWC`.
    pub const HWC: usize = 5;
/// Mirrors the `MPSGraph` framework constant `HW`.
    pub const HW: usize = 6;
/// Mirrors the `MPSGraph` framework constant `NCDHW`.
    pub const NCDHW: usize = 7;
/// Mirrors the `MPSGraph` framework constant `NDHWC`.
    pub const NDHWC: usize = 8;
/// Mirrors the `MPSGraph` framework constant `OIDHW`.
    pub const OIDHW: usize = 9;
/// Mirrors the `MPSGraph` framework constant `DHWIO`.
    pub const DHWIO: usize = 10;
}

/// `MPSGraphPaddingStyle` constants.
pub mod padding_style {
/// Mirrors the `MPSGraph` framework constant `EXPLICIT`.
    pub const EXPLICIT: usize = 0;
/// Mirrors the `MPSGraph` framework constant `TF_VALID`.
    pub const TF_VALID: usize = 1;
/// Mirrors the `MPSGraph` framework constant `TF_SAME`.
    pub const TF_SAME: usize = 2;
/// Mirrors the `MPSGraph` framework constant `EXPLICIT_OFFSET`.
    pub const EXPLICIT_OFFSET: usize = 3;
/// Mirrors the `MPSGraph` framework constant `ONNX_SAME_LOWER`.
    pub const ONNX_SAME_LOWER: usize = 4;
}

/// `MPSGraphPaddingMode` constants.
pub mod padding_mode {
/// Mirrors the `MPSGraph` framework constant `CONSTANT`.
    pub const CONSTANT: isize = 0;
/// Mirrors the `MPSGraph` framework constant `REFLECT`.
    pub const REFLECT: isize = 1;
/// Mirrors the `MPSGraph` framework constant `SYMMETRIC`.
    pub const SYMMETRIC: isize = 2;
/// Mirrors the `MPSGraph` framework constant `CLAMP_TO_EDGE`.
    pub const CLAMP_TO_EDGE: isize = 3;
/// Mirrors the `MPSGraph` framework constant `ZERO`.
    pub const ZERO: isize = 4;
/// Mirrors the `MPSGraph` framework constant `PERIODIC`.
    pub const PERIODIC: isize = 5;
/// Mirrors the `MPSGraph` framework constant `ANTI_PERIODIC`.
    pub const ANTI_PERIODIC: isize = 6;
}

macro_rules! opaque_handle {
    ($name:ident) => {
/// Mirrors the `MPSGraph` framework counterpart for this type.
        pub struct $name {
            ptr: *mut c_void,
        }

        unsafe impl Send for $name {}
        unsafe impl Sync for $name {}

        impl Drop for $name {
            fn drop(&mut self) {
                if !self.ptr.is_null() {
                    // SAFETY: `ptr` is a +1 retained Swift/ObjC object pointer owned by this wrapper.
                    unsafe { ffi::mpsgraph_object_release(self.ptr) };
                    self.ptr = ptr::null_mut();
                }
            }
        }

        impl $name {
/// Mirrors the `MPSGraph` framework constant `fn`.
            #[must_use]
            pub const fn as_ptr(&self) -> *mut c_void {
                self.ptr
            }
        }
    };
}

fn checked_byte_len(shape: &[usize], data_type: u32) -> Option<usize> {
    let element_size = data_type_size(data_type)?;
    shape
        .iter()
        .try_fold(element_size, |acc, dimension| acc.checked_mul(*dimension))
}

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
        Some(Tensor { ptr })
    }
}

fn wrap_tensor_data_results(
    handles: Vec<*mut c_void>,
    message: &'static str,
) -> Result<Vec<TensorData>> {
    let mut results = Vec::with_capacity(handles.len());
    for handle in handles {
        if handle.is_null() {
            return Err(Error::OperationFailed(message));
        }
        results.push(TensorData::from_raw(handle));
    }
    Ok(results)
}

macro_rules! impl_binary_tensor_op {
    ($fn_name:ident, $ffi_name:ident) => {
/// Calls the `MPSGraph` framework counterpart for this method.
        #[must_use]
        pub fn $fn_name(
            &self,
            primary: &Tensor,
            secondary: &Tensor,
            name: Option<&str>,
        ) -> Option<Tensor> {
            let name = optional_cstring(name);
            // SAFETY: All pointers originate from safe wrappers and remain alive for the duration of the call.
            let ptr = unsafe {
                ffi::$ffi_name(
                    self.ptr,
                    primary.as_ptr(),
                    secondary.as_ptr(),
                    cstring_ptr(&name),
                )
            };
            wrap_tensor(ptr)
        }
    };
}

macro_rules! impl_unary_tensor_op {
    ($fn_name:ident, $ffi_name:ident) => {
/// Calls the `MPSGraph` framework counterpart for this method.
        #[must_use]
        pub fn $fn_name(&self, tensor: &Tensor, name: Option<&str>) -> Option<Tensor> {
            let name = optional_cstring(name);
            // SAFETY: All pointers originate from safe wrappers and remain alive for the duration of the call.
            let ptr = unsafe { ffi::$ffi_name(self.ptr, tensor.as_ptr(), cstring_ptr(&name)) };
            wrap_tensor(ptr)
        }
    };
}

macro_rules! impl_axes_tensor_op {
    ($fn_name:ident, $ffi_name:ident) => {
/// Calls the `MPSGraph` framework counterpart for this method.
        #[must_use]
        pub fn $fn_name(
            &self,
            tensor: &Tensor,
            axes: &[usize],
            name: Option<&str>,
        ) -> Option<Tensor> {
            let name = optional_cstring(name);
            // SAFETY: All pointers originate from safe wrappers and remain alive for the duration of the call.
            let ptr = unsafe {
                ffi::$ffi_name(
                    self.ptr,
                    tensor.as_ptr(),
                    axes.as_ptr(),
                    axes.len(),
                    cstring_ptr(&name),
                )
            };
            wrap_tensor(ptr)
        }
    };
}

/// Ordered placeholder feed pairing used for graph execution.
#[derive(Clone, Copy)]
pub struct Feed<'a> {
/// Mirrors the `MPSGraph` framework property for `tensor`.
    pub tensor: &'a Tensor,
/// Mirrors the `MPSGraph` framework property for `data`.
    pub data: &'a TensorData,
}

impl<'a> Feed<'a> {
/// Mirrors the `MPSGraph` framework constant `fn`.
    #[must_use]
    pub const fn new(tensor: &'a Tensor, data: &'a TensorData) -> Self {
        Self { tensor, data }
    }
}

/// Feed metadata used to compile a graph into an executable.
#[derive(Clone, Copy)]
pub struct FeedDescription<'a> {
/// Mirrors the `MPSGraph` framework property for `tensor`.
    pub tensor: &'a Tensor,
/// Mirrors the `MPSGraph` framework property for `shape`.
    pub shape: &'a [usize],
/// Mirrors the `MPSGraph` framework property for `data_type`.
    pub data_type: u32,
}

impl<'a> FeedDescription<'a> {
/// Mirrors the `MPSGraph` framework constant `fn`.
    #[must_use]
    pub const fn new(tensor: &'a Tensor, shape: &'a [usize], data_type: u32) -> Self {
        Self {
            tensor,
            shape,
            data_type,
        }
    }
}

/// Plain-Rust configuration for `MPSGraphConvolution2DOpDescriptor`.
#[derive(Debug, Clone, Copy)]
pub struct Convolution2DDescriptorInfo {
/// Mirrors the `MPSGraph` framework property for `stride_in_x`.
    pub stride_in_x: usize,
/// Mirrors the `MPSGraph` framework property for `stride_in_y`.
    pub stride_in_y: usize,
/// Mirrors the `MPSGraph` framework property for `dilation_rate_in_x`.
    pub dilation_rate_in_x: usize,
/// Mirrors the `MPSGraph` framework property for `dilation_rate_in_y`.
    pub dilation_rate_in_y: usize,
/// Mirrors the `MPSGraph` framework property for `groups`.
    pub groups: usize,
/// Mirrors the `MPSGraph` framework property for `padding_left`.
    pub padding_left: usize,
/// Mirrors the `MPSGraph` framework property for `padding_right`.
    pub padding_right: usize,
/// Mirrors the `MPSGraph` framework property for `padding_top`.
    pub padding_top: usize,
/// Mirrors the `MPSGraph` framework property for `padding_bottom`.
    pub padding_bottom: usize,
/// Mirrors the `MPSGraph` framework property for `padding_style`.
    pub padding_style: usize,
/// Mirrors the `MPSGraph` framework property for `data_layout`.
    pub data_layout: usize,
/// Mirrors the `MPSGraph` framework property for `weights_layout`.
    pub weights_layout: usize,
}

impl Default for Convolution2DDescriptorInfo {
    fn default() -> Self {
        Self {
            stride_in_x: 1,
            stride_in_y: 1,
            dilation_rate_in_x: 1,
            dilation_rate_in_y: 1,
            groups: 1,
            padding_left: 0,
            padding_right: 0,
            padding_top: 0,
            padding_bottom: 0,
            padding_style: padding_style::EXPLICIT,
            data_layout: tensor_named_data_layout::NHWC,
            weights_layout: tensor_named_data_layout::HWIO,
        }
    }
}

opaque_handle!(Convolution2DDescriptor);
impl Convolution2DDescriptor {
/// Calls the `MPSGraph` framework counterpart for `new`.
    #[must_use]
    pub fn new(info: Convolution2DDescriptorInfo) -> Option<Self> {
        // SAFETY: All scalar configuration values are POD.
        let ptr = unsafe {
            ffi::mpsgraph_convolution2d_descriptor_new(
                info.stride_in_x,
                info.stride_in_y,
                info.dilation_rate_in_x,
                info.dilation_rate_in_y,
                info.groups,
                info.padding_left,
                info.padding_right,
                info.padding_top,
                info.padding_bottom,
                info.padding_style,
                info.data_layout,
                info.weights_layout,
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }
}

/// Plain-Rust configuration for `MPSGraphPooling2DOpDescriptor`.
#[derive(Debug, Clone, Copy)]
pub struct Pooling2DDescriptorInfo {
/// Mirrors the `MPSGraph` framework property for `kernel_width`.
    pub kernel_width: usize,
/// Mirrors the `MPSGraph` framework property for `kernel_height`.
    pub kernel_height: usize,
/// Mirrors the `MPSGraph` framework property for `stride_in_x`.
    pub stride_in_x: usize,
/// Mirrors the `MPSGraph` framework property for `stride_in_y`.
    pub stride_in_y: usize,
/// Mirrors the `MPSGraph` framework property for `dilation_rate_in_x`.
    pub dilation_rate_in_x: usize,
/// Mirrors the `MPSGraph` framework property for `dilation_rate_in_y`.
    pub dilation_rate_in_y: usize,
/// Mirrors the `MPSGraph` framework property for `padding_left`.
    pub padding_left: usize,
/// Mirrors the `MPSGraph` framework property for `padding_right`.
    pub padding_right: usize,
/// Mirrors the `MPSGraph` framework property for `padding_top`.
    pub padding_top: usize,
/// Mirrors the `MPSGraph` framework property for `padding_bottom`.
    pub padding_bottom: usize,
/// Mirrors the `MPSGraph` framework property for `padding_style`.
    pub padding_style: usize,
/// Mirrors the `MPSGraph` framework property for `data_layout`.
    pub data_layout: usize,
}

impl Pooling2DDescriptorInfo {
/// Mirrors the `MPSGraph` framework constant `fn`.
    #[must_use]
    pub const fn new(kernel_width: usize, kernel_height: usize) -> Self {
        Self {
            kernel_width,
            kernel_height,
            stride_in_x: 1,
            stride_in_y: 1,
            dilation_rate_in_x: 1,
            dilation_rate_in_y: 1,
            padding_left: 0,
            padding_right: 0,
            padding_top: 0,
            padding_bottom: 0,
            padding_style: padding_style::EXPLICIT,
            data_layout: tensor_named_data_layout::NHWC,
        }
    }
}

opaque_handle!(Pooling2DDescriptor);
impl Pooling2DDescriptor {
/// Calls the `MPSGraph` framework counterpart for `new`.
    #[must_use]
    pub fn new(info: Pooling2DDescriptorInfo) -> Option<Self> {
        // SAFETY: All scalar configuration values are POD.
        let ptr = unsafe {
            ffi::mpsgraph_pooling2d_descriptor_new(
                info.kernel_width,
                info.kernel_height,
                info.stride_in_x,
                info.stride_in_y,
                info.dilation_rate_in_x,
                info.dilation_rate_in_y,
                info.padding_left,
                info.padding_right,
                info.padding_top,
                info.padding_bottom,
                info.padding_style,
                info.data_layout,
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }
}

opaque_handle!(Graph);
opaque_handle!(Tensor);

impl Tensor {
    pub(crate) const fn from_raw(ptr: *mut c_void) -> Self {
        Self { ptr }
    }
}

impl Graph {
/// Calls the `MPSGraph` framework counterpart for `new`.
    #[must_use]
    pub fn new() -> Option<Self> {
        // SAFETY: Pure constructor with no inputs.
        let ptr = unsafe { ffi::mpsgraph_graph_new() };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

/// Calls the `MPSGraph` framework counterpart for `placeholder`.
    #[must_use]
    pub fn placeholder(
        &self,
        shape: Option<&[usize]>,
        data_type: u32,
        name: Option<&str>,
    ) -> Option<Tensor> {
        let name = optional_cstring(name);
        let (shape_ptr, shape_len) =
            shape.map_or((ptr::null(), 0), |shape| (shape.as_ptr(), shape.len()));

        // SAFETY: All pointers originate from safe wrappers and remain alive for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_placeholder(
                self.ptr,
                shape_ptr,
                shape_len,
                data_type,
                cstring_ptr(&name),
            )
        };
        wrap_tensor(ptr)
    }

/// Calls the `MPSGraph` framework counterpart for `constant_bytes`.
    #[must_use]
    pub fn constant_bytes(&self, data: &[u8], shape: &[usize], data_type: u32) -> Option<Tensor> {
        let expected = checked_byte_len(shape, data_type)?;
        if data.len() != expected {
            return None;
        }

        // SAFETY: The byte slice remains valid for the duration of the FFI call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_constant_data(
                self.ptr,
                data.as_ptr().cast(),
                data.len(),
                shape.as_ptr(),
                shape.len(),
                data_type,
            )
        };
        wrap_tensor(ptr)
    }

/// Calls the `MPSGraph` framework counterpart for `constant_f32_slice`.
    #[must_use]
    pub fn constant_f32_slice(&self, values: &[f32], shape: &[usize]) -> Option<Tensor> {
        // SAFETY: `values` is a contiguous slice of `f32` that may be viewed as bytes.
        let bytes = unsafe {
            core::slice::from_raw_parts(
                values.as_ptr().cast::<u8>(),
                core::mem::size_of_val(values),
            )
        };
        self.constant_bytes(bytes, shape, data_type::FLOAT32)
    }

/// Calls the `MPSGraph` framework counterpart for `constant_scalar`.
    #[must_use]
    pub fn constant_scalar(&self, scalar: f64, data_type: u32) -> Option<Tensor> {
        // SAFETY: Pure constructor over scalar inputs.
        let ptr = unsafe { ffi::mpsgraph_graph_constant_scalar(self.ptr, scalar, data_type) };
        wrap_tensor(ptr)
    }

/// Calls the `MPSGraph` framework counterpart for `constant_scalar_shaped`.
    #[must_use]
    pub fn constant_scalar_shaped(
        &self,
        scalar: f64,
        shape: &[usize],
        data_type: u32,
    ) -> Option<Tensor> {
        // SAFETY: Shape slice stays valid for the duration of the FFI call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_constant_scalar_shaped(
                self.ptr,
                scalar,
                shape.as_ptr(),
                shape.len(),
                data_type,
            )
        };
        wrap_tensor(ptr)
    }

    impl_binary_tensor_op!(addition, mpsgraph_graph_addition);
    impl_binary_tensor_op!(subtraction, mpsgraph_graph_subtraction);
    impl_binary_tensor_op!(multiplication, mpsgraph_graph_multiplication);
    impl_binary_tensor_op!(division, mpsgraph_graph_division);
    impl_binary_tensor_op!(matrix_multiplication, mpsgraph_graph_matrix_multiplication);
    impl_unary_tensor_op!(relu, mpsgraph_graph_relu);
    impl_unary_tensor_op!(sigmoid, mpsgraph_graph_sigmoid);
    impl_axes_tensor_op!(reduction_sum, mpsgraph_graph_reduction_sum);
    impl_axes_tensor_op!(reduction_maximum, mpsgraph_graph_reduction_maximum);
    impl_axes_tensor_op!(reduction_minimum, mpsgraph_graph_reduction_minimum);
    impl_axes_tensor_op!(mean, mpsgraph_graph_mean);

/// Calls the `MPSGraph` framework counterpart for `softmax`.
    #[must_use]
    pub fn softmax(&self, tensor: &Tensor, axis: isize, name: Option<&str>) -> Option<Tensor> {
        let name = optional_cstring(name);
        // SAFETY: All pointers originate from safe wrappers and remain alive for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_softmax(self.ptr, tensor.as_ptr(), axis, cstring_ptr(&name))
        };
        wrap_tensor(ptr)
    }

/// Calls the `MPSGraph` framework counterpart for `reshape`.
    #[must_use]
    pub fn reshape(&self, tensor: &Tensor, shape: &[usize], name: Option<&str>) -> Option<Tensor> {
        let name = optional_cstring(name);
        // SAFETY: All pointers originate from safe wrappers and remain alive for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_reshape(
                self.ptr,
                tensor.as_ptr(),
                shape.as_ptr(),
                shape.len(),
                cstring_ptr(&name),
            )
        };
        wrap_tensor(ptr)
    }

/// Calls the `MPSGraph` framework counterpart for `transpose`.
    #[must_use]
    pub fn transpose(
        &self,
        tensor: &Tensor,
        permutation: &[usize],
        name: Option<&str>,
    ) -> Option<Tensor> {
        let name = optional_cstring(name);
        // SAFETY: All pointers originate from safe wrappers and remain alive for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_transpose(
                self.ptr,
                tensor.as_ptr(),
                permutation.as_ptr(),
                permutation.len(),
                cstring_ptr(&name),
            )
        };
        wrap_tensor(ptr)
    }

/// Calls the `MPSGraph` framework counterpart for `slice`.
    #[must_use]
    pub fn slice(
        &self,
        tensor: &Tensor,
        dimension: usize,
        start: isize,
        length: isize,
        name: Option<&str>,
    ) -> Option<Tensor> {
        let name = optional_cstring(name);
        // SAFETY: All pointers originate from safe wrappers and remain alive for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_slice(
                self.ptr,
                tensor.as_ptr(),
                dimension,
                start,
                length,
                cstring_ptr(&name),
            )
        };
        wrap_tensor(ptr)
    }

/// Calls the `MPSGraph` framework counterpart for `broadcast`.
    #[must_use]
    pub fn broadcast(
        &self,
        tensor: &Tensor,
        shape: &[usize],
        name: Option<&str>,
    ) -> Option<Tensor> {
        let name = optional_cstring(name);
        // SAFETY: All pointers originate from safe wrappers and remain alive for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_broadcast(
                self.ptr,
                tensor.as_ptr(),
                shape.as_ptr(),
                shape.len(),
                cstring_ptr(&name),
            )
        };
        wrap_tensor(ptr)
    }

/// Calls the `MPSGraph` framework counterpart for `convolution2d`.
    #[must_use]
    pub fn convolution2d(
        &self,
        source: &Tensor,
        weights: &Tensor,
        descriptor: &Convolution2DDescriptor,
        name: Option<&str>,
    ) -> Option<Tensor> {
        let name = optional_cstring(name);
        // SAFETY: All pointers originate from safe wrappers and remain alive for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_convolution2d(
                self.ptr,
                source.as_ptr(),
                weights.as_ptr(),
                descriptor.as_ptr(),
                cstring_ptr(&name),
            )
        };
        wrap_tensor(ptr)
    }

/// Calls the `MPSGraph` framework counterpart for `max_pooling2d`.
    #[must_use]
    pub fn max_pooling2d(
        &self,
        source: &Tensor,
        descriptor: &Pooling2DDescriptor,
        name: Option<&str>,
    ) -> Option<Tensor> {
        let name = optional_cstring(name);
        // SAFETY: All pointers originate from safe wrappers and remain alive for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_max_pooling2d(
                self.ptr,
                source.as_ptr(),
                descriptor.as_ptr(),
                cstring_ptr(&name),
            )
        };
        wrap_tensor(ptr)
    }

/// Calls the `MPSGraph` framework counterpart for `normalize`.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn normalize(
        &self,
        tensor: &Tensor,
        mean: &Tensor,
        variance: &Tensor,
        gamma: Option<&Tensor>,
        beta: Option<&Tensor>,
        epsilon: f32,
        name: Option<&str>,
    ) -> Option<Tensor> {
        let name = optional_cstring(name);
        let gamma_ptr = gamma.map_or(ptr::null_mut(), Tensor::as_ptr);
        let beta_ptr = beta.map_or(ptr::null_mut(), Tensor::as_ptr);
        // SAFETY: All pointers originate from safe wrappers and remain alive for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_normalize(
                self.ptr,
                tensor.as_ptr(),
                mean.as_ptr(),
                variance.as_ptr(),
                gamma_ptr,
                beta_ptr,
                epsilon,
                cstring_ptr(&name),
            )
        };
        wrap_tensor(ptr)
    }

/// Calls the `MPSGraph` framework counterpart for `run`.
    pub fn run(&self, feeds: &[Feed<'_>], targets: &[&Tensor]) -> Result<Vec<TensorData>> {
        let feed_tensors = feeds
            .iter()
            .map(|feed| feed.tensor.as_ptr())
            .collect::<Vec<_>>();
        let feed_data = feeds
            .iter()
            .map(|feed| feed.data.as_ptr())
            .collect::<Vec<_>>();
        let target_tensors = targets
            .iter()
            .map(|tensor| tensor.as_ptr())
            .collect::<Vec<_>>();
        let mut results = vec![ptr::null_mut(); targets.len()];

        // SAFETY: The pointer arrays are valid for the duration of the FFI call.
        let ok = unsafe {
            ffi::mpsgraph_graph_run(
                self.ptr,
                feed_tensors.as_ptr(),
                feed_data.as_ptr(),
                feeds.len(),
                target_tensors.as_ptr(),
                targets.len(),
                results.as_mut_ptr(),
            )
        };
        if ok {
            wrap_tensor_data_results(results, "failed to run graph")
        } else {
            Err(Error::OperationFailed("failed to run graph"))
        }
    }

/// Calls the `MPSGraph` framework counterpart for `run_with_command_queue`.
    pub fn run_with_command_queue(
        &self,
        command_queue: &CommandQueue,
        feeds: &[Feed<'_>],
        targets: &[&Tensor],
    ) -> Result<Vec<TensorData>> {
        let feed_tensors = feeds
            .iter()
            .map(|feed| feed.tensor.as_ptr())
            .collect::<Vec<_>>();
        let feed_data = feeds
            .iter()
            .map(|feed| feed.data.as_ptr())
            .collect::<Vec<_>>();
        let target_tensors = targets
            .iter()
            .map(|tensor| tensor.as_ptr())
            .collect::<Vec<_>>();
        let mut results = vec![ptr::null_mut(); targets.len()];

        // SAFETY: The pointer arrays are valid for the duration of the FFI call.
        let ok = unsafe {
            ffi::mpsgraph_graph_run_with_command_queue(
                self.ptr,
                command_queue.as_ptr(),
                feed_tensors.as_ptr(),
                feed_data.as_ptr(),
                feeds.len(),
                target_tensors.as_ptr(),
                targets.len(),
                results.as_mut_ptr(),
            )
        };
        if ok {
            wrap_tensor_data_results(results, "failed to run graph with command queue")
        } else {
            Err(Error::OperationFailed(
                "failed to run graph with command queue",
            ))
        }
    }

/// Calls the `MPSGraph` framework counterpart for `compile`.
    #[must_use]
    pub fn compile(
        &self,
        device: &MetalDevice,
        feeds: &[FeedDescription<'_>],
        targets: &[&Tensor],
    ) -> Option<Executable> {
        let feed_tensors = feeds
            .iter()
            .map(|feed| feed.tensor.as_ptr())
            .collect::<Vec<_>>();
        let shape_lengths = feeds
            .iter()
            .map(|feed| feed.shape.len())
            .collect::<Vec<_>>();
        let data_types = feeds.iter().map(|feed| feed.data_type).collect::<Vec<_>>();
        let flat_shapes = feeds
            .iter()
            .flat_map(|feed| feed.shape.iter().copied())
            .collect::<Vec<_>>();
        let target_tensors = targets
            .iter()
            .map(|tensor| tensor.as_ptr())
            .collect::<Vec<_>>();

        // SAFETY: The pointer arrays are valid for the duration of the FFI call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_compile(
                self.ptr,
                device.as_ptr(),
                feed_tensors.as_ptr(),
                feeds.len(),
                flat_shapes.as_ptr(),
                shape_lengths.as_ptr(),
                data_types.as_ptr(),
                target_tensors.as_ptr(),
                targets.len(),
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Executable::from_raw(ptr, targets.len()))
        }
    }
}

/// Safe owner for a compiled `MPSGraphExecutable`.
pub struct Executable {
    ptr: *mut c_void,
    output_count: usize,
}

unsafe impl Send for Executable {}
unsafe impl Sync for Executable {}

impl Drop for Executable {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `ptr` is a +1 retained Swift/ObjC object pointer owned by this wrapper.
            unsafe { ffi::mpsgraph_object_release(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

impl Executable {
    pub(crate) const fn from_raw(ptr: *mut c_void, output_count: usize) -> Self {
        Self { ptr, output_count }
    }

/// Mirrors the `MPSGraph` framework constant `fn`.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

/// Mirrors the `MPSGraph` framework constant `fn`.
    #[must_use]
    pub const fn output_count(&self) -> usize {
        self.output_count
    }

/// Calls the `MPSGraph` framework counterpart for `run`.
    pub fn run(
        &self,
        command_queue: &CommandQueue,
        inputs: &[&TensorData],
    ) -> Result<Vec<TensorData>> {
        let input_data = inputs
            .iter()
            .map(|tensor_data| tensor_data.as_ptr())
            .collect::<Vec<_>>();
        let mut results = vec![ptr::null_mut(); self.output_count];

        // SAFETY: The pointer arrays are valid for the duration of the FFI call.
        let ok = unsafe {
            ffi::mpsgraph_executable_run(
                self.ptr,
                command_queue.as_ptr(),
                input_data.as_ptr(),
                inputs.len(),
                self.output_count,
                results.as_mut_ptr(),
            )
        };
        if ok {
            wrap_tensor_data_results(results, "failed to run executable")
        } else {
            Err(Error::OperationFailed("failed to run executable"))
        }
    }
}
