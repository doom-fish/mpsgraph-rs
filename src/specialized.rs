use crate::checks;
use crate::error::{Error, Result};
use crate::execution::{ExecutableExecutionDescriptor, ExecutionDescriptor};
use crate::ffi;
use crate::graph::{
    axes_in_range, checked_byte_len, constant_data_type, convolution2d_shapes, data_type,
    image_layout, padding_mode, padding_style, pads_window, tensor_named_data_layout,
    valid_padding_style, Convolution2DDescriptor, Convolution2DDescriptorInfo, Graph, Tensor,
};
use crate::types::{Operation, ShapedType};
use core::ffi::{c_char, c_void};
use core::ptr;
use std::ffi::CString;

fn release_handle(ptr: &mut *mut c_void) {
    if !ptr.is_null() {
        // SAFETY: `ptr` is a +1 retained Swift/ObjC object pointer owned by this wrapper.
        unsafe { ffi::mpsgraph_object_release(*ptr) };
        *ptr = ptr::null_mut();
    }
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
                release_handle(&mut self.ptr);
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

macro_rules! descriptor_handle {
    ($name:ident, $info:ident) => {
/// Mirrors the `MPSGraph` framework counterpart for this type.
        pub struct $name {
            ptr: *mut c_void,
            info: $info,
        }

        unsafe impl Send for $name {}
        unsafe impl Sync for $name {}

        impl Drop for $name {
            fn drop(&mut self) {
                release_handle(&mut self.ptr);
            }
        }

        impl $name {
/// Mirrors the `MPSGraph` framework constant `fn`.
            #[must_use]
            pub const fn as_ptr(&self) -> *mut c_void {
                self.ptr
            }

            pub(crate) const fn info(&self) -> &$info {
                &self.info
            }

            fn created(ptr: *mut c_void, info: $info) -> Result<Self> {
                if ptr.is_null() {
                    Err(Error::OperationFailed(concat!(
                        "MPSGraph did not create the ",
                        stringify!($name)
                    )))
                } else {
                    Ok(Self { ptr, info })
                }
            }
        }
    };
    ($name:ident) => {
/// Mirrors the `MPSGraph` framework counterpart for this type.
        pub struct $name {
            ptr: *mut c_void,
        }

        unsafe impl Send for $name {}
        unsafe impl Sync for $name {}

        impl Drop for $name {
            fn drop(&mut self) {
                release_handle(&mut self.ptr);
            }
        }

        impl $name {
/// Mirrors the `MPSGraph` framework constant `fn`.
            #[must_use]
            pub const fn as_ptr(&self) -> *mut c_void {
                self.ptr
            }

            fn created(ptr: *mut c_void) -> Result<Self> {
                if ptr.is_null() {
                    Err(Error::OperationFailed(concat!(
                        "MPSGraph did not create the ",
                        stringify!($name)
                    )))
                } else {
                    Ok(Self { ptr })
                }
            }
        }
    };
}

/// `MPSGraphExecutionStage` constants.
pub mod execution_stage {
/// Mirrors the `MPSGraph` framework constant `COMPLETED`.
    pub const COMPLETED: u64 = 0;
}

/// `MPSGraphReductionMode` constants.
pub mod reduction_mode {
/// Mirrors the `MPSGraph` framework constant `MIN`.
    pub const MIN: usize = 0;
/// Mirrors the `MPSGraph` framework constant `MAX`.
    pub const MAX: usize = 1;
/// Mirrors the `MPSGraph` framework constant `SUM`.
    pub const SUM: usize = 2;
/// Mirrors the `MPSGraph` framework constant `PRODUCT`.
    pub const PRODUCT: usize = 3;
/// Mirrors the `MPSGraph` framework constant `ARGUMENT_MIN`.
    pub const ARGUMENT_MIN: usize = 4;
/// Mirrors the `MPSGraph` framework constant `ARGUMENT_MAX`.
    pub const ARGUMENT_MAX: usize = 5;
}

/// `MPSGraphPoolingReturnIndicesMode` constants.
pub mod pooling_return_indices_mode {
/// Mirrors the `MPSGraph` framework constant `NONE`.
    pub const NONE: usize = 0;
/// Mirrors the `MPSGraph` framework constant `GLOBAL_FLATTEN_1D`.
    pub const GLOBAL_FLATTEN_1D: usize = 1;
/// Mirrors the `MPSGraph` framework constant `GLOBAL_FLATTEN_2D`.
    pub const GLOBAL_FLATTEN_2D: usize = 2;
/// Mirrors the `MPSGraph` framework constant `GLOBAL_FLATTEN_3D`.
    pub const GLOBAL_FLATTEN_3D: usize = 3;
/// Mirrors the `MPSGraph` framework constant `GLOBAL_FLATTEN_4D`.
    pub const GLOBAL_FLATTEN_4D: usize = 4;
/// Mirrors the `MPSGraph` framework constant `LOCAL_FLATTEN_1D`.
    pub const LOCAL_FLATTEN_1D: usize = 5;
/// Mirrors the `MPSGraph` framework constant `LOCAL_FLATTEN_2D`.
    pub const LOCAL_FLATTEN_2D: usize = 6;
/// Mirrors the `MPSGraph` framework constant `LOCAL_FLATTEN_3D`.
    pub const LOCAL_FLATTEN_3D: usize = 7;
/// Mirrors the `MPSGraph` framework constant `LOCAL_FLATTEN_4D`.
    pub const LOCAL_FLATTEN_4D: usize = 8;
}

/// `MPSGraphFFTScalingMode` constants.
pub mod fft_scaling_mode {
/// Mirrors the `MPSGraph` framework constant `NONE`.
    pub const NONE: usize = 0;
/// Mirrors the `MPSGraph` framework constant `SIZE`.
    pub const SIZE: usize = 1;
/// Mirrors the `MPSGraph` framework constant `UNITARY`.
    pub const UNITARY: usize = 2;
}

/// `MPSGraphLossReductionType` constants.
pub mod loss_reduction_type {
/// Mirrors the `MPSGraph` framework constant `NONE`.
    pub const NONE: u64 = 0;
/// Mirrors the `MPSGraph` framework constant `AXIS`.
    pub const AXIS: u64 = 0;
/// Mirrors the `MPSGraph` framework constant `SUM`.
    pub const SUM: u64 = 1;
/// Mirrors the `MPSGraph` framework constant `MEAN`.
    pub const MEAN: u64 = 2;
}

/// `MPSGraphNonMaximumSuppressionCoordinateMode` constants.
pub mod non_maximum_suppression_coordinate_mode {
/// Mirrors the `MPSGraph` framework constant `CORNERS_HEIGHT_FIRST`.
    pub const CORNERS_HEIGHT_FIRST: usize = 0;
/// Mirrors the `MPSGraph` framework constant `CORNERS_WIDTH_FIRST`.
    pub const CORNERS_WIDTH_FIRST: usize = 1;
/// Mirrors the `MPSGraph` framework constant `CENTERS_HEIGHT_FIRST`.
    pub const CENTERS_HEIGHT_FIRST: usize = 2;
/// Mirrors the `MPSGraph` framework constant `CENTERS_WIDTH_FIRST`.
    pub const CENTERS_WIDTH_FIRST: usize = 3;
}

/// `MPSGraphResizeMode` constants.
pub mod resize_mode {
/// Mirrors the `MPSGraph` framework constant `NEAREST`.
    pub const NEAREST: usize = 0;
/// Mirrors the `MPSGraph` framework constant `BILINEAR`.
    pub const BILINEAR: usize = 1;
}

/// `MPSGraphResizeNearestRoundingMode` constants.
pub mod resize_nearest_rounding_mode {
/// Mirrors the `MPSGraph` framework constant `ROUND_PREFER_CEIL`.
    pub const ROUND_PREFER_CEIL: usize = 0;
/// Mirrors the `MPSGraph` framework constant `ROUND_PREFER_FLOOR`.
    pub const ROUND_PREFER_FLOOR: usize = 1;
/// Mirrors the `MPSGraph` framework constant `CEIL`.
    pub const CEIL: usize = 2;
/// Mirrors the `MPSGraph` framework constant `FLOOR`.
    pub const FLOOR: usize = 3;
/// Mirrors the `MPSGraph` framework constant `ROUND_TO_EVEN`.
    pub const ROUND_TO_EVEN: usize = 4;
/// Mirrors the `MPSGraph` framework constant `ROUND_TO_ODD`.
    pub const ROUND_TO_ODD: usize = 5;
}

/// `MPSGraphScatterMode` constants.
pub mod scatter_mode {
/// Mirrors the `MPSGraph` framework constant `ADD`.
    pub const ADD: isize = 0;
/// Mirrors the `MPSGraph` framework constant `SUB`.
    pub const SUB: isize = 1;
/// Mirrors the `MPSGraph` framework constant `MUL`.
    pub const MUL: isize = 2;
/// Mirrors the `MPSGraph` framework constant `DIV`.
    pub const DIV: isize = 3;
/// Mirrors the `MPSGraph` framework constant `MIN`.
    pub const MIN: isize = 4;
/// Mirrors the `MPSGraph` framework constant `MAX`.
    pub const MAX: isize = 5;
/// Mirrors the `MPSGraph` framework constant `SET`.
    pub const SET: isize = 6;
}

/// `MPSGraphSparseStorageType` constants.
pub mod sparse_storage_type {
/// Mirrors the `MPSGraph` framework constant `COO`.
    pub const COO: u64 = 0;
/// Mirrors the `MPSGraph` framework constant `CSC`.
    pub const CSC: u64 = 1;
/// Mirrors the `MPSGraph` framework constant `CSR`.
    pub const CSR: u64 = 2;
}

opaque_handle!(Object);
impl Object {
    fn retain_from(ptr: *mut c_void) -> Self {
        // SAFETY: `ptr` belongs to a live `MPSGraphObject` subclass and the bridge retains it for this wrapper.
        let ptr = unsafe { ffi::mpsgraph_object_retain(ptr) };
        Self { ptr }
    }
}

opaque_handle!(GraphType);
impl GraphType {
    fn retain_from(ptr: *mut c_void) -> Self {
        // SAFETY: `ptr` belongs to a live `MPSGraphType` subclass and the bridge retains it for this wrapper.
        let ptr = unsafe { ffi::mpsgraph_object_retain(ptr) };
        Self { ptr }
    }

/// Calls the `MPSGraph` framework counterpart for `as_object`.
    #[must_use]
    pub fn as_object(&self) -> Object {
        Object::retain_from(self.ptr)
    }
}

/// Mirrors the `MPSGraph` framework counterpart for this type.
pub struct VariableOp {
    ptr: *mut c_void,
    scope: u64,
    inputs: Vec<usize>,
}

unsafe impl Send for VariableOp {}
unsafe impl Sync for VariableOp {}

impl Drop for VariableOp {
    fn drop(&mut self) {
        release_handle(&mut self.ptr);
    }
}

impl VariableOp {
/// Mirrors the `MPSGraph` framework constant `fn`.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

/// Calls the `MPSGraph` framework counterpart for `shape`.
    #[must_use]
    pub fn shape(&self) -> Vec<isize> {
        // SAFETY: `self.ptr` is a live variable-op handle.
        let len = unsafe { ffi::mpsgraph_variable_op_shape_len(self.ptr) };
        let mut shape = vec![0_isize; len];
        if len > 0 {
            // SAFETY: `shape` has space for exactly `len` elements.
            unsafe { ffi::mpsgraph_variable_op_copy_shape(self.ptr, shape.as_mut_ptr()) };
        }
        shape
    }

/// Calls the `MPSGraph` framework counterpart for `data_type`.
    #[must_use]
    pub fn data_type(&self) -> u32 {
        // SAFETY: `self.ptr` is a live variable-op handle.
        unsafe { ffi::mpsgraph_variable_op_data_type(self.ptr) }
    }

/// Calls the `MPSGraph` framework counterpart for `as_object`.
    #[must_use]
    pub fn as_object(&self) -> Object {
        Object::retain_from(self.ptr)
    }

/// Calls the `MPSGraph` framework counterpart for `as_operation`.
    #[must_use]
    pub fn as_operation(&self) -> Operation {
        // SAFETY: `self.ptr` is a live variable-op handle and retains as an operation wrapper.
        let ptr = unsafe { ffi::mpsgraph_object_retain(self.ptr) };
        Operation::from_raw(ptr, self.scope, self.inputs.clone())
    }
}

impl ShapedType {
/// Calls the `MPSGraph` framework counterpart for `as_graph_type`.
    #[must_use]
    pub fn as_graph_type(&self) -> GraphType {
        GraphType::retain_from(self.as_ptr())
    }
}

impl Operation {
/// Calls the `MPSGraph` framework counterpart for `as_variable`.
    #[must_use]
    pub fn as_variable(&self) -> Option<VariableOp> {
        // SAFETY: `self.ptr` is a live operation handle.
        let ptr = unsafe { ffi::mpsgraph_operation_as_variable(self.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(VariableOp {
                ptr,
                scope: self.scope(),
                inputs: self.inputs().to_vec(),
            })
        }
    }
}

/// Mirrors the `MPSGraph` framework counterpart for `Convolution3DDescriptorInfo`.
#[derive(Debug, Clone, Copy)]
pub struct Convolution3DDescriptorInfo {
/// Mirrors the `MPSGraph` framework property for `stride_in_x`.
    pub stride_in_x: usize,
/// Mirrors the `MPSGraph` framework property for `stride_in_y`.
    pub stride_in_y: usize,
/// Mirrors the `MPSGraph` framework property for `stride_in_z`.
    pub stride_in_z: usize,
/// Mirrors the `MPSGraph` framework property for `dilation_rate_in_x`.
    pub dilation_rate_in_x: usize,
/// Mirrors the `MPSGraph` framework property for `dilation_rate_in_y`.
    pub dilation_rate_in_y: usize,
/// Mirrors the `MPSGraph` framework property for `dilation_rate_in_z`.
    pub dilation_rate_in_z: usize,
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
/// Mirrors the `MPSGraph` framework property for `padding_front`.
    pub padding_front: usize,
/// Mirrors the `MPSGraph` framework property for `padding_back`.
    pub padding_back: usize,
/// Mirrors the `MPSGraph` framework property for `padding_style`.
    pub padding_style: usize,
/// Mirrors the `MPSGraph` framework property for `data_layout`.
    pub data_layout: usize,
/// Mirrors the `MPSGraph` framework property for `weights_layout`.
    pub weights_layout: usize,
}

impl Default for Convolution3DDescriptorInfo {
    fn default() -> Self {
        Self {
            stride_in_x: 1,
            stride_in_y: 1,
            stride_in_z: 1,
            dilation_rate_in_x: 1,
            dilation_rate_in_y: 1,
            dilation_rate_in_z: 1,
            groups: 1,
            padding_left: 0,
            padding_right: 0,
            padding_top: 0,
            padding_bottom: 0,
            padding_front: 0,
            padding_back: 0,
            padding_style: padding_style::EXPLICIT,
            data_layout: tensor_named_data_layout::NDHWC,
            weights_layout: tensor_named_data_layout::DHWIO,
        }
    }
}


/// Mirrors the `MPSGraph` framework counterpart for `DepthwiseConvolution2DDescriptorInfo`.
#[derive(Debug, Clone, Copy)]
pub struct DepthwiseConvolution2DDescriptorInfo {
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
/// Mirrors the `MPSGraph` framework property for `weights_layout`.
    pub weights_layout: usize,
}

impl Default for DepthwiseConvolution2DDescriptorInfo {
    fn default() -> Self {
        Self {
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
            weights_layout: tensor_named_data_layout::HWIO,
        }
    }
}


/// Mirrors the `MPSGraph` framework counterpart for `DepthwiseConvolution3DDescriptorInfo`.
#[derive(Debug, Clone, Copy)]
pub struct DepthwiseConvolution3DDescriptorInfo {
/// Mirrors the `MPSGraph` framework property for `strides`.
    pub strides: [usize; 3],
/// Mirrors the `MPSGraph` framework property for `dilation_rates`.
    pub dilation_rates: [usize; 3],
/// Mirrors the `MPSGraph` framework property for `padding_values`.
    pub padding_values: [usize; 6],
/// Mirrors the `MPSGraph` framework property for `padding_style`.
    pub padding_style: usize,
/// Mirrors the `MPSGraph` framework property for `channel_dimension_index`.
    pub channel_dimension_index: isize,
}

impl Default for DepthwiseConvolution3DDescriptorInfo {
    fn default() -> Self {
        Self {
            strides: [1, 1, 1],
            dilation_rates: [1, 1, 1],
            padding_values: [0, 0, 0, 0, 0, 0],
            padding_style: padding_style::EXPLICIT,
            channel_dimension_index: -1,
        }
    }
}


/// Mirrors the `MPSGraph` framework counterpart for `FftDescriptorInfo`.
#[derive(Debug, Clone, Copy)]
pub struct FftDescriptorInfo {
/// Mirrors the `MPSGraph` framework property for `inverse`.
    pub inverse: bool,
/// Mirrors the `MPSGraph` framework property for `scaling_mode`.
    pub scaling_mode: usize,
/// Mirrors the `MPSGraph` framework property for `round_to_odd_hermitean`.
    pub round_to_odd_hermitean: bool,
}

impl Default for FftDescriptorInfo {
    fn default() -> Self {
        Self {
            inverse: false,
            scaling_mode: fft_scaling_mode::NONE,
            round_to_odd_hermitean: false,
        }
    }
}


/// Mirrors the `MPSGraph` framework counterpart for `ImToColDescriptorInfo`.
#[derive(Debug, Clone, Copy)]
pub struct ImToColDescriptorInfo {
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
/// Mirrors the `MPSGraph` framework property for `data_layout`.
    pub data_layout: usize,
}

impl Default for ImToColDescriptorInfo {
    fn default() -> Self {
        Self {
            kernel_width: 1,
            kernel_height: 1,
            stride_in_x: 1,
            stride_in_y: 1,
            dilation_rate_in_x: 1,
            dilation_rate_in_y: 1,
            padding_left: 0,
            padding_right: 0,
            padding_top: 0,
            padding_bottom: 0,
            data_layout: tensor_named_data_layout::NHWC,
        }
    }
}


/// Mirrors the `MPSGraph` framework counterpart for `Pooling4DDescriptorInfo`.
#[derive(Debug, Clone, Copy)]
pub struct Pooling4DDescriptorInfo {
/// Mirrors the `MPSGraph` framework property for `kernel_sizes`.
    pub kernel_sizes: [usize; 4],
/// Mirrors the `MPSGraph` framework property for `strides`.
    pub strides: [usize; 4],
/// Mirrors the `MPSGraph` framework property for `dilation_rates`.
    pub dilation_rates: [usize; 4],
/// Mirrors the `MPSGraph` framework property for `padding_values`.
    pub padding_values: [usize; 8],
/// Mirrors the `MPSGraph` framework property for `padding_style`.
    pub padding_style: usize,
/// Mirrors the `MPSGraph` framework property for `ceil_mode`.
    pub ceil_mode: bool,
/// Mirrors the `MPSGraph` framework property for `include_zero_pad_to_average`.
    pub include_zero_pad_to_average: bool,
/// Mirrors the `MPSGraph` framework property for `return_indices_mode`.
    pub return_indices_mode: usize,
/// Mirrors the `MPSGraph` framework property for `return_indices_data_type`.
    pub return_indices_data_type: u32,
}

impl Default for Pooling4DDescriptorInfo {
    fn default() -> Self {
        Self {
            kernel_sizes: [1, 1, 1, 1],
            strides: [1, 1, 1, 1],
            dilation_rates: [1, 1, 1, 1],
            padding_values: [0, 0, 0, 0, 0, 0, 0, 0],
            padding_style: padding_style::EXPLICIT,
            ceil_mode: false,
            include_zero_pad_to_average: false,
            return_indices_mode: pooling_return_indices_mode::NONE,
            return_indices_data_type: data_type::INT32,
        }
    }
}



/// Mirrors the `MPSGraph` framework counterpart for `StencilDescriptorInfo`.
#[derive(Debug, Clone, Copy)]
pub struct StencilDescriptorInfo {
/// Mirrors the `MPSGraph` framework property for `reduction_mode`.
    pub reduction_mode: usize,
/// Mirrors the `MPSGraph` framework property for `offsets`.
    pub offsets: [isize; 4],
/// Mirrors the `MPSGraph` framework property for `strides`.
    pub strides: [usize; 4],
/// Mirrors the `MPSGraph` framework property for `dilation_rates`.
    pub dilation_rates: [usize; 4],
/// Mirrors the `MPSGraph` framework property for `explicit_padding`.
    pub explicit_padding: [usize; 8],
/// Mirrors the `MPSGraph` framework property for `boundary_mode`.
    pub boundary_mode: isize,
/// Mirrors the `MPSGraph` framework property for `padding_style`.
    pub padding_style: usize,
/// Mirrors the `MPSGraph` framework property for `padding_constant`.
    pub padding_constant: f32,
}

impl Default for StencilDescriptorInfo {
    fn default() -> Self {
        Self {
            reduction_mode: reduction_mode::SUM,
            offsets: [0, 0, 0, 0],
            strides: [1, 1, 1, 1],
            dilation_rates: [1, 1, 1, 1],
            explicit_padding: [0, 0, 0, 0, 0, 0, 0, 0],
            boundary_mode: padding_mode::ZERO,
            padding_style: padding_style::EXPLICIT,
            padding_constant: 0.0,
        }
    }
}


descriptor_handle!(Convolution3DDescriptor, Convolution3DDescriptorInfo);
impl Convolution3DDescriptor {
/// Calls the `MPSGraph` framework counterpart for `new`.
    pub fn new(info: Convolution3DDescriptorInfo) -> Result<Self> {
        positive(
            &[
                info.stride_in_x,
                info.stride_in_y,
                info.stride_in_z,
                info.dilation_rate_in_x,
                info.dilation_rate_in_y,
                info.dilation_rate_in_z,
                info.groups,
            ],
            "3D convolution strides, dilation rates and groups must be at least 1",
        )?;
        padding(info.padding_style)?;
        if !matches!(
            info.data_layout,
            tensor_named_data_layout::NDHWC | tensor_named_data_layout::NCDHW
        ) || !matches!(
            info.weights_layout,
            tensor_named_data_layout::DHWIO | tensor_named_data_layout::OIDHW
        ) {
            return Err(Error::InvalidArgument(
                "3D convolutions take NDHWC or NCDHW data and DHWIO or OIDHW weights",
            ));
        }
        // SAFETY: all arguments are POD configuration values.
        let ptr = unsafe {
            ffi::mpsgraph_convolution3d_descriptor_new(
                info.stride_in_x,
                info.stride_in_y,
                info.stride_in_z,
                info.dilation_rate_in_x,
                info.dilation_rate_in_y,
                info.dilation_rate_in_z,
                info.groups,
                info.padding_left,
                info.padding_right,
                info.padding_top,
                info.padding_bottom,
                info.padding_front,
                info.padding_back,
                info.padding_style,
                info.data_layout,
                info.weights_layout,
            )
        };
        Self::created(ptr, info)
    }
}

descriptor_handle!(DepthwiseConvolution2DDescriptor);
impl DepthwiseConvolution2DDescriptor {
/// Calls the `MPSGraph` framework counterpart for `new`.
    pub fn new(info: DepthwiseConvolution2DDescriptorInfo) -> Result<Self> {
        positive(
            &[
                info.stride_in_x,
                info.stride_in_y,
                info.dilation_rate_in_x,
                info.dilation_rate_in_y,
            ],
            "depthwise convolution strides and dilation rates must be at least 1",
        )?;
        padding(info.padding_style)?;
        if !matches!(
            info.data_layout,
            tensor_named_data_layout::NHWC | tensor_named_data_layout::NCHW
        ) || !matches!(
            info.weights_layout,
            tensor_named_data_layout::HWIO | tensor_named_data_layout::OIHW
        ) {
            return Err(Error::InvalidArgument(
                "depthwise convolutions take NHWC or NCHW data and HWIO or OIHW weights",
            ));
        }
        // SAFETY: all arguments are POD configuration values.
        let ptr = unsafe {
            ffi::mpsgraph_depthwise_convolution2d_descriptor_new(
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
                info.weights_layout,
            )
        };
        Self::created(ptr)
    }
}

descriptor_handle!(
    DepthwiseConvolution3DDescriptor,
    DepthwiseConvolution3DDescriptorInfo
);
impl DepthwiseConvolution3DDescriptor {
/// Calls the `MPSGraph` framework counterpart for `new`.
    pub fn new(info: DepthwiseConvolution3DDescriptorInfo) -> Result<Self> {
        positive(
            &[info.strides.as_slice(), info.dilation_rates.as_slice()].concat(),
            "depthwise convolution strides and dilation rates must be at least 1",
        )?;
        padding(info.padding_style)?;
        // SAFETY: all slices stay alive for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_depthwise_convolution3d_descriptor_new(
                info.strides.as_ptr(),
                info.strides.len(),
                info.dilation_rates.as_ptr(),
                info.dilation_rates.len(),
                info.padding_values.as_ptr(),
                info.padding_values.len(),
                info.padding_style,
                info.channel_dimension_index,
            )
        };
        Self::created(ptr, info)
    }
}

descriptor_handle!(FftDescriptor);
impl FftDescriptor {
/// Calls the `MPSGraph` framework counterpart for `new`.
    pub fn new(info: FftDescriptorInfo) -> Result<Self> {
        if info.scaling_mode > fft_scaling_mode::UNITARY {
            return Err(Error::InvalidArgument("unknown MPSGraphFFTScalingMode"));
        }
        // SAFETY: all arguments are POD configuration values.
        let ptr = unsafe {
            ffi::mpsgraph_fft_descriptor_new(
                info.inverse,
                info.scaling_mode,
                info.round_to_odd_hermitean,
            )
        };
        Self::created(ptr)
    }
}

descriptor_handle!(ImToColDescriptor, ImToColDescriptorInfo);
impl ImToColDescriptor {
/// Calls the `MPSGraph` framework counterpart for `new`.
    pub fn new(info: ImToColDescriptorInfo) -> Result<Self> {
        positive(
            &[
                info.kernel_width,
                info.kernel_height,
                info.stride_in_x,
                info.stride_in_y,
                info.dilation_rate_in_x,
                info.dilation_rate_in_y,
            ],
            "im2col kernel sizes, strides and dilation rates must be at least 1",
        )?;
        if !matches!(
            info.data_layout,
            tensor_named_data_layout::NHWC | tensor_named_data_layout::NCHW
        ) {
            return Err(Error::InvalidArgument("im2col takes NHWC or NCHW data"));
        }
        // SAFETY: all arguments are POD configuration values.
        let ptr = unsafe {
            ffi::mpsgraph_im_to_col_descriptor_new(
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
                info.data_layout,
            )
        };
        Self::created(ptr, info)
    }
}

descriptor_handle!(Pooling4DDescriptor);
impl Pooling4DDescriptor {
/// Calls the `MPSGraph` framework counterpart for `new`.
    pub fn new(info: Pooling4DDescriptorInfo) -> Result<Self> {
        positive(
            &[
                info.kernel_sizes.as_slice(),
                info.strides.as_slice(),
                info.dilation_rates.as_slice(),
            ]
            .concat(),
            "pooling kernel sizes, strides and dilation rates must be at least 1",
        )?;
        padding(info.padding_style)?;
        if info.return_indices_mode > pooling_return_indices_mode::LOCAL_FLATTEN_4D {
            return Err(Error::InvalidArgument(
                "unknown MPSGraphPoolingReturnIndicesMode",
            ));
        }
        if !constant_data_type(info.return_indices_data_type) {
            return Err(Error::UnsupportedDataType(info.return_indices_data_type));
        }
        // SAFETY: all slices stay alive for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_pooling4d_descriptor_new(
                info.kernel_sizes.as_ptr(),
                info.kernel_sizes.len(),
                info.strides.as_ptr(),
                info.strides.len(),
                info.dilation_rates.as_ptr(),
                info.dilation_rates.len(),
                info.padding_values.as_ptr(),
                info.padding_values.len(),
                info.padding_style,
                info.ceil_mode,
                info.include_zero_pad_to_average,
                info.return_indices_mode,
                info.return_indices_data_type,
            )
        };
        Self::created(ptr)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SparseInfo {
    storage_type: u64,
}

descriptor_handle!(CreateSparseDescriptor, SparseInfo);
impl CreateSparseDescriptor {
/// Calls the `MPSGraph` framework counterpart for `new`.
    pub fn new(storage_type: u64, data_type: u32) -> Result<Self> {
        if storage_type > sparse_storage_type::CSR {
            return Err(Error::InvalidArgument("unknown MPSGraphSparseStorageType"));
        }
        if data_type != data_type::FLOAT32 {
            return Err(Error::InvalidDataType(
                "MPSGraph materializes float32 sparse tensors only",
            ));
        }
        // SAFETY: all arguments are POD configuration values.
        let ptr = unsafe { ffi::mpsgraph_sparse_descriptor_new(storage_type, data_type) };
        Self::created(ptr, SparseInfo { storage_type })
    }
}

descriptor_handle!(StencilDescriptor, StencilDescriptorInfo);
impl StencilDescriptor {
/// Calls the `MPSGraph` framework counterpart for `new`.
    pub fn new(info: StencilDescriptorInfo) -> Result<Self> {
        positive(
            &[info.strides.as_slice(), info.dilation_rates.as_slice()].concat(),
            "stencil strides and dilation rates must be at least 1",
        )?;
        padding(info.padding_style)?;
        if info.reduction_mode > reduction_mode::ARGUMENT_MAX {
            return Err(Error::InvalidArgument("unknown MPSGraphReductionMode"));
        }
        boundary(info.boundary_mode)?;
        // SAFETY: all slices stay alive for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_stencil_descriptor_new(
                info.reduction_mode,
                info.offsets.as_ptr(),
                info.offsets.len(),
                info.strides.as_ptr(),
                info.strides.len(),
                info.dilation_rates.as_ptr(),
                info.dilation_rates.len(),
                info.explicit_padding.as_ptr(),
                info.explicit_padding.len(),
                info.boundary_mode,
                info.padding_style,
                info.padding_constant,
            )
        };
        Self::created(ptr, info)
    }
}

fn optional_cstring(name: Option<&str>) -> Option<CString> {
    name.and_then(|value| CString::new(value).ok())
}

#[allow(clippy::ref_option)]
fn cstring_ptr(value: &Option<CString>) -> *const c_char {
    value.as_ref().map_or(ptr::null(), |value| value.as_ptr())
}

fn positive(values: &[usize], message: &'static str) -> Result<()> {
    if values.contains(&0) {
        Err(Error::InvalidArgument(message))
    } else {
        Ok(())
    }
}

fn padding(style: usize) -> Result<()> {
    if valid_padding_style(style) {
        Ok(())
    } else {
        Err(Error::InvalidArgument("unknown MPSGraphPaddingStyle"))
    }
}

pub(crate) fn boundary(mode: isize) -> Result<()> {
    if matches!(
        mode,
        padding_mode::CONSTANT
            | padding_mode::REFLECT
            | padding_mode::SYMMETRIC
            | padding_mode::CLAMP_TO_EDGE
            | padding_mode::ZERO
    ) {
        Ok(())
    } else {
        Err(Error::InvalidArgument(
            "the padding mode must be CONSTANT, REFLECT, SYMMETRIC, CLAMP_TO_EDGE or ZERO; MPSGraph does not run PERIODIC or ANTI_PERIODIC",
        ))
    }
}

fn check_scatter_mode(mode: isize) -> Result<()> {
    if (scatter_mode::ADD..=scatter_mode::SET).contains(&mode) {
        Ok(())
    } else {
        Err(Error::InvalidArgument("unknown MPSGraphScatterMode"))
    }
}

fn window(
    extent: isize,
    kernel: isize,
    dilation: usize,
    pads: (usize, usize),
) -> bool {
    checks::known(kernel)
        .is_none_or(|kernel| kernel > 0 && checks::window_fits(extent, kernel, dilation, pads))
}

const WINDOW_TOO_LARGE: &str = "the dilated window is larger than the padded source";

fn volume_layout(layout: usize, dims: &[isize]) -> Result<(isize, [isize; 3], isize)> {
    let &[first, second, third, fourth, fifth] = dims else {
        return Err(Error::InvalidShape("3D volume tensors must have rank 5"));
    };
    match layout {
        tensor_named_data_layout::NDHWC => Ok((first, [second, third, fourth], fifth)),
        _ => Ok((first, [third, fourth, fifth], second)),
    }
}

fn check_convolution3d(
    info: &Convolution3DDescriptorInfo,
    source: &Tensor,
    weights: &Tensor,
) -> Result<()> {
    checks::float_tensor(source, "convolutions need a floating-point source")?;
    checks::same_data_type(
        source,
        weights,
        "the convolution weights need the source's data type",
    )?;
    let (Some(source_dims), Some(weight_dims)) = (
        checks::rank_exactly(source, 5, "the 3D convolution source must have rank 5")?,
        checks::rank_exactly(weights, 5, "3D convolution weights must have rank 5")?,
    ) else {
        return Ok(());
    };
    let (_, spatial, channels) = volume_layout(info.data_layout, &source_dims)?;
    let &[w0, w1, w2, w3, w4] = weight_dims.as_slice() else {
        return Ok(());
    };
    let (kernel, weight_inputs, weight_outputs) = match info.weights_layout {
        tensor_named_data_layout::DHWIO => ([w0, w1, w2], w3, w4),
        _ => ([w2, w3, w4], w1, w0),
    };
    if let (Some(channels), Some(inputs)) = (checks::known(channels), checks::known(weight_inputs)) {
        if inputs.checked_mul(info.groups) != Some(channels) {
            return Err(Error::InvalidShape(
                "the input channels must equal groups times the weights' input channels",
            ));
        }
    }
    if checks::known(weight_outputs).is_some_and(|outputs| outputs % info.groups != 0) {
        return Err(Error::InvalidShape(
            "the weights' output channels must be divisible by groups",
        ));
    }
    if let Some(padded) = pads_window(info.padding_style) {
        let pads = if padded {
            [
                (info.padding_front, info.padding_back),
                (info.padding_top, info.padding_bottom),
                (info.padding_left, info.padding_right),
            ]
        } else {
            [(0, 0); 3]
        };
        let dilations = [
            info.dilation_rate_in_z,
            info.dilation_rate_in_y,
            info.dilation_rate_in_x,
        ];
        for axis in 0..3 {
            if !window(spatial[axis], kernel[axis], dilations[axis], pads[axis]) {
                return Err(Error::InvalidShape(WINDOW_TOO_LARGE));
            }
        }
    }
    Ok(())
}

fn check_convolution_transpose2d(
    info: &Convolution2DDescriptorInfo,
    source: &Tensor,
    weights: &Tensor,
    output_shape: &[usize],
) -> Result<()> {
    checks::float_tensor(source, "convolutions need a floating-point source")?;
    checks::same_data_type(
        source,
        weights,
        "the convolution weights need the source's data type",
    )?;
    let output = checks::shape_to_dims(output_shape)?;
    if output.len() != 4 {
        return Err(Error::InvalidShape(
            "the transposed convolution output shape needs 4 values",
        ));
    }
    let (Some(source_dims), Some(weight_dims)) = (
        checks::rank_exactly(source, 4, "the convolution source must have rank 4")?,
        checks::rank_exactly(weights, 4, "convolution weights must have rank 4")?,
    ) else {
        return Ok(());
    };
    let shapes = convolution2d_shapes(info, &source_dims, &weight_dims)?;
    let (output_batch, _, _, output_channels) = image_layout(info.data_layout, &output)?;
    if !checks::same_extent(shapes.channels, shapes.weight_outputs) {
        return Err(Error::InvalidShape(
            "the source channels must equal the weights' output channels",
        ));
    }
    let expected_channels = if shapes.weight_inputs < 0 {
        -1
    } else {
        isize::try_from(info.groups)
            .ok()
            .and_then(|groups| shapes.weight_inputs.checked_mul(groups))
            .unwrap_or(-1)
    };
    if !checks::same_extent(output_channels, expected_channels)
        || !checks::same_extent(output_batch, shapes.batch)
    {
        return Err(Error::InvalidShape(
            "the output shape's batch and channels must match the source and weights",
        ));
    }
    Ok(())
}

fn check_depthwise(source: &Tensor, weights: &Tensor, source_rank: Option<usize>) -> Result<()> {
    checks::float_tensor(source, "convolutions need a floating-point source")?;
    checks::same_data_type(
        source,
        weights,
        "the convolution weights need the source's data type",
    )?;
    match source_rank {
        Some(rank) => checks::rank_exactly(source, rank, "the depthwise source must have rank 4")?,
        None => checks::rank_at_least(source, 4, "the depthwise source needs rank 4 or more")?,
    };
    checks::rank_exactly(weights, 4, "depthwise weights must have rank 4")?;
    Ok(())
}

fn check_im_to_col(info: &ImToColDescriptorInfo, source: &Tensor) -> Result<()> {
    if source.data_type() != data_type::FLOAT32 {
        return Err(Error::InvalidDataType("im2col needs a float32 source"));
    }
    let Some(dims) = checks::rank_exactly(source, 4, "the im2col source must have rank 4")? else {
        return Ok(());
    };
    let (_, height, width, _) = image_layout(info.data_layout, &dims)?;
    let fits = |extent, kernel: usize, dilation, pads| {
        checks::window_fits(extent, kernel, dilation, pads)
    };
    if fits(
        height,
        info.kernel_height,
        info.dilation_rate_in_y,
        (info.padding_top, info.padding_bottom),
    ) && fits(
        width,
        info.kernel_width,
        info.dilation_rate_in_x,
        (info.padding_left, info.padding_right),
    ) {
        Ok(())
    } else {
        Err(Error::InvalidShape(WINDOW_TOO_LARGE))
    }
}

fn check_stencil(info: &StencilDescriptorInfo, source: &Tensor, weights: &Tensor) -> Result<()> {
    let source_dims =
        checks::rank_at_least(source, 4, "the stencil source needs rank 4 or more")?;
    let weight_dims = checks::rank_exactly(weights, 4, "stencil weights must have rank 4")?;
    let (Some(source_dims), Some(weight_dims), Some(padded)) =
        (source_dims, weight_dims, pads_window(info.padding_style))
    else {
        return Ok(());
    };
    let spatial = &source_dims[source_dims.len() - 4..];
    for axis in 0..4 {
        let pads = if padded {
            (
                info.explicit_padding[2 * axis],
                info.explicit_padding[2 * axis + 1],
            )
        } else {
            (0, 0)
        };
        if !window(
            spatial[axis],
            weight_dims[axis],
            info.dilation_rates[axis],
            pads,
        ) {
            return Err(Error::InvalidShape(WINDOW_TOO_LARGE));
        }
    }
    Ok(())
}

fn check_resize_layout(images: &Tensor, layout: usize) -> Result<()> {
    let expected_rank = match layout {
        tensor_named_data_layout::NHWC | tensor_named_data_layout::NCHW => 4,
        tensor_named_data_layout::HWC | tensor_named_data_layout::CHW => 3,
        _ => {
            return Err(Error::InvalidArgument(
                "resize takes NHWC, NCHW, HWC or CHW images",
            ))
        }
    };
    checks::rank_exactly(
        images,
        expected_rank,
        "NHWC and NCHW images need rank 4, HWC and CHW images rank 3",
    )
    .map(drop)
}

fn check_scatter_nd(updates: &Tensor, indices: &Tensor, shape: &[isize], batch: usize) -> Result<()> {
    checks::integer_tensor(indices, "indices must be an integer tensor")?;
    let (Some(update_dims), Some(index_dims)) = (updates.shape(), indices.shape()) else {
        return Ok(());
    };
    if batch >= update_dims.len() || batch >= index_dims.len() || batch >= shape.len() {
        return Err(Error::InvalidShape(
            "batch_dimensions must be less than the rank of every operand",
        ));
    }
    if !checks::same_dims(&update_dims[..batch], &index_dims[..batch])
        || !checks::same_dims(&shape[..batch], &index_dims[..batch])
    {
        return Err(Error::InvalidShape(
            "the operands must match along the batch dimensions",
        ));
    }
    let Some(depth) = index_dims.last().copied().and_then(checks::known) else {
        return Ok(());
    };
    let result_rank = shape.len() - batch;
    let index_rank = index_dims.len() - batch;
    if depth > result_rank {
        return Err(Error::InvalidShape(
            "the index depth exceeds the result's rank after the batch dimensions",
        ));
    }
    let slice = &shape[batch + depth..];
    let expected = [&index_dims[..index_dims.len() - 1], slice].concat();
    if update_dims.len() != batch + result_rank - depth + index_rank - 1
        || !checks::same_dims(&update_dims, &expected)
    {
        return Err(Error::InvalidShape(
            "the updates must have the indices' leading shape followed by the scattered slice",
        ));
    }
    Ok(())
}

fn check_scatter(updates: &Tensor, indices: &Tensor, shape: &[isize], axis: isize) -> Result<()> {
    checks::integer_tensor(indices, "indices must be an integer tensor")?;
    let axis = checks::normalized_axis(axis, shape.len())
        .ok_or(Error::InvalidShape("the axis is outside -rank..rank"))?;
    let index_dims = checks::rank_exactly(indices, 1, "scatter indices must have rank 1")?;
    let Some(update_dims) = updates.shape() else {
        return Ok(());
    };
    if update_dims.len() != shape.len() {
        return Err(Error::InvalidShape(
            "the updates must have the result's rank",
        ));
    }
    let mismatched = update_dims
        .iter()
        .zip(shape)
        .enumerate()
        .any(|(index, (update, result))| index != axis && !checks::same_extent(*update, *result));
    let count_mismatch = index_dims.is_some_and(|dims| !checks::same_extent(dims[0], update_dims[axis]));
    if mismatched || count_mismatch {
        return Err(Error::InvalidShape(
            "the updates must match the result except along the axis, where they need one entry per index",
        ));
    }
    Ok(())
}

fn check_scatter_along_axis(
    updates: &Tensor,
    indices: &Tensor,
    shape: &[isize],
    axis: isize,
) -> Result<()> {
    checks::integer_tensor(indices, "indices must be an integer tensor")?;
    let axis = checks::normalized_axis(axis, shape.len())
        .ok_or(Error::InvalidShape("the axis is outside -rank..rank"))?;
    for tensor in [updates, indices] {
        let Some(dims) = tensor.shape() else {
            continue;
        };
        let mismatched = dims.len() != shape.len()
            || dims
                .iter()
                .zip(shape)
                .enumerate()
                .any(|(index, (extent, result))| index != axis && !checks::same_extent(*extent, *result));
        if mismatched {
            return Err(Error::InvalidShape(
                "the updates and indices must match the result except along the axis",
            ));
        }
    }
    if let (Some(update_dims), Some(index_dims)) = (updates.shape(), indices.shape()) {
        if !checks::same_dims(&update_dims, &index_dims) {
            return Err(Error::InvalidShape(
                "the updates and indices need the same shape",
            ));
        }
    }
    Ok(())
}

fn check_sparse(
    storage_type: u64,
    tensors: &[&Tensor],
    shape: &[isize],
) -> Result<()> {
    let &[values, first, second] = tensors else {
        return Err(Error::InvalidArgument(
            "sparse tensors need [values, index0, index1]",
        ));
    };
    let &[rows, columns] = shape else {
        return Err(Error::InvalidShape("sparse tensors must have rank 2"));
    };
    if values.data_type() != data_type::FLOAT32 {
        return Err(Error::InvalidDataType("sparse values must be float32"));
    }
    let values_dims = checks::rank_exactly(values, 1, "sparse values must have rank 1")?;
    let nonzeros = values_dims.map_or(-1, |dims| dims[0]);
    let starts = match storage_type {
        sparse_storage_type::COO => nonzeros,
        sparse_storage_type::CSR => rows.checked_add(1).ok_or(Error::Overflow)?,
        _ => columns.checked_add(1).ok_or(Error::Overflow)?,
    };
    for (index, expected) in [(first, nonzeros), (second, starts)] {
        if !checks::is_index(index.data_type()) {
            return Err(Error::InvalidDataType(
                "sparse indices must be int32 or int64",
            ));
        }
        let dims = checks::rank_exactly(index, 1, "sparse indices must have rank 1")?;
        if dims.is_some_and(|dims| !checks::same_extent(dims[0], expected)) {
            return Err(Error::InvalidShape(
                "COO indices need one entry per value, and CSR/CSC starts one entry per row/column plus one",
            ));
        }
    }
    Ok(())
}

impl Graph {
/// Calls the `MPSGraph` framework counterpart for `convolution3d`.
    pub fn convolution3d(
        &self,
        source: &Tensor,
        weights: &Tensor,
        descriptor: &Convolution3DDescriptor,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[source, weights])?;
        check_convolution3d(descriptor.info(), source, weights)?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_convolution3d(
                self.as_ptr(),
                source.as_ptr(),
                weights.as_ptr(),
                descriptor.as_ptr(),
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[source, weights], "MPSGraph did not create the convolution")
    }

/// Calls the `MPSGraph` framework counterpart for `convolution_transpose2d`.
    pub fn convolution_transpose2d(
        &self,
        source: &Tensor,
        weights: &Tensor,
        output_shape: &[usize],
        descriptor: &Convolution2DDescriptor,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[source, weights])?;
        check_convolution_transpose2d(descriptor.info(), source, weights, output_shape)?;
        let name = optional_cstring(name);
        // SAFETY: all handles and slices remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_convolution_transpose2d(
                self.as_ptr(),
                source.as_ptr(),
                weights.as_ptr(),
                output_shape.as_ptr(),
                output_shape.len(),
                descriptor.as_ptr(),
                cstring_ptr(&name),
            )
        };
        self.output(
            ptr,
            &[source, weights],
            "MPSGraph did not create the transposed convolution",
        )
    }

/// Calls the `MPSGraph` framework counterpart for `cumulative_sum`.
    pub fn cumulative_sum(
        &self,
        tensor: &Tensor,
        axis: isize,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[tensor])?;
        checks::axis(tensor, axis)?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_cumulative_sum(
                self.as_ptr(),
                tensor.as_ptr(),
                axis,
                exclusive,
                reverse,
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[tensor], "MPSGraph did not create the cumulative sum")
    }

/// Calls the `MPSGraph` framework counterpart for `depthwise_convolution2d`.
    pub fn depthwise_convolution2d(
        &self,
        source: &Tensor,
        weights: &Tensor,
        descriptor: &DepthwiseConvolution2DDescriptor,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[source, weights])?;
        check_depthwise(source, weights, Some(4))?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_depthwise_convolution2d(
                self.as_ptr(),
                source.as_ptr(),
                weights.as_ptr(),
                descriptor.as_ptr(),
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[source, weights], "MPSGraph did not create the convolution")
    }

/// Calls the `MPSGraph` framework counterpart for `depthwise_convolution3d`.
    pub fn depthwise_convolution3d(
        &self,
        source: &Tensor,
        weights: &Tensor,
        descriptor: &DepthwiseConvolution3DDescriptor,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[source, weights])?;
        check_depthwise(source, weights, None)?;
        checks::axis(source, descriptor.info().channel_dimension_index)?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_depthwise_convolution3d(
                self.as_ptr(),
                source.as_ptr(),
                weights.as_ptr(),
                descriptor.as_ptr(),
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[source, weights], "MPSGraph did not create the convolution")
    }

/// Calls the `MPSGraph` framework counterpart for `fast_fourier_transform`.
    pub fn fast_fourier_transform(
        &self,
        tensor: &Tensor,
        axes: &[usize],
        descriptor: &FftDescriptor,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[tensor])?;
        let data_type = tensor.data_type();
        if !checks::is_float(data_type) && !checks::is_complex(data_type) {
            return Err(Error::InvalidDataType(
                "Fourier transforms need a floating-point or complex tensor",
            ));
        }
        axes_in_range(tensor, axes)?;
        let name = optional_cstring(name);
        // SAFETY: all handles and slices remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_fast_fourier_transform(
                self.as_ptr(),
                tensor.as_ptr(),
                axes.as_ptr(),
                axes.len(),
                descriptor.as_ptr(),
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[tensor], "MPSGraph did not create the Fourier transform")
    }

/// Calls the `MPSGraph` framework counterpart for `im_to_col`.
    pub fn im_to_col(
        &self,
        source: &Tensor,
        descriptor: &ImToColDescriptor,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[source])?;
        check_im_to_col(descriptor.info(), source)?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_im_to_col(
                self.as_ptr(),
                source.as_ptr(),
                descriptor.as_ptr(),
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[source], "MPSGraph did not create im2col")
    }

/// Calls the `MPSGraph` framework counterpart for `band_part`.
    pub fn band_part(
        &self,
        tensor: &Tensor,
        num_lower: isize,
        num_upper: isize,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[tensor])?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_band_part(
                self.as_ptr(),
                tensor.as_ptr(),
                num_lower,
                num_upper,
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[tensor], "MPSGraph did not create the band part")
    }

/// Calls the `MPSGraph` framework counterpart for `softmax_cross_entropy`.
    pub fn softmax_cross_entropy(
        &self,
        source: &Tensor,
        labels: &Tensor,
        axis: isize,
        reduction_type: u64,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[source, labels])?;
        checks::float_tensor(source, "the softmax cross entropy needs floating-point values")?;
        checks::elementwise(source, labels)?;
        checks::axis(source, axis)?;
        if reduction_type > loss_reduction_type::MEAN {
            return Err(Error::InvalidArgument("unknown MPSGraphLossReductionType"));
        }
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_softmax_cross_entropy(
                self.as_ptr(),
                source.as_ptr(),
                labels.as_ptr(),
                axis,
                reduction_type,
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[source, labels], "MPSGraph did not create the loss")
    }

/// Calls the `MPSGraph` framework counterpart for `matrix_inverse`.
    pub fn matrix_inverse(&self, tensor: &Tensor, name: Option<&str>) -> Result<Tensor> {
        self.check(&[tensor])?;
        checks::float_tensor(tensor, "matrix inversion needs a floating-point tensor")?;
        if let Some(dims) =
            checks::rank_at_least(tensor, 2, "matrix inversion needs a tensor of rank 2 or more")?
        {
            if !checks::same_extent(dims[dims.len() - 1], dims[dims.len() - 2]) {
                return Err(Error::InvalidShape(
                    "matrix inversion needs square matrices",
                ));
            }
        }
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_matrix_inverse(self.as_ptr(), tensor.as_ptr(), cstring_ptr(&name))
        };
        self.output(ptr, &[tensor], "MPSGraph did not create the matrix inverse")
    }

/// Calls the `MPSGraph` framework counterpart for `variable_bytes`.
    pub fn variable_bytes(
        &self,
        data: &[u8],
        shape: &[usize],
        data_type: u32,
        name: Option<&str>,
    ) -> Result<Tensor> {
        if !constant_data_type(data_type) {
            return Err(Error::UnsupportedDataType(data_type));
        }
        checks::shape_to_dims(shape)?;
        let expected = checked_byte_len(shape, data_type).ok_or(Error::Overflow)?;
        if data.len() != expected {
            return Err(Error::InvalidLength {
                expected,
                actual: data.len(),
            });
        }

        let name = optional_cstring(name);
        // SAFETY: all handles and slices remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_variable_data(
                self.as_ptr(),
                data.as_ptr().cast(),
                data.len(),
                shape.as_ptr(),
                shape.len(),
                data_type,
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[], "MPSGraph did not create the variable")
    }

/// Calls the `MPSGraph` framework counterpart for `variable_f32_slice`.
    pub fn variable_f32_slice(
        &self,
        values: &[f32],
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<Tensor> {
        // SAFETY: `values` is a contiguous slice of `f32` that may be viewed as bytes.
        let bytes = unsafe {
            core::slice::from_raw_parts(
                values.as_ptr().cast::<u8>(),
                core::mem::size_of_val(values),
            )
        };
        self.variable_bytes(bytes, shape, data_type::FLOAT32, name)
    }

/// Calls the `MPSGraph` framework counterpart for `read_variable`.
    pub fn read_variable(&self, variable: &Tensor, name: Option<&str>) -> Result<Tensor> {
        self.check(&[variable])?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_read_variable(self.as_ptr(), variable.as_ptr(), cstring_ptr(&name))
        };
        self.output(ptr, &[variable], "MPSGraph did not read the variable")
    }

/// Calls the `MPSGraph` framework counterpart for `assign_variable`.
    pub fn assign_variable(
        &self,
        variable: &Tensor,
        value: &Tensor,
        name: Option<&str>,
    ) -> Result<Operation> {
        self.check(&[variable, value])?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_assign_variable(
                self.as_ptr(),
                variable.as_ptr(),
                value.as_ptr(),
                cstring_ptr(&name),
            )
        };
        self.operation(ptr, &[variable, value], "MPSGraph did not create the assignment")
    }

/// Calls the `MPSGraph` framework counterpart for `non_maximum_suppression`.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn non_maximum_suppression(
        &self,
        boxes: &Tensor,
        scores: &Tensor,
        iou_threshold: f32,
        score_threshold: f32,
        per_class_suppression: bool,
        coordinate_mode: usize,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[boxes, scores])?;
        if boxes.data_type() != data_type::FLOAT32 || scores.data_type() != data_type::FLOAT32 {
            return Err(Error::InvalidDataType(
                "non-maximum suppression needs float32 boxes and scores",
            ));
        }
        if coordinate_mode > non_maximum_suppression_coordinate_mode::CENTERS_WIDTH_FIRST {
            return Err(Error::InvalidArgument(
                "unknown MPSGraphNonMaximumSuppressionCoordinateMode",
            ));
        }
        let boxes_dims = checks::rank_exactly(boxes, 3, "boxes must have shape [N, B, 4]")?;
        let scores_dims = checks::rank_exactly(scores, 3, "scores must have shape [N, B, K]")?;
        if let Some(dims) = &boxes_dims {
            checks::dims_match(boxes, &[dims[0], dims[1], 4], "boxes must have shape [N, B, 4]")?;
        }
        if let (Some(boxes_dims), Some(scores_dims)) = (boxes_dims, scores_dims) {
            if !checks::same_dims(&boxes_dims[..2], &scores_dims[..2]) {
                return Err(Error::InvalidShape(
                    "boxes and scores need the same batch and box counts",
                ));
            }
        }
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_non_maximum_suppression(
                self.as_ptr(),
                boxes.as_ptr(),
                scores.as_ptr(),
                iou_threshold,
                score_threshold,
                per_class_suppression,
                coordinate_mode,
                cstring_ptr(&name),
            )
        };
        self.output(
            ptr,
            &[boxes, scores],
            "MPSGraph did not create non-maximum suppression",
        )
    }

/// Calls the `MPSGraph` framework counterpart for `non_zero_indices`.
    pub fn non_zero_indices(&self, tensor: &Tensor, name: Option<&str>) -> Result<Tensor> {
        self.check(&[tensor])?;
        checks::rank_at_least(tensor, 1, "non-zero indices need a tensor of rank 1 or more")?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_non_zero_indices(self.as_ptr(), tensor.as_ptr(), cstring_ptr(&name))
        };
        self.output(ptr, &[tensor], "MPSGraph did not create non-zero indices")
    }

/// Calls the `MPSGraph` framework counterpart for `one_hot`.
    pub fn one_hot(
        &self,
        indices: &Tensor,
        depth: usize,
        data_type: u32,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[indices])?;
        if !constant_data_type(data_type) {
            return Err(Error::UnsupportedDataType(data_type));
        }
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_one_hot(
                self.as_ptr(),
                indices.as_ptr(),
                depth,
                data_type,
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[indices], "MPSGraph did not create the one-hot encoding")
    }

/// Calls the `MPSGraph` framework counterpart for `stochastic_gradient_descent`.
    pub fn stochastic_gradient_descent(
        &self,
        learning_rate: &Tensor,
        values: &Tensor,
        gradient: &Tensor,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[learning_rate, values, gradient])?;
        checks::elementwise(values, gradient)?;
        checks::elementwise(learning_rate, gradient)?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_stochastic_gradient_descent(
                self.as_ptr(),
                learning_rate.as_ptr(),
                values.as_ptr(),
                gradient.as_ptr(),
                cstring_ptr(&name),
            )
        };
        self.output(
            ptr,
            &[learning_rate, values, gradient],
            "MPSGraph did not create the optimizer step",
        )
    }

/// Calls the `MPSGraph` framework counterpart for `max_pooling4d`.
    pub fn max_pooling4d(
        &self,
        source: &Tensor,
        descriptor: &Pooling4DDescriptor,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[source])?;
        checks::rank_at_least(source, 4, "4D pooling needs a source of rank 4 or more")?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_max_pooling4d(
                self.as_ptr(),
                source.as_ptr(),
                descriptor.as_ptr(),
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[source], "MPSGraph did not create the pooling")
    }

/// Calls the `MPSGraph` framework counterpart for `max_pooling4d_return_indices`.
    pub fn max_pooling4d_return_indices(
        &self,
        source: &Tensor,
        descriptor: &Pooling4DDescriptor,
        name: Option<&str>,
    ) -> Result<(Tensor, Tensor)> {
        self.check(&[source])?;
        checks::rank_at_least(source, 4, "4D pooling needs a source of rank 4 or more")?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_graph_max_pooling4d_return_indices(
                self.as_ptr(),
                source.as_ptr(),
                descriptor.as_ptr(),
                cstring_ptr(&name),
            )
        };
        self.output_pair(box_handle, &[source], "MPSGraph did not create the pooling")
    }

/// Calls the `MPSGraph` framework counterpart for `quantize`.
    pub fn quantize(
        &self,
        tensor: &Tensor,
        scale: f64,
        zero_point: f64,
        data_type: u32,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[tensor])?;
        checks::float_tensor(tensor, "quantization needs a floating-point tensor")?;
        if !matches!(data_type, data_type::INT8 | data_type::UINT8) {
            return Err(Error::InvalidDataType(
                "scalar quantization produces int8 or uint8 values",
            ));
        }
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_quantize(
                self.as_ptr(),
                tensor.as_ptr(),
                scale,
                zero_point,
                data_type,
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[tensor], "MPSGraph did not create the quantization")
    }

/// Calls the `MPSGraph` framework counterpart for `dequantize`.
    pub fn dequantize(
        &self,
        tensor: &Tensor,
        scale: f64,
        zero_point: f64,
        data_type: u32,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[tensor])?;
        if !matches!(tensor.data_type(), data_type::INT8 | data_type::UINT8) {
            return Err(Error::InvalidDataType(
                "scalar dequantization reads int8 or uint8 values",
            ));
        }
        if !matches!(data_type, data_type::FLOAT16 | data_type::FLOAT32) {
            return Err(Error::InvalidDataType(
                "scalar dequantization produces float16 or float32 values",
            ));
        }
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_dequantize(
                self.as_ptr(),
                tensor.as_ptr(),
                scale,
                zero_point,
                data_type,
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[tensor], "MPSGraph did not create the dequantization")
    }

/// Calls the `MPSGraph` framework counterpart for `resize`.
    #[allow(clippy::too_many_arguments)]
    pub fn resize(
        &self,
        images: &Tensor,
        size: &[usize],
        mode: usize,
        center_result: bool,
        align_corners: bool,
        layout: usize,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[images])?;
        check_resize_layout(images, layout)?;
        if size.len() != 2 {
            return Err(Error::InvalidShape("resize needs a [height, width] size"));
        }
        checks::shape_to_dims(size)?;
        if mode > resize_mode::BILINEAR {
            return Err(Error::InvalidArgument("unknown MPSGraphResizeMode"));
        }
        let name = optional_cstring(name);
        // SAFETY: all handles and slices remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_resize(
                self.as_ptr(),
                images.as_ptr(),
                size.as_ptr(),
                size.len(),
                mode,
                center_result,
                align_corners,
                layout,
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[images], "MPSGraph did not create the resize")
    }

/// Calls the `MPSGraph` framework counterpart for `resize_nearest`.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn resize_nearest(
        &self,
        images: &Tensor,
        size_tensor: &Tensor,
        nearest_rounding_mode: usize,
        center_result: bool,
        align_corners: bool,
        layout: usize,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[images, size_tensor])?;
        check_resize_layout(images, layout)?;
        checks::index_vector(
            size_tensor,
            Some(2),
            "the size tensor must be a rank-1 int32 or int64 tensor of [height, width]",
        )?;
        if nearest_rounding_mode > resize_nearest_rounding_mode::ROUND_TO_ODD {
            return Err(Error::InvalidArgument(
                "unknown MPSGraphResizeNearestRoundingMode",
            ));
        }
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_resize_nearest(
                self.as_ptr(),
                images.as_ptr(),
                size_tensor.as_ptr(),
                nearest_rounding_mode,
                center_result,
                align_corners,
                layout,
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[images, size_tensor], "MPSGraph did not create the resize")
    }

/// Calls the `MPSGraph` framework counterpart for `sample_grid`.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_grid(
        &self,
        source: &Tensor,
        coordinates: &Tensor,
        layout: usize,
        normalize_coordinates: bool,
        relative_coordinates: bool,
        align_corners: bool,
        padding_mode: isize,
        sampling_mode: usize,
        constant_value: f64,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[source, coordinates])?;
        boundary(padding_mode)?;
        if sampling_mode > resize_mode::BILINEAR {
            return Err(Error::InvalidArgument(
                "the sampling mode must be NEAREST or BILINEAR",
            ));
        }
        let source_dims = checks::rank_exactly(source, 4, "grid sampling needs a rank-4 source")?;
        if !matches!(
            layout,
            tensor_named_data_layout::NHWC | tensor_named_data_layout::NCHW
        ) {
            return Err(Error::InvalidArgument("grid sampling takes NHWC or NCHW data"));
        }
        let coordinate_dims = checks::rank_exactly(
            coordinates,
            4,
            "grid coordinates must have shape [N, H, W, 2]",
        )?;
        if let Some(dims) = &coordinate_dims {
            checks::dims_match(
                coordinates,
                &[dims[0], dims[1], dims[2], 2],
                "grid coordinates must have shape [N, H, W, 2]",
            )?;
        }
        if let (Some(source_dims), Some(coordinate_dims)) = (source_dims, coordinate_dims) {
            if !checks::same_extent(source_dims[0], coordinate_dims[0]) {
                return Err(Error::InvalidShape(
                    "the source and coordinates need the same batch size",
                ));
            }
        }
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_sample_grid(
                self.as_ptr(),
                source.as_ptr(),
                coordinates.as_ptr(),
                layout,
                normalize_coordinates,
                relative_coordinates,
                align_corners,
                padding_mode,
                sampling_mode,
                constant_value,
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[source, coordinates], "MPSGraph did not create the grid sample")
    }

/// Calls the `MPSGraph` framework counterpart for `scatter_nd`.
    pub fn scatter_nd(
        &self,
        updates: &Tensor,
        indices: &Tensor,
        shape: &[usize],
        batch_dimensions: usize,
        mode: isize,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[updates, indices])?;
        check_scatter_mode(mode)?;
        check_scatter_nd(updates, indices, &checks::shape_to_dims(shape)?, batch_dimensions)?;
        let name = optional_cstring(name);
        // SAFETY: all handles and slices remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_scatter_nd(
                self.as_ptr(),
                updates.as_ptr(),
                indices.as_ptr(),
                shape.as_ptr(),
                shape.len(),
                batch_dimensions,
                mode,
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[updates, indices], "MPSGraph did not create the scatter")
    }

/// Calls the `MPSGraph` framework counterpart for `scatter`.
    pub fn scatter(
        &self,
        updates: &Tensor,
        indices: &Tensor,
        shape: &[usize],
        axis: isize,
        mode: isize,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[updates, indices])?;
        check_scatter_mode(mode)?;
        check_scatter(updates, indices, &checks::shape_to_dims(shape)?, axis)?;
        let name = optional_cstring(name);
        // SAFETY: all handles and slices remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_scatter(
                self.as_ptr(),
                updates.as_ptr(),
                indices.as_ptr(),
                shape.as_ptr(),
                shape.len(),
                axis,
                mode,
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[updates, indices], "MPSGraph did not create the scatter")
    }

/// Calls the `MPSGraph` framework counterpart for `scatter_along_axis`.
    pub fn scatter_along_axis(
        &self,
        axis: isize,
        updates: &Tensor,
        indices: &Tensor,
        shape: &[usize],
        mode: isize,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[updates, indices])?;
        check_scatter_mode(mode)?;
        check_scatter_along_axis(updates, indices, &checks::shape_to_dims(shape)?, axis)?;
        let name = optional_cstring(name);
        // SAFETY: all handles and slices remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_scatter_along_axis(
                self.as_ptr(),
                axis,
                updates.as_ptr(),
                indices.as_ptr(),
                shape.as_ptr(),
                shape.len(),
                mode,
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[updates, indices], "MPSGraph did not create the scatter")
    }

/// Calls the `MPSGraph` framework counterpart for `sort`.
    pub fn sort(
        &self,
        tensor: &Tensor,
        axis: isize,
        descending: bool,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[tensor])?;
        checks::axis(tensor, axis)?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_sort(
                self.as_ptr(),
                tensor.as_ptr(),
                axis,
                descending,
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[tensor], "MPSGraph did not create the sort")
    }

/// Calls the `MPSGraph` framework counterpart for `arg_sort`.
    pub fn arg_sort(
        &self,
        tensor: &Tensor,
        axis: isize,
        descending: bool,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[tensor])?;
        checks::axis(tensor, axis)?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_arg_sort(
                self.as_ptr(),
                tensor.as_ptr(),
                axis,
                descending,
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[tensor], "MPSGraph did not create the sort")
    }

/// Calls the `MPSGraph` framework counterpart for `sparse_tensor_with_descriptor`.
    pub fn sparse_tensor_with_descriptor(
        &self,
        descriptor: &CreateSparseDescriptor,
        tensors: &[&Tensor],
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(tensors)?;
        check_sparse(
            descriptor.info().storage_type,
            tensors,
            &checks::shape_to_dims(shape)?,
        )?;
        let name = optional_cstring(name);
        let handles = tensors
            .iter()
            .map(|tensor| tensor.as_ptr())
            .collect::<Vec<_>>();
        // SAFETY: all handles and slices remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_sparse_tensor_with_descriptor(
                self.as_ptr(),
                descriptor.as_ptr(),
                handles.as_ptr(),
                handles.len(),
                shape.as_ptr(),
                shape.len(),
                cstring_ptr(&name),
            )
        };
        self.output(ptr, tensors, "MPSGraph did not create the sparse tensor")
    }

/// Calls the `MPSGraph` framework counterpart for `stencil`.
    pub fn stencil(
        &self,
        source: &Tensor,
        weights: &Tensor,
        descriptor: &StencilDescriptor,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[source, weights])?;
        check_stencil(descriptor.info(), source, weights)?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_stencil(
                self.as_ptr(),
                source.as_ptr(),
                weights.as_ptr(),
                descriptor.as_ptr(),
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[source, weights], "MPSGraph did not create the stencil")
    }

/// Calls the `MPSGraph` framework counterpart for `top_k_gradient`.
    pub fn top_k_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        k: usize,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[gradient, source])?;
        checks::same_data_type(
            gradient,
            source,
            "the gradient needs the source's data type",
        )?;
        if k == 0 {
            return Err(Error::InvalidArgument("k must be at least 1"));
        }
        if let Some(dims) =
            checks::rank_at_least(source, 1, "top-k needs a source of rank 1 or more")?
        {
            let last = dims.len() - 1;
            if checks::known(dims[last]).is_some_and(|extent| k > extent) {
                return Err(Error::InvalidShape(
                    "k is larger than the source's last dimension",
                ));
            }
            let mut expected = dims;
            expected[last] = isize::try_from(k).map_err(|_| Error::Overflow)?;
            checks::dims_match(
                gradient,
                &expected,
                "the gradient must have the source's shape with k in the last dimension",
            )?;
        }
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_topk_gradient(
                self.as_ptr(),
                gradient.as_ptr(),
                source.as_ptr(),
                k,
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[gradient, source], "MPSGraph did not create the top-k gradient")
    }
}

impl ExecutionDescriptor {
    /// # Safety
    ///
    /// `event_handle` must be a valid `id<MTLSharedEvent>` for the lifetime of the call.
    pub unsafe fn wait_for_shared_event_raw(
        &self,
        event_handle: *mut c_void,
        value: u64,
    ) -> Result<()> {
        // SAFETY: caller guarantees `event_handle` is a valid shared-event pointer.
        let ok = unsafe {
            ffi::mpsgraph_execution_descriptor_wait_for_event(self.as_ptr(), event_handle, value)
        };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed(
                "failed to register execution descriptor shared-event wait",
            ))
        }
    }

    /// # Safety
    ///
    /// `event_handle` must be a valid `id<MTLSharedEvent>` for the lifetime of the call.
    pub unsafe fn signal_shared_event_raw(
        &self,
        event_handle: *mut c_void,
        execution_stage: u64,
        value: u64,
    ) -> Result<()> {
        // SAFETY: caller guarantees `event_handle` is a valid shared-event pointer.
        let ok = unsafe {
            ffi::mpsgraph_execution_descriptor_signal_event(
                self.as_ptr(),
                event_handle,
                execution_stage,
                value,
            )
        };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed(
                "failed to register execution descriptor shared-event signal",
            ))
        }
    }
}

impl ExecutableExecutionDescriptor {
    /// # Safety
    ///
    /// `event_handle` must be a valid `id<MTLSharedEvent>` for the lifetime of the call.
    pub unsafe fn wait_for_shared_event_raw(
        &self,
        event_handle: *mut c_void,
        value: u64,
    ) -> Result<()> {
        // SAFETY: caller guarantees `event_handle` is a valid shared-event pointer.
        let ok = unsafe {
            ffi::mpsgraph_executable_execution_descriptor_wait_for_event(
                self.as_ptr(),
                event_handle,
                value,
            )
        };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed(
                "failed to register executable execution descriptor shared-event wait",
            ))
        }
    }

    /// # Safety
    ///
    /// `event_handle` must be a valid `id<MTLSharedEvent>` for the lifetime of the call.
    pub unsafe fn signal_shared_event_raw(
        &self,
        event_handle: *mut c_void,
        execution_stage: u64,
        value: u64,
    ) -> Result<()> {
        // SAFETY: caller guarantees `event_handle` is a valid shared-event pointer.
        let ok = unsafe {
            ffi::mpsgraph_executable_execution_descriptor_signal_event(
                self.as_ptr(),
                event_handle,
                execution_stage,
                value,
            )
        };
        if ok {
            Ok(())
        } else {
            Err(Error::OperationFailed(
                "failed to register executable execution descriptor shared-event signal",
            ))
        }
    }
}
