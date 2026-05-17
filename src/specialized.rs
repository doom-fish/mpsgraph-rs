use crate::error::{Error, Result};
use crate::execution::{ExecutableExecutionDescriptor, ExecutionDescriptor};
use crate::ffi;
use crate::graph::{
    data_type, data_type_size, padding_mode, padding_style, tensor_named_data_layout,
    Convolution2DDescriptor, Graph, Tensor,
};
use crate::types::{collect_owned_tensors, Operation, ShapedType};
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
        Some(Tensor::from_raw(ptr))
    }
}

fn wrap_operation(ptr: *mut c_void) -> Option<Operation> {
    if ptr.is_null() {
        None
    } else {
        Some(Operation::from_raw(ptr))
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

macro_rules! opaque_handle {
    ($name:ident) => {
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
            #[must_use]
            pub const fn as_ptr(&self) -> *mut c_void {
                self.ptr
            }
        }
    };
}

/// `MPSGraphExecutionStage` constants.
pub mod execution_stage {
    pub const COMPLETED: u64 = 0;
}

/// `MPSGraphReductionMode` constants.
pub mod reduction_mode {
    pub const MIN: usize = 0;
    pub const MAX: usize = 1;
    pub const SUM: usize = 2;
    pub const PRODUCT: usize = 3;
    pub const ARGUMENT_MIN: usize = 4;
    pub const ARGUMENT_MAX: usize = 5;
}

/// `MPSGraphPoolingReturnIndicesMode` constants.
pub mod pooling_return_indices_mode {
    pub const NONE: usize = 0;
    pub const GLOBAL_FLATTEN_1D: usize = 1;
    pub const GLOBAL_FLATTEN_2D: usize = 2;
    pub const GLOBAL_FLATTEN_3D: usize = 3;
    pub const GLOBAL_FLATTEN_4D: usize = 4;
    pub const LOCAL_FLATTEN_1D: usize = 5;
    pub const LOCAL_FLATTEN_2D: usize = 6;
    pub const LOCAL_FLATTEN_3D: usize = 7;
    pub const LOCAL_FLATTEN_4D: usize = 8;
}

/// `MPSGraphFFTScalingMode` constants.
pub mod fft_scaling_mode {
    pub const NONE: usize = 0;
    pub const SIZE: usize = 1;
    pub const UNITARY: usize = 2;
}

/// `MPSGraphLossReductionType` constants.
pub mod loss_reduction_type {
    pub const NONE: u64 = 0;
    pub const AXIS: u64 = 0;
    pub const SUM: u64 = 1;
    pub const MEAN: u64 = 2;
}

/// `MPSGraphNonMaximumSuppressionCoordinateMode` constants.
pub mod non_maximum_suppression_coordinate_mode {
    pub const CORNERS_HEIGHT_FIRST: usize = 0;
    pub const CORNERS_WIDTH_FIRST: usize = 1;
    pub const CENTERS_HEIGHT_FIRST: usize = 2;
    pub const CENTERS_WIDTH_FIRST: usize = 3;
}

/// `MPSGraphResizeMode` constants.
pub mod resize_mode {
    pub const NEAREST: usize = 0;
    pub const BILINEAR: usize = 1;
}

/// `MPSGraphResizeNearestRoundingMode` constants.
pub mod resize_nearest_rounding_mode {
    pub const ROUND_PREFER_CEIL: usize = 0;
    pub const ROUND_PREFER_FLOOR: usize = 1;
    pub const CEIL: usize = 2;
    pub const FLOOR: usize = 3;
    pub const ROUND_TO_EVEN: usize = 4;
    pub const ROUND_TO_ODD: usize = 5;
}

/// `MPSGraphScatterMode` constants.
pub mod scatter_mode {
    pub const ADD: isize = 0;
    pub const SUB: isize = 1;
    pub const MUL: isize = 2;
    pub const DIV: isize = 3;
    pub const MIN: isize = 4;
    pub const MAX: isize = 5;
    pub const SET: isize = 6;
}

/// `MPSGraphSparseStorageType` constants.
pub mod sparse_storage_type {
    pub const COO: u64 = 0;
    pub const CSC: u64 = 1;
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

    #[must_use]
    pub fn as_object(&self) -> Object {
        Object::retain_from(self.ptr)
    }
}

opaque_handle!(VariableOp);
impl VariableOp {
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

    #[must_use]
    pub fn data_type(&self) -> u32 {
        // SAFETY: `self.ptr` is a live variable-op handle.
        unsafe { ffi::mpsgraph_variable_op_data_type(self.ptr) }
    }

    #[must_use]
    pub fn as_object(&self) -> Object {
        Object::retain_from(self.ptr)
    }

    #[must_use]
    pub fn as_operation(&self) -> Operation {
        // SAFETY: `self.ptr` is a live variable-op handle and retains as an operation wrapper.
        let ptr = unsafe { ffi::mpsgraph_object_retain(self.ptr) };
        Operation::from_raw(ptr)
    }
}

impl ShapedType {
    #[must_use]
    pub fn as_graph_type(&self) -> GraphType {
        GraphType::retain_from(self.as_ptr())
    }
}

impl Operation {
    #[must_use]
    pub fn as_variable(&self) -> Option<VariableOp> {
        // SAFETY: `self.ptr` is a live operation handle.
        let ptr = unsafe { ffi::mpsgraph_operation_as_variable(self.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(VariableOp { ptr })
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Convolution3DDescriptorInfo {
    pub stride_in_x: usize,
    pub stride_in_y: usize,
    pub stride_in_z: usize,
    pub dilation_rate_in_x: usize,
    pub dilation_rate_in_y: usize,
    pub dilation_rate_in_z: usize,
    pub groups: usize,
    pub padding_left: usize,
    pub padding_right: usize,
    pub padding_top: usize,
    pub padding_bottom: usize,
    pub padding_front: usize,
    pub padding_back: usize,
    pub padding_style: usize,
    pub data_layout: usize,
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

opaque_handle!(Convolution3DDescriptor);
impl Convolution3DDescriptor {
    #[must_use]
    pub fn new(info: Convolution3DDescriptorInfo) -> Option<Self> {
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
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DepthwiseConvolution2DDescriptorInfo {
    pub stride_in_x: usize,
    pub stride_in_y: usize,
    pub dilation_rate_in_x: usize,
    pub dilation_rate_in_y: usize,
    pub padding_left: usize,
    pub padding_right: usize,
    pub padding_top: usize,
    pub padding_bottom: usize,
    pub padding_style: usize,
    pub data_layout: usize,
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

opaque_handle!(DepthwiseConvolution2DDescriptor);
impl DepthwiseConvolution2DDescriptor {
    #[must_use]
    pub fn new(info: DepthwiseConvolution2DDescriptorInfo) -> Option<Self> {
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
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DepthwiseConvolution3DDescriptorInfo {
    pub strides: [usize; 3],
    pub dilation_rates: [usize; 3],
    pub padding_values: [usize; 6],
    pub padding_style: usize,
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

opaque_handle!(DepthwiseConvolution3DDescriptor);
impl DepthwiseConvolution3DDescriptor {
    #[must_use]
    pub fn new(info: DepthwiseConvolution3DDescriptorInfo) -> Option<Self> {
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
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FftDescriptorInfo {
    pub inverse: bool,
    pub scaling_mode: usize,
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

opaque_handle!(FftDescriptor);
impl FftDescriptor {
    #[must_use]
    pub fn new(info: FftDescriptorInfo) -> Option<Self> {
        // SAFETY: all arguments are POD configuration values.
        let ptr = unsafe {
            ffi::mpsgraph_fft_descriptor_new(
                info.inverse,
                info.scaling_mode,
                info.round_to_odd_hermitean,
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ImToColDescriptorInfo {
    pub kernel_width: usize,
    pub kernel_height: usize,
    pub stride_in_x: usize,
    pub stride_in_y: usize,
    pub dilation_rate_in_x: usize,
    pub dilation_rate_in_y: usize,
    pub padding_left: usize,
    pub padding_right: usize,
    pub padding_top: usize,
    pub padding_bottom: usize,
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

opaque_handle!(ImToColDescriptor);
impl ImToColDescriptor {
    #[must_use]
    pub fn new(info: ImToColDescriptorInfo) -> Option<Self> {
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
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Pooling4DDescriptorInfo {
    pub kernel_sizes: [usize; 4],
    pub strides: [usize; 4],
    pub dilation_rates: [usize; 4],
    pub padding_values: [usize; 8],
    pub padding_style: usize,
    pub ceil_mode: bool,
    pub include_zero_pad_to_average: bool,
    pub return_indices_mode: usize,
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

opaque_handle!(Pooling4DDescriptor);
impl Pooling4DDescriptor {
    #[must_use]
    pub fn new(info: Pooling4DDescriptorInfo) -> Option<Self> {
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
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }
}

opaque_handle!(CreateSparseDescriptor);
impl CreateSparseDescriptor {
    #[must_use]
    pub fn new(storage_type: u64, data_type: u32) -> Option<Self> {
        // SAFETY: all arguments are POD configuration values.
        let ptr = unsafe { ffi::mpsgraph_sparse_descriptor_new(storage_type, data_type) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StencilDescriptorInfo {
    pub reduction_mode: usize,
    pub offsets: [isize; 4],
    pub strides: [usize; 4],
    pub dilation_rates: [usize; 4],
    pub explicit_padding: [usize; 8],
    pub boundary_mode: isize,
    pub padding_style: usize,
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

opaque_handle!(StencilDescriptor);
impl StencilDescriptor {
    #[must_use]
    pub fn new(info: StencilDescriptorInfo) -> Option<Self> {
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
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }
}

impl Graph {
    #[must_use]
    pub fn convolution3d(
        &self,
        source: &Tensor,
        weights: &Tensor,
        descriptor: &Convolution3DDescriptor,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn convolution_transpose2d(
        &self,
        source: &Tensor,
        weights: &Tensor,
        output_shape: &[usize],
        descriptor: &Convolution2DDescriptor,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn cumulative_sum(
        &self,
        tensor: &Tensor,
        axis: isize,
        exclusive: bool,
        reverse: bool,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn depthwise_convolution2d(
        &self,
        source: &Tensor,
        weights: &Tensor,
        descriptor: &DepthwiseConvolution2DDescriptor,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn depthwise_convolution3d(
        &self,
        source: &Tensor,
        weights: &Tensor,
        descriptor: &DepthwiseConvolution3DDescriptor,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn fast_fourier_transform(
        &self,
        tensor: &Tensor,
        axes: &[usize],
        descriptor: &FftDescriptor,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn im_to_col(
        &self,
        source: &Tensor,
        descriptor: &ImToColDescriptor,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn band_part(
        &self,
        tensor: &Tensor,
        num_lower: isize,
        num_upper: isize,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn softmax_cross_entropy(
        &self,
        source: &Tensor,
        labels: &Tensor,
        axis: isize,
        reduction_type: u64,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn matrix_inverse(&self, tensor: &Tensor, name: Option<&str>) -> Option<Tensor> {
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_matrix_inverse(self.as_ptr(), tensor.as_ptr(), cstring_ptr(&name))
        };
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn variable_bytes(
        &self,
        data: &[u8],
        shape: &[usize],
        data_type: u32,
        name: Option<&str>,
    ) -> Option<Tensor> {
        let expected = checked_byte_len(shape, data_type)?;
        if data.len() != expected {
            return None;
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
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn variable_f32_slice(
        &self,
        values: &[f32],
        shape: &[usize],
        name: Option<&str>,
    ) -> Option<Tensor> {
        // SAFETY: `values` is a contiguous slice of `f32` that may be viewed as bytes.
        let bytes = unsafe {
            core::slice::from_raw_parts(
                values.as_ptr().cast::<u8>(),
                core::mem::size_of_val(values),
            )
        };
        self.variable_bytes(bytes, shape, data_type::FLOAT32, name)
    }

    #[must_use]
    pub fn read_variable(&self, variable: &Tensor, name: Option<&str>) -> Option<Tensor> {
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_read_variable(self.as_ptr(), variable.as_ptr(), cstring_ptr(&name))
        };
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn assign_variable(
        &self,
        variable: &Tensor,
        value: &Tensor,
        name: Option<&str>,
    ) -> Option<Operation> {
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
        wrap_operation(ptr)
    }

    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn non_maximum_suppression(
        &self,
        boxes: &Tensor,
        scores: &Tensor,
        iou_threshold: f32,
        score_threshold: f32,
        per_class_suppression: bool,
        coordinate_mode: usize,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn non_zero_indices(&self, tensor: &Tensor, name: Option<&str>) -> Option<Tensor> {
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_non_zero_indices(self.as_ptr(), tensor.as_ptr(), cstring_ptr(&name))
        };
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn one_hot(
        &self,
        indices: &Tensor,
        depth: usize,
        data_type: u32,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn stochastic_gradient_descent(
        &self,
        learning_rate: &Tensor,
        values: &Tensor,
        gradient: &Tensor,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn max_pooling4d(
        &self,
        source: &Tensor,
        descriptor: &Pooling4DDescriptor,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn max_pooling4d_return_indices(
        &self,
        source: &Tensor,
        descriptor: &Pooling4DDescriptor,
        name: Option<&str>,
    ) -> Option<(Tensor, Tensor)> {
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
        wrap_tensor_pair(box_handle)
    }

    #[must_use]
    pub fn quantize(
        &self,
        tensor: &Tensor,
        scale: f64,
        zero_point: f64,
        data_type: u32,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn dequantize(
        &self,
        tensor: &Tensor,
        scale: f64,
        zero_point: f64,
        data_type: u32,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

    #[must_use]
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
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn resize_nearest(
        &self,
        images: &Tensor,
        size_tensor: &Tensor,
        nearest_rounding_mode: usize,
        center_result: bool,
        align_corners: bool,
        layout: usize,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

    #[must_use]
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
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn scatter_nd(
        &self,
        updates: &Tensor,
        indices: &Tensor,
        shape: &[usize],
        batch_dimensions: usize,
        mode: isize,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn scatter(
        &self,
        updates: &Tensor,
        indices: &Tensor,
        shape: &[usize],
        axis: isize,
        mode: isize,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn scatter_along_axis(
        &self,
        axis: isize,
        updates: &Tensor,
        indices: &Tensor,
        shape: &[usize],
        mode: isize,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn sort(
        &self,
        tensor: &Tensor,
        axis: isize,
        descending: bool,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn arg_sort(
        &self,
        tensor: &Tensor,
        axis: isize,
        descending: bool,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn sparse_tensor_with_descriptor(
        &self,
        descriptor: &CreateSparseDescriptor,
        tensors: &[&Tensor],
        shape: &[usize],
        name: Option<&str>,
    ) -> Option<Tensor> {
        let name = optional_cstring(name);
        let handles = tensors.iter().map(|tensor| tensor.as_ptr()).collect::<Vec<_>>();
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
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn stencil(
        &self,
        source: &Tensor,
        weights: &Tensor,
        descriptor: &StencilDescriptor,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

    #[must_use]
    pub fn top_k_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        k: usize,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
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
        let ok = unsafe { ffi::mpsgraph_execution_descriptor_wait_for_event(self.as_ptr(), event_handle, value) };
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
