use crate::checks::{self, Dims};
use crate::data::TensorData;
use crate::error::{Error, Result};
use crate::ffi;
use crate::types::Operation;
use apple_metal::{CommandQueue, MetalDevice};
use core::cell::RefCell;
use core::ffi::{c_char, c_void};
use core::marker::PhantomData;
use core::ptr;
use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::sync::atomic::{AtomicU64, Ordering};


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
    pub const BFLOAT16: u32 = 0x9000_0010;
    pub const COMPLEX_FLOAT16: u32 = 0x1100_0020;
    pub const COMPLEX_FLOAT32: u32 = 0x1100_0040;
    pub const COMPLEX_BFLOAT16: u32 = 0x9100_0020;
    pub const INT2: u32 = 0x2000_0002;
    pub const INT4: u32 = 0x2000_0004;
    pub const UINT2: u32 = 0x0000_0002;
    pub const UINT4: u32 = 0x0000_0004;
    pub const FLOAT8_E4M3: u32 = 0x1043_0008;
    pub const FLOAT8_E5M2: u32 = 0x1052_0008;
    pub const FLOAT8_E8M0: u32 = 0x1080_0008;
    pub const FLOAT4_E2M1: u32 = 0x1021_0004;
}

#[must_use]
pub const fn data_type_bits(data_type: u32) -> Option<usize> {
    use data_type::{
        BFLOAT16, BOOL, COMPLEX_BFLOAT16, COMPLEX_FLOAT16, COMPLEX_FLOAT32, FLOAT16, FLOAT32,
        FLOAT4_E2M1, FLOAT8_E4M3, FLOAT8_E5M2, FLOAT8_E8M0, INT16, INT2, INT32, INT4, INT64, INT8,
        UINT16, UINT2, UINT32, UINT4, UINT64, UINT8, UNORM8,
    };
    match data_type {
        INT2 | UINT2 => Some(2),
        INT4 | UINT4 | FLOAT4_E2M1 => Some(4),
        INT8 | UINT8 | BOOL | UNORM8 | FLOAT8_E4M3 | FLOAT8_E5M2 | FLOAT8_E8M0 => Some(8),
        FLOAT16 | BFLOAT16 | INT16 | UINT16 => Some(16),
        FLOAT32 | INT32 | UINT32 | COMPLEX_FLOAT16 | COMPLEX_BFLOAT16 => Some(32),
        INT64 | UINT64 | COMPLEX_FLOAT32 => Some(64),
        _ => None,
    }
}

/// Return the byte width of a supported `MPSDataType`.
#[must_use]
pub const fn data_type_size(data_type: u32) -> Option<usize> {
    match data_type_bits(data_type) {
        Some(bits) if bits % 8 == 0 => Some(bits / 8),
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

            pub(crate) const fn info(&self) -> &$info {
                &self.info
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

pub(crate) fn checked_byte_len(shape: &[usize], data_type: u32) -> Option<usize> {
    let bits = data_type_bits(data_type)?;
    shape
        .iter()
        .try_fold(bits, |acc, dimension| acc.checked_mul(*dimension))
        .map(|bits| bits.div_ceil(8))
        .filter(|bytes| isize::try_from(*bytes).is_ok())
}

fn optional_cstring(name: Option<&str>) -> Option<CString> {
    name.and_then(|value| CString::new(value).ok())
}

#[allow(clippy::ref_option)]
fn cstring_ptr(value: &Option<CString>) -> *const c_char {
    value.as_ref().map_or(ptr::null(), |value| value.as_ptr())
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
    ($fn_name:ident, $ffi_name:ident, $compatible:path) => {
/// Calls the `MPSGraph` framework counterpart for this method.
        pub fn $fn_name(
            &self,
            primary: &Tensor,
            secondary: &Tensor,
            name: Option<&str>,
        ) -> Result<Tensor> {
            self.check(&[primary, secondary])?;
            $compatible(primary, secondary)?;
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
            self.output(
                ptr,
                &[primary, secondary],
                concat!("MPSGraph did not create ", stringify!($fn_name)),
            )
        }
    };
}

macro_rules! impl_unary_tensor_op {
    ($fn_name:ident, $ffi_name:ident) => {
/// Calls the `MPSGraph` framework counterpart for this method.
        pub fn $fn_name(&self, tensor: &Tensor, name: Option<&str>) -> Result<Tensor> {
            self.check(&[tensor])?;
            let name = optional_cstring(name);
            // SAFETY: All pointers originate from safe wrappers and remain alive for the duration of the call.
            let ptr = unsafe { ffi::$ffi_name(self.ptr, tensor.as_ptr(), cstring_ptr(&name)) };
            self.output(
                ptr,
                &[tensor],
                concat!("MPSGraph did not create ", stringify!($fn_name)),
            )
        }
    };
}

macro_rules! impl_axes_tensor_op {
    ($fn_name:ident, $ffi_name:ident) => {
/// Calls the `MPSGraph` framework counterpart for this method.
        pub fn $fn_name(
            &self,
            tensor: &Tensor,
            axes: &[usize],
            name: Option<&str>,
        ) -> Result<Tensor> {
            self.check(&[tensor])?;
            axes_in_range(tensor, axes)?;
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
            self.output(
                ptr,
                &[tensor],
                concat!("MPSGraph did not create ", stringify!($fn_name)),
            )
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

pub(crate) const fn valid_padding_style(style: usize) -> bool {
    style <= padding_style::ONNX_SAME_LOWER
}

pub(crate) const fn pads_window(style: usize) -> Option<bool> {
    match style {
        padding_style::EXPLICIT | padding_style::EXPLICIT_OFFSET => Some(true),
        padding_style::TF_VALID => Some(false),
        _ => None,
    }
}

pub(crate) fn image_layout(layout: usize, dims: &[isize]) -> Result<(isize, isize, isize, isize)> {
    let &[first, second, third, fourth] = dims else {
        return Err(Error::InvalidShape("2D image tensors must have rank 4"));
    };
    match layout {
        tensor_named_data_layout::NHWC => Ok((first, second, third, fourth)),
        tensor_named_data_layout::NCHW => Ok((first, third, fourth, second)),
        _ => Err(Error::InvalidArgument("the data layout must be NHWC or NCHW")),
    }
}

descriptor_handle!(Convolution2DDescriptor, Convolution2DDescriptorInfo);
impl Convolution2DDescriptor {
/// Calls the `MPSGraph` framework counterpart for `new`.
    pub fn new(info: Convolution2DDescriptorInfo) -> Result<Self> {
        if [
            info.stride_in_x,
            info.stride_in_y,
            info.dilation_rate_in_x,
            info.dilation_rate_in_y,
            info.groups,
        ]
        .contains(&0)
        {
            return Err(Error::InvalidArgument(
                "convolution strides, dilation rates and groups must be at least 1",
            ));
        }
        if !valid_padding_style(info.padding_style) {
            return Err(Error::InvalidArgument("unknown MPSGraphPaddingStyle"));
        }
        if !matches!(
            info.data_layout,
            tensor_named_data_layout::NHWC | tensor_named_data_layout::NCHW
        ) || !matches!(
            info.weights_layout,
            tensor_named_data_layout::HWIO | tensor_named_data_layout::OIHW
        ) {
            return Err(Error::InvalidArgument(
                "2D convolutions take NHWC or NCHW data and HWIO or OIHW weights",
            ));
        }
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
            Err(Error::OperationFailed(
                "MPSGraph did not create the convolution descriptor",
            ))
        } else {
            Ok(Self { ptr, info })
        }
    }
}

pub(crate) struct Convolution2DShapes {
    pub(crate) batch: isize,
    pub(crate) height: isize,
    pub(crate) width: isize,
    pub(crate) channels: isize,
    pub(crate) kernel_height: isize,
    pub(crate) kernel_width: isize,
    pub(crate) weight_inputs: isize,
    pub(crate) weight_outputs: isize,
}

pub(crate) fn convolution2d_shapes(
    info: &Convolution2DDescriptorInfo,
    source: &[isize],
    weights: &[isize],
) -> Result<Convolution2DShapes> {
    let (batch, height, width, channels) = image_layout(info.data_layout, source)?;
    let &[w0, w1, w2, w3] = weights else {
        return Err(Error::InvalidShape("convolution weights must have rank 4"));
    };
    let (kernel_height, kernel_width, weight_inputs, weight_outputs) = match info.weights_layout
    {
        tensor_named_data_layout::HWIO => (w0, w1, w2, w3),
        _ => (w2, w3, w1, w0),
    };
    Ok(Convolution2DShapes {
        batch,
        height,
        width,
        channels,
        kernel_height,
        kernel_width,
        weight_inputs,
        weight_outputs,
    })
}

fn check_groups(channels: isize, weight_inputs: isize, weight_outputs: isize, groups: usize) -> Result<()> {
    if let (Some(channels), Some(inputs)) = (checks::known(channels), checks::known(weight_inputs)) {
        if inputs.checked_mul(groups) != Some(channels) {
            return Err(Error::InvalidShape(
                "the input channels must equal groups times the weights' input channels",
            ));
        }
    }
    if checks::known(weight_outputs).is_some_and(|outputs| outputs % groups != 0) {
        return Err(Error::InvalidShape(
            "the weights' output channels must be divisible by groups",
        ));
    }
    Ok(())
}

fn check_convolution2d(
    info: &Convolution2DDescriptorInfo,
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
        checks::rank_exactly(source, 4, "the convolution source must have rank 4")?,
        checks::rank_exactly(weights, 4, "convolution weights must have rank 4")?,
    ) else {
        return Ok(());
    };
    let shapes = convolution2d_shapes(info, &source_dims, &weight_dims)?;
    check_groups(
        shapes.channels,
        shapes.weight_inputs,
        shapes.weight_outputs,
        info.groups,
    )?;
    if let Some(padded) = pads_window(info.padding_style) {
        let (top, bottom, left, right) = if padded {
            (
                info.padding_top,
                info.padding_bottom,
                info.padding_left,
                info.padding_right,
            )
        } else {
            (0, 0, 0, 0)
        };
        let fits = |extent, kernel: isize, dilation, pads| {
            checks::known(kernel).is_none_or(|kernel| {
                kernel > 0 && checks::window_fits(extent, kernel, dilation, pads)
            })
        };
        if !fits(
            shapes.height,
            shapes.kernel_height,
            info.dilation_rate_in_y,
            (top, bottom),
        ) || !fits(
            shapes.width,
            shapes.kernel_width,
            info.dilation_rate_in_x,
            (left, right),
        ) {
            return Err(Error::InvalidShape(
                "the dilated kernel is larger than the padded source",
            ));
        }
    }
    Ok(())
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

descriptor_handle!(Pooling2DDescriptor);
impl Pooling2DDescriptor {
/// Calls the `MPSGraph` framework counterpart for `new`.
    pub fn new(info: Pooling2DDescriptorInfo) -> Result<Self> {
        if [
            info.kernel_width,
            info.kernel_height,
            info.stride_in_x,
            info.stride_in_y,
            info.dilation_rate_in_x,
            info.dilation_rate_in_y,
        ]
        .contains(&0)
        {
            return Err(Error::InvalidArgument(
                "pooling kernel sizes, strides and dilation rates must be at least 1",
            ));
        }
        if !valid_padding_style(info.padding_style) {
            return Err(Error::InvalidArgument("unknown MPSGraphPaddingStyle"));
        }
        if !matches!(
            info.data_layout,
            tensor_named_data_layout::NHWC | tensor_named_data_layout::NCHW
        ) {
            return Err(Error::InvalidArgument(
                "2D pooling takes NHWC or NCHW data",
            ));
        }
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
        Self::created(ptr)
    }
}

static NEXT_SCOPE: AtomicU64 = AtomicU64::new(1);

struct Scope {
    id: u64,
    created: Vec<usize>,
}

impl Scope {
    fn new() -> Self {
        Self {
            id: NEXT_SCOPE.fetch_add(1, Ordering::Relaxed),
            created: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TypeSignature {
    pub(crate) shape: Option<Dims>,
    pub(crate) data_type: u32,
}

impl TypeSignature {
    pub(crate) fn of(tensor: &Tensor) -> Self {
        Self {
            shape: tensor.shape(),
            data_type: tensor.data_type(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CallSignature {
    pub(crate) symbol: String,
    pub(crate) inputs: Vec<TypeSignature>,
    pub(crate) outputs: Vec<TypeSignature>,
}

struct GraphState {
    scopes: Vec<Scope>,
    deps: HashMap<usize, Vec<usize>>,
    placeholders: Vec<(usize, u64)>,
    calls: Vec<CallSignature>,
}

impl GraphState {
    fn is_open(&self, scope: u64) -> bool {
        self.scopes.iter().any(|open| open.id == scope)
    }

    fn record(&mut self, handle: usize, deps: &[usize]) -> u64 {
        if !deps.is_empty() {
            self.deps.entry(handle).or_default().extend_from_slice(deps);
        }
        self.scopes.last_mut().map_or(0, |scope| {
            scope.created.push(handle);
            scope.id
        })
    }
}

/// Mirrors the `MPSGraph` framework counterpart for this type.
pub struct Graph {
    ptr: *mut c_void,
    state: RefCell<GraphState>,
}

unsafe impl Send for Graph {}

impl Drop for Graph {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `ptr` is a +1 retained Swift/ObjC object pointer owned by this wrapper.
            unsafe { ffi::mpsgraph_object_release(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

/// Mirrors the `MPSGraph` framework counterpart for this type.
pub struct Tensor {
    ptr: *mut c_void,
    scope: u64,
}

unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

impl Drop for Tensor {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `ptr` is a +1 retained Swift/ObjC object pointer owned by this wrapper.
            unsafe { ffi::mpsgraph_object_release(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

impl Tensor {
/// Mirrors the `MPSGraph` framework constant `fn`.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub(crate) const fn from_raw(ptr: *mut c_void, scope: u64) -> Self {
        Self { ptr, scope }
    }

    pub(crate) const fn scope(&self) -> u64 {
        self.scope
    }
}

pub(crate) fn handles(tensors: &[&Tensor]) -> Vec<usize> {
    tensors.iter().map(|tensor| tensor.ptr as usize).collect()
}

pub(crate) fn volume(dimensions: &[usize]) -> Option<usize> {
    dimensions.iter().try_fold(1_usize, |elements, dimension| {
        elements.checked_mul(*dimension)
    })
}

fn static_dimensions(tensor: &Tensor) -> Option<Vec<usize>> {
    tensor
        .shape()?
        .into_iter()
        .map(|dimension| usize::try_from(dimension).ok())
        .collect()
}

fn dimension_matches(expected: isize, actual: usize) -> bool {
    expected < 0 || usize::try_from(expected).ok() == Some(actual)
}

pub(crate) fn axis_in_range(tensor: &Tensor, axis: isize) -> Result<()> {
    checks::axis(tensor, axis).map(drop)
}

pub(crate) fn axes_in_range(tensor: &Tensor, axes: &[usize]) -> Result<()> {
    match tensor.shape() {
        Some(shape) if axes.iter().any(|axis| *axis >= shape.len()) => Err(Error::InvalidShape(
            "every axis must be below the tensor's rank",
        )),
        _ => Ok(()),
    }
}

fn matrix_operands(primary: &Tensor, secondary: &Tensor) -> Result<()> {
    checks::same_data_type(
        primary,
        secondary,
        "matrix multiplication needs operands of one data type",
    )?;
    let data_type = primary.data_type();
    if !checks::is_float(data_type) && !checks::is_complex(data_type) {
        return Err(Error::InvalidDataType(
            "matrix multiplication needs floating-point or complex operands",
        ));
    }
    let (Some(mut left), Some(mut right)) = (primary.shape(), secondary.shape()) else {
        return Ok(());
    };
    if left.is_empty() || right.is_empty() {
        return Err(Error::InvalidShape(
            "matrix multiplication operands need rank 1 or more",
        ));
    }
    for shape in [&mut left, &mut right] {
        if shape.len() == 1 {
            shape.insert(0, 1);
        }
    }
    let (inner, outer) = (left[left.len() - 1], right[right.len() - 2]);
    if !checks::same_extent(inner, outer)
        || !checks::broadcastable(&left[..left.len() - 2], &right[..right.len() - 2])
    {
        return Err(Error::InvalidShape(
            "the matrix operands' inner or batch dimensions do not match",
        ));
    }
    Ok(())
}

fn reshape_is_valid(tensor: &Tensor, shape: &[usize]) -> Result<()> {
    let target = volume(shape).ok_or(Error::Overflow)?;
    checks::shape_to_dims(shape)?;
    if static_dimensions(tensor).is_some_and(|source| volume(&source) != Some(target)) {
        return Err(Error::InvalidShape(
            "a reshape must keep the tensor's element count",
        ));
    }
    Ok(())
}

fn is_permutation(tensor: &Tensor, permutation: &[usize]) -> Result<()> {
    let mut seen = vec![false; permutation.len()];
    let valid = permutation
        .iter()
        .all(|axis| *axis < seen.len() && !core::mem::replace(&mut seen[*axis], true))
        && tensor
            .shape()
            .is_none_or(|shape| shape.len() == permutation.len());
    if valid {
        Ok(())
    } else {
        Err(Error::InvalidShape(
            "the permutation must reorder every axis exactly once",
        ))
    }
}

fn slice_in_range(tensor: &Tensor, dimension: usize, start: isize, length: isize) -> Result<()> {
    let invalid = Err(Error::InvalidShape(
        "the slice lies outside the tensor's dimension",
    ));
    if length < 0 {
        return invalid;
    }
    let Some(shape) = tensor.shape() else {
        return Ok(());
    };
    let Some(&size) = shape.get(dimension) else {
        return invalid;
    };
    if size < 0 {
        return Ok(());
    }
    let start = if start < 0 { start + size } else { start };
    if (0..=size).contains(&start) && start.checked_add(length).is_some_and(|end| end <= size) {
        Ok(())
    } else {
        invalid
    }
}

fn broadcast_is_valid(tensor: &Tensor, shape: &[usize]) -> Result<()> {
    checks::shape_to_dims(shape)?;
    let valid = tensor.shape().is_none_or(|source| {
        source.len() <= shape.len()
            && source
                .iter()
                .rev()
                .zip(shape.iter().rev())
                .all(|(from, to)| *from == 1 || dimension_matches(*from, *to))
    });
    if valid {
        Ok(())
    } else {
        Err(Error::InvalidShape(
            "the tensor cannot be broadcast to that shape",
        ))
    }
}

pub(crate) fn feed_matches(tensor: &Tensor, shape: &[usize], data_type: u32) -> bool {
    tensor.data_type() == data_type
        && tensor.shape().is_none_or(|expected| {
            expected.len() == shape.len()
                && expected
                    .iter()
                    .zip(shape)
                    .all(|(expected, actual)| dimension_matches(*expected, *actual))
        })
}

fn feeds_match(feeds: &[Feed<'_>]) -> bool {
    feeds
        .iter()
        .all(|feed| feed_matches(feed.tensor, &feed.data.shape(), feed.data.data_type()))
}

pub(crate) const fn constant_data_type(data_type: u32) -> bool {
    matches!(
        data_type,
        data_type::FLOAT32
            | data_type::FLOAT16
            | data_type::BFLOAT16
            | data_type::INT8
            | data_type::INT16
            | data_type::INT32
            | data_type::INT64
            | data_type::UINT8
            | data_type::UINT16
            | data_type::UINT32
            | data_type::UINT64
            | data_type::BOOL
            | data_type::COMPLEX_FLOAT16
            | data_type::COMPLEX_FLOAT32
    )
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
            Some(Self {
                ptr,
                state: RefCell::new(GraphState {
                    scopes: vec![Scope::new()],
                    deps: HashMap::new(),
                    placeholders: Vec::new(),
                    calls: Vec::new(),
                }),
            })
        }
    }

/// Mirrors the `MPSGraph` framework constant `fn`.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub(crate) fn check(&self, tensors: &[&Tensor]) -> Result<()> {
        let state = self.state.borrow();
        if tensors.iter().all(|tensor| state.is_open(tensor.scope)) {
            Ok(())
        } else {
            Err(Error::ForeignTensor)
        }
    }

    pub(crate) fn check_operations(&self, operations: &[&Operation]) -> Result<()> {
        let state = self.state.borrow();
        if operations
            .iter()
            .all(|operation| state.is_open(operation.scope()))
        {
            Ok(())
        } else {
            Err(Error::ForeignOperation)
        }
    }

    pub(crate) fn output(
        &self,
        ptr: *mut c_void,
        inputs: &[&Tensor],
        failure: &'static str,
    ) -> Result<Tensor> {
        if ptr.is_null() {
            return Err(Error::OperationFailed(failure));
        }
        let scope = self
            .state
            .borrow_mut()
            .record(ptr as usize, &handles(inputs));
        Ok(Tensor { ptr, scope })
    }

    pub(crate) fn outputs(
        &self,
        box_handle: *mut c_void,
        deps: &[usize],
        count: Option<usize>,
        failure: &'static str,
    ) -> Result<Vec<Tensor>> {
        if box_handle.is_null() {
            return Err(Error::OperationFailed(failure));
        }
        let raw = take_tensor_handles(box_handle);
        if raw.iter().any(|handle| handle.is_null()) || count.is_some_and(|count| count != raw.len())
        {
            release_handles(raw);
            return Err(Error::OperationFailed(failure));
        }
        let mut state = self.state.borrow_mut();
        Ok(raw
            .into_iter()
            .map(|ptr| {
                let scope = state.record(ptr as usize, deps);
                Tensor { ptr, scope }
            })
            .collect())
    }

    pub(crate) fn output_pair(
        &self,
        box_handle: *mut c_void,
        inputs: &[&Tensor],
        failure: &'static str,
    ) -> Result<(Tensor, Tensor)> {
        let mut values = self.outputs(box_handle, &handles(inputs), Some(2), failure)?;
        let second = values.pop().ok_or(Error::OperationFailed(failure))?;
        let first = values.pop().ok_or(Error::OperationFailed(failure))?;
        Ok((first, second))
    }

    pub(crate) fn operation(
        &self,
        ptr: *mut c_void,
        inputs: &[&Tensor],
        failure: &'static str,
    ) -> Result<Operation> {
        if ptr.is_null() {
            return Err(Error::OperationFailed(failure));
        }
        let deps = handles(inputs);
        let scope = self.state.borrow_mut().record(ptr as usize, &deps);
        Ok(Operation::from_raw(ptr, scope, deps))
    }

    pub(crate) fn open_block(&self) -> u64 {
        let scope = Scope::new();
        let id = scope.id;
        self.state.borrow_mut().scopes.push(scope);
        id
    }

    pub(crate) fn close_block(&self, id: u64) -> Vec<usize> {
        let mut state = self.state.borrow_mut();
        if state.scopes.len() > 1 && state.scopes.last().is_some_and(|top| top.id == id) {
            state.scopes.pop().map(|scope| scope.created).unwrap_or_default()
        } else {
            Vec::new()
        }
    }

    pub(crate) fn created_mark(&self) -> usize {
        self.state
            .borrow()
            .scopes
            .last()
            .map_or(0, |scope| scope.created.len())
    }

    pub(crate) fn created_since(&self, mark: usize) -> Vec<usize> {
        self.state
            .borrow()
            .scopes
            .last()
            .map(|scope| scope.created.get(mark..).unwrap_or_default().to_vec())
            .unwrap_or_default()
    }

    pub(crate) fn record_call(&self, call: CallSignature) {
        self.state.borrow_mut().calls.push(call);
    }

    pub(crate) fn calls(&self) -> Vec<CallSignature> {
        self.state.borrow().calls.clone()
    }

    pub(crate) fn placeholder_scope(&self, handle: *mut c_void) -> Option<u64> {
        self.state
            .borrow()
            .placeholders
            .iter()
            .find(|(placeholder, _)| *placeholder == handle as usize)
            .map(|(_, scope)| *scope)
    }

    pub(crate) fn check_execution(&self, feeds: &[&Tensor], targets: &[&Tensor]) -> Result<()> {
        let state = self.state.borrow();
        if state.scopes.len() != 1 {
            return Err(Error::InvalidArgument(
                "a graph cannot run or compile while a control-flow block is being built",
            ));
        }
        let root = state.scopes[0].id;
        if targets.iter().any(|target| target.scope != root) {
            return Err(Error::ForeignTensor);
        }
        let placeholders = state
            .placeholders
            .iter()
            .map(|(handle, _)| *handle)
            .collect::<HashSet<_>>();
        let mut fed = HashSet::with_capacity(feeds.len());
        for feed in feeds {
            let handle = feed.ptr as usize;
            if !placeholders.contains(&handle) {
                return Err(if state.is_open(feed.scope) {
                    Error::InvalidArgument("only this graph's placeholders can be fed")
                } else {
                    Error::ForeignTensor
                });
            }
            fed.insert(handle);
        }
        let mut pending = handles(targets);
        let mut visited = HashSet::new();
        while let Some(handle) = pending.pop() {
            if !visited.insert(handle) {
                continue;
            }
            if placeholders.contains(&handle) && !fed.contains(&handle) {
                return Err(Error::MissingFeed);
            }
            if let Some(deps) = state.deps.get(&handle) {
                pending.extend(deps.iter().filter(|dep| !visited.contains(*dep)));
            }
        }
        Ok(())
    }

    fn check_no_calls(&self) -> Result<()> {
        self.state
            .borrow()
            .calls
            .first()
            .map_or(Ok(()), |call| Err(Error::MissingCallable(call.symbol.clone())))
    }

/// Calls the `MPSGraph` framework counterpart for `placeholder`.
    pub fn placeholder(
        &self,
        shape: Option<&[usize]>,
        data_type: u32,
        name: Option<&str>,
    ) -> Result<Tensor> {
        if let Some(shape) = shape {
            checks::shape_to_dims(shape)?;
        }
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
        let tensor = self.output(ptr, &[], "MPSGraph did not create the placeholder")?;
        self.state
            .borrow_mut()
            .placeholders
            .push((tensor.ptr as usize, tensor.scope));
        Ok(tensor)
    }

/// Calls the `MPSGraph` framework counterpart for `constant_bytes`.
    pub fn constant_bytes(&self, data: &[u8], shape: &[usize], data_type: u32) -> Result<Tensor> {
        let expected = checked_byte_len(shape, data_type).ok_or_else(|| {
            if data_type_bits(data_type).is_some() {
                Error::Overflow
            } else {
                Error::UnsupportedDataType(data_type)
            }
        })?;
        if data.len() != expected {
            return Err(Error::InvalidLength {
                expected,
                actual: data.len(),
            });
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
        self.output(ptr, &[], "MPSGraph did not create the constant")
    }

/// Calls the `MPSGraph` framework counterpart for `constant_f32_slice`.
    pub fn constant_f32_slice(&self, values: &[f32], shape: &[usize]) -> Result<Tensor> {
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
    pub fn constant_scalar(&self, scalar: f64, data_type: u32) -> Result<Tensor> {
        if !constant_data_type(data_type) {
            return Err(Error::UnsupportedDataType(data_type));
        }
        // SAFETY: Pure constructor over scalar inputs.
        let ptr = unsafe { ffi::mpsgraph_graph_constant_scalar(self.ptr, scalar, data_type) };
        self.output(ptr, &[], "MPSGraph did not create the constant")
    }

/// Calls the `MPSGraph` framework counterpart for `constant_scalar_shaped`.
    pub fn constant_scalar_shaped(
        &self,
        scalar: f64,
        shape: &[usize],
        data_type: u32,
    ) -> Result<Tensor> {
        if !constant_data_type(data_type) {
            return Err(Error::UnsupportedDataType(data_type));
        }
        if shape.contains(&0) {
            return Err(Error::InvalidShape(
                "splatted constants need every dimension to be at least 1",
            ));
        }
        checks::shape_to_dims(shape)?;
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
        self.output(ptr, &[], "MPSGraph did not create the constant")
    }

    impl_binary_tensor_op!(addition, mpsgraph_graph_addition, checks::elementwise);
    impl_binary_tensor_op!(subtraction, mpsgraph_graph_subtraction, checks::elementwise);
    impl_binary_tensor_op!(
        multiplication,
        mpsgraph_graph_multiplication,
        checks::elementwise
    );
    impl_binary_tensor_op!(division, mpsgraph_graph_division, checks::elementwise);
    impl_binary_tensor_op!(
        matrix_multiplication,
        mpsgraph_graph_matrix_multiplication,
        matrix_operands
    );
    impl_unary_tensor_op!(relu, mpsgraph_graph_relu);
    impl_unary_tensor_op!(sigmoid, mpsgraph_graph_sigmoid);
    impl_axes_tensor_op!(reduction_sum, mpsgraph_graph_reduction_sum);
    impl_axes_tensor_op!(reduction_maximum, mpsgraph_graph_reduction_maximum);
    impl_axes_tensor_op!(reduction_minimum, mpsgraph_graph_reduction_minimum);
    impl_axes_tensor_op!(mean, mpsgraph_graph_mean);

/// Calls the `MPSGraph` framework counterpart for `softmax`.
    pub fn softmax(&self, tensor: &Tensor, axis: isize, name: Option<&str>) -> Result<Tensor> {
        self.check(&[tensor])?;
        axis_in_range(tensor, axis)?;
        checks::float_tensor(tensor, "softmax needs a floating-point tensor")?;
        let name = optional_cstring(name);
        // SAFETY: All pointers originate from safe wrappers and remain alive for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_softmax(self.ptr, tensor.as_ptr(), axis, cstring_ptr(&name))
        };
        self.output(ptr, &[tensor], "MPSGraph did not create softmax")
    }

/// Calls the `MPSGraph` framework counterpart for `reshape`.
    pub fn reshape(&self, tensor: &Tensor, shape: &[usize], name: Option<&str>) -> Result<Tensor> {
        self.check(&[tensor])?;
        reshape_is_valid(tensor, shape)?;
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
        self.output(ptr, &[tensor], "MPSGraph did not create the reshape")
    }

/// Calls the `MPSGraph` framework counterpart for `transpose`.
    pub fn transpose(
        &self,
        tensor: &Tensor,
        permutation: &[usize],
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[tensor])?;
        is_permutation(tensor, permutation)?;
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
        self.output(ptr, &[tensor], "MPSGraph did not create the transpose")
    }

/// Calls the `MPSGraph` framework counterpart for `slice`.
    pub fn slice(
        &self,
        tensor: &Tensor,
        dimension: usize,
        start: isize,
        length: isize,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[tensor])?;
        slice_in_range(tensor, dimension, start, length)?;
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
        self.output(ptr, &[tensor], "MPSGraph did not create the slice")
    }

/// Calls the `MPSGraph` framework counterpart for `broadcast`.
    pub fn broadcast(
        &self,
        tensor: &Tensor,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[tensor])?;
        broadcast_is_valid(tensor, shape)?;
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
        self.output(ptr, &[tensor], "MPSGraph did not create the broadcast")
    }

/// Calls the `MPSGraph` framework counterpart for `convolution2d`.
    pub fn convolution2d(
        &self,
        source: &Tensor,
        weights: &Tensor,
        descriptor: &Convolution2DDescriptor,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[source, weights])?;
        check_convolution2d(descriptor.info(), source, weights)?;
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
        self.output(ptr, &[source, weights], "MPSGraph did not create the convolution")
    }

/// Calls the `MPSGraph` framework counterpart for `max_pooling2d`.
    pub fn max_pooling2d(
        &self,
        source: &Tensor,
        descriptor: &Pooling2DDescriptor,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[source])?;
        checks::rank_exactly(source, 4, "2D pooling needs a rank-4 source")?;
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
        self.output(ptr, &[source], "MPSGraph did not create the pooling")
    }

/// Calls the `MPSGraph` framework counterpart for `normalize`.
    #[allow(clippy::too_many_arguments)]
    pub fn normalize(
        &self,
        tensor: &Tensor,
        mean: &Tensor,
        variance: &Tensor,
        gamma: Option<&Tensor>,
        beta: Option<&Tensor>,
        epsilon: f32,
        name: Option<&str>,
    ) -> Result<Tensor> {
        let inputs = [Some(tensor), Some(mean), Some(variance), gamma, beta]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        self.check(&inputs)?;
        checks::float_tensor(tensor, "normalization needs a floating-point tensor")?;
        for statistic in &inputs[1..] {
            checks::same_data_type(
                tensor,
                statistic,
                "normalization statistics need the tensor's data type",
            )?;
            checks::broadcast_with(
                tensor,
                statistic,
                "normalization statistics must broadcast to the tensor",
            )?;
        }
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
        self.output(ptr, &inputs, "MPSGraph did not create the normalization")
    }

/// Calls the `MPSGraph` framework counterpart for `run`.
    pub fn run(&self, feeds: &[Feed<'_>], targets: &[&Tensor]) -> Result<Vec<TensorData>> {
        self.check_run(feeds, targets)?;
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

    fn check_run(&self, feeds: &[Feed<'_>], targets: &[&Tensor]) -> Result<()> {
        let feed_tensors = feeds.iter().map(|feed| feed.tensor).collect::<Vec<_>>();
        self.check_execution(&feed_tensors, targets)?;
        self.check_no_calls()?;
        if feeds_match(feeds) {
            Ok(())
        } else {
            Err(Error::InvalidShape(FEED_MISMATCH))
        }
    }

/// Calls the `MPSGraph` framework counterpart for `run_with_command_queue`.
    pub fn run_with_command_queue(
        &self,
        command_queue: &CommandQueue,
        feeds: &[Feed<'_>],
        targets: &[&Tensor],
    ) -> Result<Vec<TensorData>> {
        self.check_run(feeds, targets)?;
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
    pub fn compile(
        &self,
        device: &MetalDevice,
        feeds: &[FeedDescription<'_>],
        targets: &[&Tensor],
    ) -> Result<Executable> {
        self.check_compile(feeds, targets)?;
        self.check_no_calls()?;
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
            Err(Error::OperationFailed("MPSGraph did not compile the graph"))
        } else {
            Ok(Executable::from_raw(ptr, targets.len()).with_signature(feeds, targets))
        }
    }

    pub(crate) fn check_compile(
        &self,
        feeds: &[FeedDescription<'_>],
        targets: &[&Tensor],
    ) -> Result<()> {
        let feed_tensors = feeds.iter().map(|feed| feed.tensor).collect::<Vec<_>>();
        self.check_execution(&feed_tensors, targets)?;
        for feed in feeds {
            checks::shape_to_dims(feed.shape)?;
        }
        if feed_descriptions_match(feeds) {
            Ok(())
        } else {
            Err(Error::InvalidShape(FEED_MISMATCH))
        }
    }
}

const FEED_MISMATCH: &str = "a feed's shape or data type does not match its tensor";

pub(crate) fn feed_descriptions_match(feeds: &[FeedDescription<'_>]) -> bool {
    feeds
        .iter()
        .all(|feed| feed_matches(feed.tensor, feed.shape, feed.data_type))
}

pub(crate) fn take_tensor_handles(handle: *mut c_void) -> Vec<*mut c_void> {
    if handle.is_null() {
        return Vec::new();
    }
    // SAFETY: `handle` is a retained tensor-array box created by the Swift bridge.
    let len = unsafe { ffi::mpsgraph_tensor_array_box_len(handle) };
    let values = (0..len)
        // SAFETY: indices are bounded by the just-read length.
        .map(|index| unsafe { ffi::mpsgraph_tensor_array_box_get(handle, index) })
        .collect();
    unsafe { ffi::mpsgraph_object_release(handle) };
    values
}

pub(crate) fn release_handles(handles: Vec<*mut c_void>) {
    for handle in handles {
        if !handle.is_null() {
            unsafe { ffi::mpsgraph_object_release(handle) };
        }
    }
}

/// Safe owner for a compiled `MPSGraphExecutable`.
pub struct Executable {
    ptr: *mut c_void,
    output_count: usize,
    feed_types: Vec<(usize, Vec<usize>, u32)>,
    output_types: Vec<TypeSignature>,
    _not_sync: PhantomData<core::cell::Cell<()>>,
}

unsafe impl Send for Executable {}

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
        Self {
            ptr,
            output_count,
            feed_types: Vec::new(),
            output_types: Vec::new(),
            _not_sync: PhantomData,
        }
    }

    pub(crate) fn with_signature(mut self, feeds: &[FeedDescription<'_>], targets: &[&Tensor]) -> Self {
        self.feed_types = feeds
            .iter()
            .map(|feed| {
                (
                    feed.tensor.as_ptr() as usize,
                    feed.shape.to_vec(),
                    feed.data_type,
                )
            })
            .collect();
        self.output_types = targets.iter().map(|target| TypeSignature::of(target)).collect();
        self
    }

    pub(crate) fn callable_signature(&self) -> Option<(Vec<TypeSignature>, Vec<TypeSignature>)> {
        if self.output_types.is_empty() {
            return None;
        }
        Some((self.input_signature().ok()?, self.output_types.clone()))
    }

    pub(crate) fn input_signature(&self) -> Result<Vec<TypeSignature>> {
        self.feed_tensors()
            .iter()
            .map(|tensor| {
                let (_, shape, data_type) = self
                    .feed_types
                    .iter()
                    .find(|(identity, ..)| *identity == tensor.as_ptr() as usize)
                    .ok_or(Error::Unsupported(
                        "the compiled input types of this executable are unknown",
                    ))?;
                Ok(TypeSignature {
                    shape: Some(checks::shape_to_dims(shape)?),
                    data_type: *data_type,
                })
            })
            .collect()
    }

    pub(crate) fn check_inputs(&self, inputs: &[&TensorData]) -> Result<()> {
        let feeds = self.feed_tensors();
        if feeds.is_empty() {
            return Ok(());
        }
        if inputs.len() != feeds.len() {
            return Err(Error::InvalidShape(
                "the executable needs one input per feed tensor",
            ));
        }
        for (tensor, input) in feeds.iter().zip(inputs) {
            let (shape, data_type) = (input.shape(), input.data_type());
            let compiled = self
                .feed_types
                .iter()
                .find(|(identity, ..)| *identity == tensor.as_ptr() as usize)
                .is_none_or(|(_, expected, expected_type)| {
                    *expected == shape && *expected_type == data_type
                });
            if !compiled || !feed_matches(tensor, &shape, data_type) {
                return Err(Error::InvalidShape(FEED_MISMATCH));
            }
        }
        Ok(())
    }

    pub(crate) fn check_results(&self, results: &[&TensorData]) -> Result<()> {
        if results.len() != self.output_count {
            return Err(Error::InvalidShape(
                "preallocated results need one tensor per executable output",
            ));
        }
        let targets = self.target_tensors();
        if targets.len() != results.len() {
            return Err(Error::InvalidShape(
                "preallocated results need an executable compiled from a graph",
            ));
        }
        for (target, result) in targets.iter().zip(results) {
            let fits = static_dimensions(target).is_some_and(|shape| shape == result.shape())
                && target.data_type() == result.data_type();
            if !fits {
                return Err(Error::InvalidShape(
                    "preallocated results must match statically shaped outputs",
                ));
            }
        }
        Ok(())
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
        self.check_inputs(inputs)?;
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
