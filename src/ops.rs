use crate::checks;
use crate::error::{Error, Result};
use crate::ffi;
use crate::graph::{axes_in_range, axis_in_range, handles, padding_mode, Tensor};
use core::ffi::c_char;
use std::ffi::CString;

fn axis_length(tensor: &Tensor, axis: isize) -> Option<usize> {
    let shape = tensor.shape()?;
    let index = checks::normalized_axis(axis, shape.len())?;
    checks::known(*shape.get(index)?)
}

fn optional_cstring(name: Option<&str>) -> Option<CString> {
    name.and_then(|value| CString::new(value).ok())
}

#[allow(clippy::ref_option)]
fn cstring_ptr(value: &Option<CString>) -> *const c_char {
    value
        .as_ref()
        .map_or(core::ptr::null(), |value| value.as_ptr())
}

/// Mirrors the `MPSGraph` framework counterpart for `UnaryArithmeticOp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum UnaryArithmeticOp {
/// Mirrors the `MPSGraph` framework case `Identity`.
    Identity = 0,
/// Mirrors the `MPSGraph` framework case `Exponent`.
    Exponent = 1,
/// Mirrors the `MPSGraph` framework case `ExponentBase2`.
    ExponentBase2 = 2,
/// Mirrors the `MPSGraph` framework case `ExponentBase10`.
    ExponentBase10 = 3,
/// Mirrors the `MPSGraph` framework case `Logarithm`.
    Logarithm = 4,
/// Mirrors the `MPSGraph` framework case `LogarithmBase2`.
    LogarithmBase2 = 5,
/// Mirrors the `MPSGraph` framework case `LogarithmBase10`.
    LogarithmBase10 = 6,
/// Mirrors the `MPSGraph` framework case `Square`.
    Square = 7,
/// Mirrors the `MPSGraph` framework case `SquareRoot`.
    SquareRoot = 8,
/// Mirrors the `MPSGraph` framework case `Reciprocal`.
    Reciprocal = 9,
/// Mirrors the `MPSGraph` framework case `Absolute`.
    Absolute = 10,
/// Mirrors the `MPSGraph` framework case `Negative`.
    Negative = 11,
/// Mirrors the `MPSGraph` framework case `Sign`.
    Sign = 12,
/// Mirrors the `MPSGraph` framework case `SignBit`.
    SignBit = 13,
/// Mirrors the `MPSGraph` framework case `Ceil`.
    Ceil = 14,
/// Mirrors the `MPSGraph` framework case `Floor`.
    Floor = 15,
/// Mirrors the `MPSGraph` framework case `Round`.
    Round = 16,
/// Mirrors the `MPSGraph` framework case `Rint`.
    Rint = 17,
/// Mirrors the `MPSGraph` framework case `Sin`.
    Sin = 18,
/// Mirrors the `MPSGraph` framework case `Cos`.
    Cos = 19,
/// Mirrors the `MPSGraph` framework case `Tan`.
    Tan = 20,
/// Mirrors the `MPSGraph` framework case `Sinh`.
    Sinh = 21,
/// Mirrors the `MPSGraph` framework case `Cosh`.
    Cosh = 22,
/// Mirrors the `MPSGraph` framework case `Tanh`.
    Tanh = 23,
/// Mirrors the `MPSGraph` framework case `Asin`.
    Asin = 24,
/// Mirrors the `MPSGraph` framework case `Acos`.
    Acos = 25,
/// Mirrors the `MPSGraph` framework case `Atan`.
    Atan = 26,
/// Mirrors the `MPSGraph` framework case `Asinh`.
    Asinh = 27,
/// Mirrors the `MPSGraph` framework case `Acosh`.
    Acosh = 28,
/// Mirrors the `MPSGraph` framework case `Atanh`.
    Atanh = 29,
/// Mirrors the `MPSGraph` framework case `IsNaN`.
    IsNaN = 30,
/// Mirrors the `MPSGraph` framework case `IsInfinite`.
    IsInfinite = 31,
}

/// Mirrors the `MPSGraph` framework counterpart for `BinaryArithmeticOp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum BinaryArithmeticOp {
/// Mirrors the `MPSGraph` framework case `Addition`.
    Addition = 0,
/// Mirrors the `MPSGraph` framework case `Subtraction`.
    Subtraction = 1,
/// Mirrors the `MPSGraph` framework case `Multiplication`.
    Multiplication = 2,
/// Mirrors the `MPSGraph` framework case `Division`.
    Division = 3,
/// Mirrors the `MPSGraph` framework case `DivisionNoNaN`.
    DivisionNoNaN = 4,
/// Mirrors the `MPSGraph` framework case `Power`.
    Power = 5,
/// Mirrors the `MPSGraph` framework case `Minimum`.
    Minimum = 6,
/// Mirrors the `MPSGraph` framework case `Maximum`.
    Maximum = 7,
/// Mirrors the `MPSGraph` framework case `Equal`.
    Equal = 8,
/// Mirrors the `MPSGraph` framework case `NotEqual`.
    NotEqual = 9,
/// Mirrors the `MPSGraph` framework case `GreaterThan`.
    GreaterThan = 10,
/// Mirrors the `MPSGraph` framework case `GreaterThanOrEqualTo`.
    GreaterThanOrEqualTo = 11,
/// Mirrors the `MPSGraph` framework case `LessThan`.
    LessThan = 12,
/// Mirrors the `MPSGraph` framework case `LessThanOrEqualTo`.
    LessThanOrEqualTo = 13,
/// Mirrors the `MPSGraph` framework case `LogicalAnd`.
    LogicalAnd = 14,
/// Mirrors the `MPSGraph` framework case `LogicalOr`.
    LogicalOr = 15,
/// Mirrors the `MPSGraph` framework case `Atan2`.
    Atan2 = 16,
/// Mirrors the `MPSGraph` framework case `FloorModulo`.
    FloorModulo = 17,
}

/// Mirrors the `MPSGraph` framework counterpart for `ReductionAxisOp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ReductionAxisOp {
/// Mirrors the `MPSGraph` framework case `Sum`.
    Sum = 0,
/// Mirrors the `MPSGraph` framework case `Maximum`.
    Maximum = 1,
/// Mirrors the `MPSGraph` framework case `Minimum`.
    Minimum = 2,
/// Mirrors the `MPSGraph` framework case `Product`.
    Product = 3,
}

/// Mirrors the `MPSGraph` framework counterpart for `ReductionAxesOp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ReductionAxesOp {
/// Mirrors the `MPSGraph` framework case `Sum`.
    Sum = 0,
/// Mirrors the `MPSGraph` framework case `Maximum`.
    Maximum = 1,
/// Mirrors the `MPSGraph` framework case `Minimum`.
    Minimum = 2,
/// Mirrors the `MPSGraph` framework case `Product`.
    Product = 3,
}


impl crate::graph::Graph {
/// Calls the `MPSGraph` framework counterpart for `unary_arithmetic`.
    pub fn unary_arithmetic(
        &self,
        op: UnaryArithmeticOp,
        tensor: &Tensor,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[tensor])?;
        if matches!(op, UnaryArithmeticOp::IsNaN | UnaryArithmeticOp::IsInfinite) {
            checks::float_tensor(
                tensor,
                "isNaN and isInfinite need a floating-point tensor",
            )?;
        }
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_arithmetic_unary(
                self.as_ptr(),
                op as u32,
                tensor.as_ptr(),
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[tensor], "MPSGraph did not create the unary operation")
    }

/// Calls the `MPSGraph` framework counterpart for `binary_arithmetic`.
    pub fn binary_arithmetic(
        &self,
        op: BinaryArithmeticOp,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[primary, secondary])?;
        checks::elementwise(primary, secondary)?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_arithmetic_binary(
                self.as_ptr(),
                op as u32,
                primary.as_ptr(),
                secondary.as_ptr(),
                cstring_ptr(&name),
            )
        };
        self.output(
            ptr,
            &[primary, secondary],
            "MPSGraph did not create the binary operation",
        )
    }

/// Calls the `MPSGraph` framework counterpart for `select`.
    pub fn select(
        &self,
        predicate: &Tensor,
        true_tensor: &Tensor,
        false_tensor: &Tensor,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[predicate, true_tensor, false_tensor])?;
        checks::elementwise(true_tensor, false_tensor)?;
        for value in [true_tensor, false_tensor] {
            checks::broadcast_with(
                predicate,
                value,
                "the predicate must broadcast with both values",
            )?;
        }
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_select(
                self.as_ptr(),
                predicate.as_ptr(),
                true_tensor.as_ptr(),
                false_tensor.as_ptr(),
                cstring_ptr(&name),
            )
        };
        self.output(
            ptr,
            &[predicate, true_tensor, false_tensor],
            "MPSGraph did not create the select",
        )
    }

/// Calls the `MPSGraph` framework counterpart for `relu_gradient`.
    pub fn relu_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[gradient, source])?;
        checks::elementwise(gradient, source)?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_relu_gradient(
                self.as_ptr(),
                gradient.as_ptr(),
                source.as_ptr(),
                cstring_ptr(&name),
            )
        };
        self.output(
            ptr,
            &[gradient, source],
            "MPSGraph did not create the ReLU gradient",
        )
    }

/// Calls the `MPSGraph` framework counterpart for `sigmoid_gradient`.
    pub fn sigmoid_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[gradient, source])?;
        checks::elementwise(gradient, source)?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_sigmoid_gradient(
                self.as_ptr(),
                gradient.as_ptr(),
                source.as_ptr(),
                cstring_ptr(&name),
            )
        };
        self.output(
            ptr,
            &[gradient, source],
            "MPSGraph did not create the sigmoid gradient",
        )
    }

/// Calls the `MPSGraph` framework counterpart for `softmax_gradient`.
    pub fn softmax_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        axis: isize,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[gradient, source])?;
        axis_in_range(source, axis)?;
        checks::float_tensor(source, "the softmax gradient needs a floating-point source")?;
        checks::elementwise(gradient, source)?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_softmax_gradient(
                self.as_ptr(),
                gradient.as_ptr(),
                source.as_ptr(),
                axis,
                cstring_ptr(&name),
            )
        };
        self.output(
            ptr,
            &[gradient, source],
            "MPSGraph did not create the softmax gradient",
        )
    }

/// Calls the `MPSGraph` framework counterpart for `leaky_relu`.
    pub fn leaky_relu(&self, tensor: &Tensor, alpha: f64, name: Option<&str>) -> Result<Tensor> {
        self.check(&[tensor])?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_leaky_relu_scalar(
                self.as_ptr(),
                tensor.as_ptr(),
                alpha,
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[tensor], "MPSGraph did not create the leaky ReLU")
    }

/// Calls the `MPSGraph` framework counterpart for `leaky_relu_tensor`.
    pub fn leaky_relu_tensor(
        &self,
        tensor: &Tensor,
        alpha_tensor: &Tensor,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[tensor, alpha_tensor])?;
        checks::elementwise(tensor, alpha_tensor)?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_leaky_relu_tensor(
                self.as_ptr(),
                tensor.as_ptr(),
                alpha_tensor.as_ptr(),
                cstring_ptr(&name),
            )
        };
        self.output(
            ptr,
            &[tensor, alpha_tensor],
            "MPSGraph did not create the leaky ReLU",
        )
    }

/// Calls the `MPSGraph` framework counterpart for `leaky_relu_gradient`.
    pub fn leaky_relu_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        alpha_tensor: &Tensor,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[gradient, source, alpha_tensor])?;
        checks::elementwise(gradient, source)?;
        checks::elementwise(source, alpha_tensor)?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_leaky_relu_gradient(
                self.as_ptr(),
                gradient.as_ptr(),
                source.as_ptr(),
                alpha_tensor.as_ptr(),
                cstring_ptr(&name),
            )
        };
        self.output(
            ptr,
            &[gradient, source, alpha_tensor],
            "MPSGraph did not create the leaky ReLU gradient",
        )
    }

/// Calls the `MPSGraph` framework counterpart for `reduce_axis`.
    pub fn reduce_axis(
        &self,
        op: ReductionAxisOp,
        tensor: &Tensor,
        axis: isize,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[tensor])?;
        axis_in_range(tensor, axis)?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_reduction_axis(
                self.as_ptr(),
                op as u32,
                tensor.as_ptr(),
                axis,
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[tensor], "MPSGraph did not create the reduction")
    }

/// Calls the `MPSGraph` framework counterpart for `reduce_axes`.
    pub fn reduce_axes(
        &self,
        op: ReductionAxesOp,
        tensor: &Tensor,
        axes: &[usize],
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[tensor])?;
        axes_in_range(tensor, axes)?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_reduction_axes(
                self.as_ptr(),
                op as u32,
                tensor.as_ptr(),
                axes.as_ptr(),
                axes.len(),
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[tensor], "MPSGraph did not create the reduction")
    }

/// Calls the `MPSGraph` framework counterpart for `concat_pair`.
    pub fn concat_pair(
        &self,
        first: &Tensor,
        second: &Tensor,
        dimension: isize,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[first, second])?;
        check_concat(&[first, second], dimension, false)?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_concat_pair(
                self.as_ptr(),
                first.as_ptr(),
                second.as_ptr(),
                dimension,
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[first, second], "MPSGraph did not create the concat")
    }

/// Calls the `MPSGraph` framework counterpart for `concat_tensors`.
    pub fn concat_tensors(
        &self,
        tensors: &[&Tensor],
        dimension: isize,
        interleave: bool,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(tensors)?;
        check_concat(tensors, dimension, interleave)?;
        let name = optional_cstring(name);
        let handles = tensors
            .iter()
            .map(|tensor| tensor.as_ptr())
            .collect::<Vec<_>>();
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_concat_tensors(
                self.as_ptr(),
                handles.as_ptr(),
                handles.len(),
                dimension,
                interleave,
                cstring_ptr(&name),
            )
        };
        self.output(ptr, tensors, "MPSGraph did not create the concat")
    }

/// Calls the `MPSGraph` framework counterpart for `split_sizes`.
    pub fn split_sizes(
        &self,
        tensor: &Tensor,
        split_sizes: &[usize],
        axis: isize,
        name: Option<&str>,
    ) -> Result<Vec<Tensor>> {
        self.check(&[tensor])?;
        axis_in_range(tensor, axis)?;
        let total = split_sizes
            .iter()
            .try_fold(0_usize, |total, size| total.checked_add(*size));
        if split_sizes.is_empty()
            || split_sizes.contains(&0)
            || axis_length(tensor, axis).is_some_and(|length| total != Some(length))
        {
            return Err(Error::InvalidShape(
                "split sizes must be positive and add up to the axis length",
            ));
        }
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_graph_split_sizes(
                self.as_ptr(),
                tensor.as_ptr(),
                split_sizes.as_ptr(),
                split_sizes.len(),
                axis,
                cstring_ptr(&name),
            )
        };
        self.outputs(
            box_handle,
            &handles(&[tensor]),
            Some(split_sizes.len()),
            "MPSGraph did not create the split",
        )
    }

/// Calls the `MPSGraph` framework counterpart for `split_sizes_tensor`.
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn split_sizes_tensor(
        &self,
        tensor: &Tensor,
        split_sizes_tensor: &Tensor,
        axis: isize,
        name: Option<&str>,
    ) -> Result<Vec<Tensor>> {
        self.check(&[tensor, split_sizes_tensor])?;
        axis_in_range(tensor, axis)?;
        checks::index_vector(
            split_sizes_tensor,
            None,
            "split sizes must be a rank-1 int32 or int64 tensor",
        )?;
        let count = split_sizes_tensor
            .shape()
            .and_then(|shape| checks::known(shape[0]))
            .filter(|count| *count > 0)
            .ok_or(Error::InvalidShape(
                "the split sizes tensor needs a static, non-zero length",
            ))?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_graph_split_sizes_tensor(
                self.as_ptr(),
                tensor.as_ptr(),
                split_sizes_tensor.as_ptr(),
                axis,
                cstring_ptr(&name),
            )
        };
        self.outputs(
            box_handle,
            &handles(&[tensor, split_sizes_tensor]),
            Some(count),
            "MPSGraph did not create the split",
        )
    }

/// Calls the `MPSGraph` framework counterpart for `split_num`.
    pub fn split_num(
        &self,
        tensor: &Tensor,
        num_splits: usize,
        axis: isize,
        name: Option<&str>,
    ) -> Result<Vec<Tensor>> {
        self.check(&[tensor])?;
        axis_in_range(tensor, axis)?;
        if num_splits == 0 || axis_length(tensor, axis).is_some_and(|length| num_splits > length) {
            return Err(Error::InvalidShape(
                "the split count must be between 1 and the axis length",
            ));
        }
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_graph_split_num(
                self.as_ptr(),
                tensor.as_ptr(),
                num_splits,
                axis,
                cstring_ptr(&name),
            )
        };
        self.outputs(
            box_handle,
            &handles(&[tensor]),
            Some(num_splits),
            "MPSGraph did not create the split",
        )
    }

/// Calls the `MPSGraph` framework counterpart for `stack`.
    pub fn stack(&self, tensors: &[&Tensor], axis: isize, name: Option<&str>) -> Result<Tensor> {
        self.check(tensors)?;
        check_stack(tensors, axis)?;
        let name = optional_cstring(name);
        let handles = tensors
            .iter()
            .map(|tensor| tensor.as_ptr())
            .collect::<Vec<_>>();
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_stack(
                self.as_ptr(),
                handles.as_ptr(),
                handles.len(),
                axis,
                cstring_ptr(&name),
            )
        };
        self.output(ptr, tensors, "MPSGraph did not create the stack")
    }

/// Calls the `MPSGraph` framework counterpart for `pad`.
    pub fn pad(
        &self,
        tensor: &Tensor,
        padding_mode: isize,
        left_padding: &[isize],
        right_padding: &[isize],
        constant_value: f64,
        name: Option<&str>,
    ) -> Result<Tensor> {
        self.check(&[tensor])?;
        check_pad(tensor, padding_mode, left_padding, right_padding)?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let ptr = unsafe {
            ffi::mpsgraph_graph_pad(
                self.as_ptr(),
                tensor.as_ptr(),
                padding_mode,
                left_padding.as_ptr(),
                left_padding.len(),
                right_padding.as_ptr(),
                right_padding.len(),
                constant_value,
                cstring_ptr(&name),
            )
        };
        self.output(ptr, &[tensor], "MPSGraph did not create the pad")
    }

/// Calls the `MPSGraph` framework counterpart for `top_k`.
    pub fn top_k(&self, source: &Tensor, k: usize, name: Option<&str>) -> Result<(Tensor, Tensor)> {
        self.check(&[source])?;
        check_top_k(source, Some(k))?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_graph_top_k(self.as_ptr(), source.as_ptr(), k, cstring_ptr(&name))
        };
        self.output_pair(box_handle, &[source], "MPSGraph did not create top-k")
    }

/// Calls the `MPSGraph` framework counterpart for `top_k_tensor`.
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn top_k_tensor(
        &self,
        source: &Tensor,
        k_tensor: &Tensor,
        name: Option<&str>,
    ) -> Result<(Tensor, Tensor)> {
        self.check(&[source, k_tensor])?;
        check_top_k(source, None)?;
        checks::index_scalar(k_tensor, "k must be a rank-0 int32 or int64 tensor")?;
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_graph_top_k_tensor(
                self.as_ptr(),
                source.as_ptr(),
                k_tensor.as_ptr(),
                cstring_ptr(&name),
            )
        };
        self.output_pair(
            box_handle,
            &[source, k_tensor],
            "MPSGraph did not create top-k",
        )
    }
}

fn check_concat(tensors: &[&Tensor], dimension: isize, interleave: bool) -> Result<()> {
    let Some(first) = tensors.first() else {
        return Err(Error::InvalidArgument("concat needs at least one tensor"));
    };
    if tensors
        .iter()
        .any(|tensor| tensor.data_type() != first.data_type())
    {
        return Err(Error::InvalidDataType(
            "concatenated tensors need one data type",
        ));
    }
    let shapes = tensors.iter().filter_map(|tensor| tensor.shape()).collect::<Vec<_>>();
    let Some(reference) = shapes.first() else {
        return Ok(());
    };
    let axis = checks::normalized_axis(dimension, reference.len())
        .ok_or(Error::InvalidShape("the concat axis is outside -rank..rank"))?;
    for shape in &shapes {
        if shape.len() != reference.len() {
            return Err(Error::InvalidShape("concatenated tensors need one rank"));
        }
        let mismatched = shape
            .iter()
            .zip(reference)
            .enumerate()
            .any(|(index, (extent, expected))| {
                (index != axis || interleave) && !checks::same_extent(*extent, *expected)
            });
        if mismatched {
            return Err(Error::InvalidShape(if interleave {
                "interleaved tensors need identical shapes"
            } else {
                "concatenated tensors must match except along the axis"
            }));
        }
    }
    Ok(())
}

fn check_stack(tensors: &[&Tensor], axis: isize) -> Result<()> {
    let Some(first) = tensors.first() else {
        return Err(Error::InvalidArgument("stack needs at least one tensor"));
    };
    if tensors
        .iter()
        .any(|tensor| tensor.data_type() != first.data_type())
    {
        return Err(Error::InvalidDataType("stacked tensors need one data type"));
    }
    let shapes = tensors.iter().filter_map(|tensor| tensor.shape()).collect::<Vec<_>>();
    let Some(reference) = shapes.first() else {
        return Ok(());
    };
    if shapes
        .iter()
        .any(|shape| !checks::same_dims(shape, reference))
    {
        return Err(Error::InvalidShape("stacked tensors need identical shapes"));
    }
    checks::normalized_axis(axis, reference.len() + 1)
        .map(drop)
        .ok_or(Error::InvalidShape("the stack axis is outside -(rank+1)..=rank"))
}

fn check_pad(tensor: &Tensor, mode: isize, left: &[isize], right: &[isize]) -> Result<()> {
    if !matches!(
        mode,
        padding_mode::CONSTANT
            | padding_mode::REFLECT
            | padding_mode::SYMMETRIC
            | padding_mode::CLAMP_TO_EDGE
            | padding_mode::ZERO
    ) {
        return Err(Error::InvalidArgument(
            "padding_mode must be CONSTANT, REFLECT, SYMMETRIC, CLAMP_TO_EDGE or ZERO; MPSGraph does not run PERIODIC or ANTI_PERIODIC",
        ));
    }
    if left.len() != right.len() {
        return Err(Error::InvalidShape(
            "left and right padding need the same length",
        ));
    }
    let Some(dims) = tensor.shape() else {
        return Ok(());
    };
    if dims.is_empty() || dims.len() != left.len() {
        return Err(Error::InvalidShape(
            "padding needs one value per dimension of a tensor of rank 1 or more",
        ));
    }
    for ((extent, before), after) in dims.iter().zip(left).zip(right) {
        let extent = *extent;
        if extent < 0 {
            continue;
        }
        let limit = match mode {
            padding_mode::REFLECT => Some(extent - 1),
            padding_mode::SYMMETRIC => Some(extent),
            _ => None,
        };
        if limit.is_some_and(|limit| *before > limit || *after > limit) {
            return Err(Error::InvalidShape(
                "reflect padding must be below the dimension and symmetric padding at most the dimension",
            ));
        }
        let padded = extent
            .checked_add(*before)
            .and_then(|padded| padded.checked_add(*after));
        if padded.is_none_or(|padded| padded < 0) {
            return Err(Error::InvalidShape(
                "negative padding removes more than the dimension holds",
            ));
        }
    }
    Ok(())
}

fn check_top_k(source: &Tensor, k: Option<usize>) -> Result<()> {
    if k == Some(0) {
        return Err(Error::InvalidArgument("k must be at least 1"));
    }
    let Some(dims) = source.shape() else {
        return Ok(());
    };
    let Some(&last) = dims.last() else {
        return Err(Error::InvalidShape("top-k needs a source of rank 1 or more"));
    };
    match (k, checks::known(last)) {
        (Some(k), Some(extent)) if k > extent => Err(Error::InvalidShape(
            "k is larger than the source's last dimension",
        )),
        _ => Ok(()),
    }
}
