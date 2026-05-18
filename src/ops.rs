use crate::ffi;
use crate::graph::Tensor;
use crate::types::collect_owned_tensors;
use core::ffi::{c_char, c_void};
use std::ffi::CString;

fn optional_cstring(name: Option<&str>) -> Option<CString> {
    name.and_then(|value| CString::new(value).ok())
}

#[allow(clippy::ref_option)]
fn cstring_ptr(value: &Option<CString>) -> *const c_char {
    value
        .as_ref()
        .map_or(core::ptr::null(), |value| value.as_ptr())
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
    #[must_use]
    pub fn unary_arithmetic(
        &self,
        op: UnaryArithmeticOp,
        tensor: &Tensor,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

/// Calls the `MPSGraph` framework counterpart for `binary_arithmetic`.
    #[must_use]
    pub fn binary_arithmetic(
        &self,
        op: BinaryArithmeticOp,
        primary: &Tensor,
        secondary: &Tensor,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

/// Calls the `MPSGraph` framework counterpart for `select`.
    #[must_use]
    pub fn select(
        &self,
        predicate: &Tensor,
        true_tensor: &Tensor,
        false_tensor: &Tensor,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

/// Calls the `MPSGraph` framework counterpart for `relu_gradient`.
    #[must_use]
    pub fn relu_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

/// Calls the `MPSGraph` framework counterpart for `sigmoid_gradient`.
    #[must_use]
    pub fn sigmoid_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

/// Calls the `MPSGraph` framework counterpart for `softmax_gradient`.
    #[must_use]
    pub fn softmax_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        axis: isize,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

/// Calls the `MPSGraph` framework counterpart for `leaky_relu`.
    #[must_use]
    pub fn leaky_relu(&self, tensor: &Tensor, alpha: f64, name: Option<&str>) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

/// Calls the `MPSGraph` framework counterpart for `leaky_relu_tensor`.
    #[must_use]
    pub fn leaky_relu_tensor(
        &self,
        tensor: &Tensor,
        alpha_tensor: &Tensor,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

/// Calls the `MPSGraph` framework counterpart for `leaky_relu_gradient`.
    #[must_use]
    pub fn leaky_relu_gradient(
        &self,
        gradient: &Tensor,
        source: &Tensor,
        alpha_tensor: &Tensor,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

/// Calls the `MPSGraph` framework counterpart for `reduce_axis`.
    #[must_use]
    pub fn reduce_axis(
        &self,
        op: ReductionAxisOp,
        tensor: &Tensor,
        axis: isize,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

/// Calls the `MPSGraph` framework counterpart for `reduce_axes`.
    #[must_use]
    pub fn reduce_axes(
        &self,
        op: ReductionAxesOp,
        tensor: &Tensor,
        axes: &[usize],
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

/// Calls the `MPSGraph` framework counterpart for `concat_pair`.
    #[must_use]
    pub fn concat_pair(
        &self,
        first: &Tensor,
        second: &Tensor,
        dimension: isize,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

/// Calls the `MPSGraph` framework counterpart for `concat_tensors`.
    #[must_use]
    pub fn concat_tensors(
        &self,
        tensors: &[&Tensor],
        dimension: isize,
        interleave: bool,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

/// Calls the `MPSGraph` framework counterpart for `split_sizes`.
    #[must_use]
    pub fn split_sizes(
        &self,
        tensor: &Tensor,
        split_sizes: &[usize],
        axis: isize,
        name: Option<&str>,
    ) -> Vec<Tensor> {
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
        collect_owned_tensors(box_handle)
    }

/// Calls the `MPSGraph` framework counterpart for `split_sizes_tensor`.
    #[must_use]
    pub fn split_sizes_tensor(
        &self,
        tensor: &Tensor,
        split_sizes_tensor: &Tensor,
        axis: isize,
        name: Option<&str>,
    ) -> Vec<Tensor> {
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
        collect_owned_tensors(box_handle)
    }

/// Calls the `MPSGraph` framework counterpart for `split_num`.
    #[must_use]
    pub fn split_num(
        &self,
        tensor: &Tensor,
        num_splits: usize,
        axis: isize,
        name: Option<&str>,
    ) -> Vec<Tensor> {
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
        collect_owned_tensors(box_handle)
    }

/// Calls the `MPSGraph` framework counterpart for `stack`.
    #[must_use]
    pub fn stack(&self, tensors: &[&Tensor], axis: isize, name: Option<&str>) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

/// Calls the `MPSGraph` framework counterpart for `pad`.
    #[must_use]
    pub fn pad(
        &self,
        tensor: &Tensor,
        padding_mode: isize,
        left_padding: &[isize],
        right_padding: &[isize],
        constant_value: f64,
        name: Option<&str>,
    ) -> Option<Tensor> {
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
        wrap_tensor(ptr)
    }

/// Calls the `MPSGraph` framework counterpart for `top_k`.
    #[must_use]
    pub fn top_k(&self, source: &Tensor, k: usize, name: Option<&str>) -> Option<(Tensor, Tensor)> {
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_graph_top_k(self.as_ptr(), source.as_ptr(), k, cstring_ptr(&name))
        };
        wrap_tensor_pair(box_handle)
    }

/// Calls the `MPSGraph` framework counterpart for `top_k_tensor`.
    #[must_use]
    pub fn top_k_tensor(
        &self,
        source: &Tensor,
        k_tensor: &Tensor,
        name: Option<&str>,
    ) -> Option<(Tensor, Tensor)> {
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
        wrap_tensor_pair(box_handle)
    }
}
