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
    value.as_ref().map_or(core::ptr::null(), |value| value.as_ptr())
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum UnaryArithmeticOp {
    Identity = 0,
    Exponent = 1,
    ExponentBase2 = 2,
    ExponentBase10 = 3,
    Logarithm = 4,
    LogarithmBase2 = 5,
    LogarithmBase10 = 6,
    Square = 7,
    SquareRoot = 8,
    Reciprocal = 9,
    Absolute = 10,
    Negative = 11,
    Sign = 12,
    SignBit = 13,
    Ceil = 14,
    Floor = 15,
    Round = 16,
    Rint = 17,
    Sin = 18,
    Cos = 19,
    Tan = 20,
    Sinh = 21,
    Cosh = 22,
    Tanh = 23,
    Asin = 24,
    Acos = 25,
    Atan = 26,
    Asinh = 27,
    Acosh = 28,
    Atanh = 29,
    IsNaN = 30,
    IsInfinite = 31,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum BinaryArithmeticOp {
    Addition = 0,
    Subtraction = 1,
    Multiplication = 2,
    Division = 3,
    DivisionNoNaN = 4,
    Power = 5,
    Minimum = 6,
    Maximum = 7,
    Equal = 8,
    NotEqual = 9,
    GreaterThan = 10,
    GreaterThanOrEqualTo = 11,
    LessThan = 12,
    LessThanOrEqualTo = 13,
    LogicalAnd = 14,
    LogicalOr = 15,
    Atan2 = 16,
    FloorModulo = 17,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ReductionAxisOp {
    Sum = 0,
    Maximum = 1,
    Minimum = 2,
    Product = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ReductionAxesOp {
    Sum = 0,
    Maximum = 1,
    Minimum = 2,
    Product = 3,
}

impl crate::graph::Graph {
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

    #[must_use]
    pub fn leaky_relu(
        &self,
        tensor: &Tensor,
        alpha: f64,
        name: Option<&str>,
    ) -> Option<Tensor> {
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

    #[must_use]
    pub fn concat_tensors(
        &self,
        tensors: &[&Tensor],
        dimension: isize,
        interleave: bool,
        name: Option<&str>,
    ) -> Option<Tensor> {
        let name = optional_cstring(name);
        let handles = tensors.iter().map(|tensor| tensor.as_ptr()).collect::<Vec<_>>();
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

    #[must_use]
    pub fn stack(&self, tensors: &[&Tensor], axis: isize, name: Option<&str>) -> Option<Tensor> {
        let name = optional_cstring(name);
        let handles = tensors.iter().map(|tensor| tensor.as_ptr()).collect::<Vec<_>>();
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

    #[must_use]
    pub fn top_k(&self, source: &Tensor, k: usize, name: Option<&str>) -> Option<(Tensor, Tensor)> {
        let name = optional_cstring(name);
        // SAFETY: all handles remain valid for the duration of the call.
        let box_handle = unsafe {
            ffi::mpsgraph_graph_top_k(self.as_ptr(), source.as_ptr(), k, cstring_ptr(&name))
        };
        wrap_tensor_pair(box_handle)
    }

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
