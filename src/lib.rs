#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

/// Groups `MPSGraph` framework constants for `call`.
pub mod call;
/// Groups `MPSGraph` framework constants for `control_flow`.
pub mod control_flow;
/// Groups `MPSGraph` framework constants for `data`.
pub mod data;
/// Groups `MPSGraph` framework constants for `error`.
pub mod error;
/// Groups `MPSGraph` framework constants for `execution`.
pub mod execution;
/// Groups `MPSGraph` framework constants for `ffi`.
pub mod ffi;
/// Groups `MPSGraph` framework constants for `gather`.
pub mod gather;
/// Groups `MPSGraph` framework constants for `graph`.
pub mod graph;
/// Groups `MPSGraph` framework constants for `ops`.
pub mod ops;
/// Groups `MPSGraph` framework constants for `random`.
pub mod random;
/// Groups `MPSGraph` framework constants for `rnn`.
pub mod rnn;
/// Groups `MPSGraph` framework constants for `specialized`.
pub mod specialized;
/// Groups `MPSGraph` framework constants for `types`.
pub mod types;

/// Re-exports the `MPSGraph` framework surface for this item.
pub use crate::control_flow::WhileBeforeResult;
/// Re-exports the `MPSGraph` framework surface for this item.
pub use crate::data::TensorData;
/// Re-exports the `MPSGraph` framework surface for this item.
pub use crate::error::{Error, Result};
/// Re-exports the `MPSGraph` framework surface for this item.
pub use crate::execution::{
    deployment_platform, graph_options, optimization, optimization_profile,
    reduced_precision_fast_math, CompilationDescriptor, ExecutableExecutionDescriptor,
    ExecutableSerializationDescriptor, ExecutionDescriptor,
};
/// Re-exports the `MPSGraph` framework surface for this item.
pub use crate::graph::{
    data_type, data_type_size, padding_mode, padding_style, tensor_named_data_layout,
    Convolution2DDescriptor, Convolution2DDescriptorInfo, Executable, Feed, FeedDescription, Graph,
    Pooling2DDescriptor, Pooling2DDescriptorInfo, Tensor,
};
/// Re-exports the `MPSGraph` framework surface for this item.
pub use crate::ops::{BinaryArithmeticOp, ReductionAxesOp, ReductionAxisOp, UnaryArithmeticOp};
/// Re-exports the `MPSGraph` framework surface for this item.
pub use crate::random::{random_distribution, random_normal_sampling_method, RandomOpDescriptor};
/// Re-exports the `MPSGraph` framework surface for this item.
pub use crate::rnn::{rnn_activation, GRUDescriptor, LSTMDescriptor, SingleGateRNNDescriptor};
/// Re-exports the `MPSGraph` framework surface for this item.
pub use crate::specialized::{
    execution_stage, fft_scaling_mode, loss_reduction_type,
    non_maximum_suppression_coordinate_mode, pooling_return_indices_mode, reduction_mode,
    resize_mode, resize_nearest_rounding_mode, scatter_mode, sparse_storage_type,
    Convolution3DDescriptor, Convolution3DDescriptorInfo, CreateSparseDescriptor,
    DepthwiseConvolution2DDescriptor, DepthwiseConvolution2DDescriptorInfo,
    DepthwiseConvolution3DDescriptor, DepthwiseConvolution3DDescriptorInfo, FftDescriptor,
    FftDescriptorInfo, GraphType, ImToColDescriptor, ImToColDescriptorInfo, Object,
    Pooling4DDescriptor, Pooling4DDescriptorInfo, StencilDescriptor, StencilDescriptorInfo,
    VariableOp,
};
/// Re-exports the `Metal` tensor handle used by `TensorData::from_tensor`.
pub use apple_metal::MetalTensor;
/// Re-exports the `MPSGraph` framework surface for this item.
pub use crate::types::{graph_device_type, GraphDevice, Operation, ShapedType};
