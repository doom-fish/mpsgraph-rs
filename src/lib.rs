#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod data;
pub mod error;
pub mod execution;
pub mod ffi;
pub mod graph;
pub mod ops;
pub mod types;

pub use crate::data::TensorData;
pub use crate::error::{Error, Result};
pub use crate::execution::{
    deployment_platform, graph_options, optimization, optimization_profile,
    reduced_precision_fast_math, CompilationDescriptor, ExecutableExecutionDescriptor,
    ExecutableSerializationDescriptor, ExecutionDescriptor,
};
pub use crate::graph::{
    data_type, data_type_size, padding_style, tensor_named_data_layout, Convolution2DDescriptor,
    Convolution2DDescriptorInfo, Executable, Feed, FeedDescription, Graph, Pooling2DDescriptor,
    Pooling2DDescriptorInfo, Tensor,
};
pub use crate::ops::{BinaryArithmeticOp, ReductionAxesOp, ReductionAxisOp, UnaryArithmeticOp};
pub use crate::types::{graph_device_type, GraphDevice, Operation, ShapedType};
