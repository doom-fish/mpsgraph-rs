#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod data;
pub mod error;
pub mod ffi;
pub mod graph;

pub use crate::data::TensorData;
pub use crate::error::{Error, Result};
pub use crate::graph::{
    data_type, data_type_size, padding_style, tensor_named_data_layout, Convolution2DDescriptor,
    Convolution2DDescriptorInfo, Executable, Feed, FeedDescription, Graph, Pooling2DDescriptor,
    Pooling2DDescriptorInfo, Tensor,
};
