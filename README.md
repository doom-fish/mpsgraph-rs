# mpsgraph-rs

Safe Rust bindings for Apple's
[`MetalPerformanceShadersGraph`](https://developer.apple.com/documentation/metalperformanceshadersgraph)
framework on macOS.

The GitHub repository is `mpsgraph-rs`; the published crates.io package is
`apple-mpsgraph` because the short package name is already taken.

## Install

```bash
cargo add apple-mpsgraph apple-metal
```

## Quick start

```rust,no_run
use apple_metal::MetalDevice;
use apple_mpsgraph::{data_type, Feed, Graph, TensorData};

let device = MetalDevice::system_default().expect("no Metal device");
let graph = Graph::new().expect("graph");
let input = graph
    .placeholder(Some(&[2, 2]), data_type::FLOAT32, Some("input"))
    .expect("input placeholder");
let bias = graph.constant_scalar(1.0, data_type::FLOAT32).expect("bias");
let added = graph.addition(&input, &bias, Some("add")).expect("add");
let output = graph.relu(&added, Some("relu")).expect("relu");
let data = TensorData::from_f32_slice(&device, &[1.0, -2.0, 3.0, -4.0], &[2, 2])
    .expect("tensor data");
let results = graph
    .run(&[Feed::new(&input, &data)], &[&output])
    .expect("graph run");
let values = results[0].read_f32().expect("read result");
assert_eq!(values, vec![2.0, 0.0, 4.0, 0.0]);
```

## v0.2.3 surface

- Core wrappers for `Graph`, `Tensor`, `TensorData`, `Executable`, `Feed`, and `FeedDescription`
- Metadata and descriptor coverage for `GraphDevice`, `ShapedType`, `GraphType`, `Object`, `Operation`, `VariableOp`, `CompilationDescriptor`, `ExecutionDescriptor`, `ExecutableExecutionDescriptor`, `ExecutableSerializationDescriptor`, and the audited convolution / FFT / pooling / sparse / stencil descriptor families
- Graph / executable introspection helpers such as `placeholder_tensors`, `feed_tensors`, `target_tensors`, `output_types`, tensor `shape`, tensor `data_type`, tensor-data `graph_device`, and raw shared-event wait/signal hooks on execution descriptors
- Graph construction and execution helpers for:
  - placeholders, constants, variable tensors, `read_variable`, and `assign_variable`
  - matrix multiplication, `band_part`, and matrix inverse
  - unary arithmetic (`identity`, exponent/log variants, square/sqrt/reciprocal, abs/neg/sign, rounding, trig/hyperbolic, `isNaN`, `isInfinite`)
  - binary arithmetic (`+`, `-`, `*`, `/`, `divisionNoNaN`, `power`, min/max, comparisons, logical `and`/`or`, `atan2`, `floorModulo`, `select`)
  - activations (`reLU`, `leakyReLU`, `sigmoid`, `softMax`) and gradient helpers for `reLU`, `sigmoid`, and `softMax`
  - shape ops (`reshape`, `transpose`/`permute`, `slice`, `broadcast`, `concat`, `split`, `stack`, `pad`)
  - reductions, cumulative sum, `topK`, and `top_k_gradient`
  - convolution helpers (`convolution2d`, `convolution_transpose2d`, `convolution3d`, `depthwise_convolution2d`, `depthwise_convolution3d`)
  - pooling helpers (`max_pooling2d`, `max_pooling4d`, `max_pooling4d_return_indices`), normalization, FFT, and `im_to_col`
  - loss / labeling helpers (`softmax_cross_entropy`, `one_hot`, `non_zero_indices`, `non_maximum_suppression`)
  - quantization / resize / sample-grid / scatter / sort / sparse / stencil helpers
  - call ops via `Graph::call` plus `CompilationDescriptor::set_callable`
  - control-flow builders for control dependencies, `if`/`then`/`else`, `while`, and `for`
  - gather ops (`gather`, `gatherND`, `gatherAlongAxis`, `gatherAlongAxisTensor`)
  - descriptor-driven random ops (`RandomOpDescriptor`, seeded/stateful random tensors, dropout)
  - recurrent layers (`singleGateRNN`, `LSTM`, `GRU`) plus descriptor wrappers
- Shared constants for `MPSDataType`, `MPSGraphTensorNamedDataLayout`, `MPSGraphPaddingStyle`, graph options, optimization levels, deployment platform values, random distributions, random sampling modes, RNN activations, execution stages, reduction / FFT / loss / resize / scatter / sparse / NMS coordinate enums, and pooling return-indices modes

This crate now covers the full 90-symbol audited SDK surface. See [`COVERAGE.md`](COVERAGE.md) for the header-by-header status and the broader SDK families that remain partial.

## Smoke examples

```bash
cargo run --example 01_add_relu
cargo run --example 02_compile_matmul
cargo run --example 03_arithmetic_topk
cargo run --example 04_descriptor_compile
cargo run --example 05_concat_split
cargo run --example 06_control_flow_call
cargo run --example 07_gather_random_rnn
```
