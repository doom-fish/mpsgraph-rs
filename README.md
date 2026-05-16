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

## v0.2 surface

- Core wrappers for `Graph`, `Tensor`, `TensorData`, `Executable`, `Feed`, and `FeedDescription`
- Metadata and descriptor coverage for `GraphDevice`, `ShapedType`, `Operation`, `CompilationDescriptor`, `ExecutionDescriptor`, `ExecutableExecutionDescriptor`, and `ExecutableSerializationDescriptor`
- Graph / executable introspection helpers such as `placeholder_tensors`, `feed_tensors`, `target_tensors`, `output_types`, tensor `shape`, tensor `data_type`, and tensor-data `graph_device`
- Graph construction and execution helpers for:
  - placeholders and constants
  - matrix multiplication
  - unary arithmetic (`identity`, exponent/log variants, square/sqrt/reciprocal, abs/neg/sign, rounding, trig/hyperbolic, `isNaN`, `isInfinite`)
  - binary arithmetic (`+`, `-`, `*`, `/`, `divisionNoNaN`, `power`, min/max, comparisons, logical `and`/`or`, `atan2`, `floorModulo`, `select`)
  - activations (`reLU`, `leakyReLU`, `sigmoid`, `softMax`) and gradient helpers for `reLU`, `sigmoid`, and `softMax`
  - shape ops (`reshape`, `transpose`/`permute`, `slice`, `broadcast`, `concat`, `split`, `stack`, `pad`)
  - reductions (existing sum/max/min/mean plus axis/axes sum/max/min/product)
  - `topK`
  - 2D convolution, max pooling, and normalization helpers
- Shared constants for `MPSDataType`, `MPSGraphTensorNamedDataLayout`, `MPSGraphPaddingStyle`, graph options, optimization levels, and deployment platform values

This crate still covers a subset of the full SDK. See [`COVERAGE.md`](COVERAGE.md) for the audited header-by-header status and deferred areas.

## Smoke examples

```bash
cargo run --example 01_add_relu
cargo run --example 02_compile_matmul
cargo run --example 03_arithmetic_topk
cargo run --example 04_descriptor_compile
cargo run --example 05_concat_split
```
