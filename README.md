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

## v0.1 surface

- `Graph`, `Tensor`, and ordered `Feed` / `FeedDescription` helpers
- `TensorData` from CPU bytes, `f32` slices, or existing `MTLBuffer`s
- Direct graph execution plus compiled `Executable` runs on `MTLCommandQueue`
- Core graph construction ops:
  - placeholders and constants
  - addition, subtraction, multiplication, division
  - matrix multiplication
  - reshape, transpose/permute, slice, broadcast
  - `reLU`, `sigmoid`, `softMax`
  - reduction sum / max / min and mean
  - 2D convolution, max pooling, and normalization helpers
- Shared constants for `MPSDataType`, `MPSGraphTensorNamedDataLayout`, and `MPSGraphPaddingStyle`

## Smoke examples

```bash
cargo run --example 01_add_relu
cargo run --example 02_compile_matmul
```
