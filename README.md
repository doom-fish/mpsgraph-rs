# mpsgraph-rs

Safe Rust bindings for Apple's
[`MetalPerformanceShadersGraph`](https://developer.apple.com/documentation/metalperformanceshadersgraph)
framework on macOS.

The GitHub repository is `mpsgraph-rs`; the published crates.io package is
`apple-mpsgraph` because the short package name is already taken.

## Install

```toml
[dependencies]
apple-mpsgraph = "0.3"
apple-metal = "0.10"
```

Requires macOS 11 or later and Rust 1.82. Many newer graph APIs need macOS 12 to 15 and
return `None` or `Err` on older systems. `TensorData::from_tensor` needs macOS 26, and
the Float8 and Float4 data types need macOS 27.

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

## Surface

- Core wrappers for `Graph`, `Tensor`, `TensorData`, `Executable`, `Feed`, and `FeedDescription`
- Tensor data from bytes, `f32` slices, `MTLBuffer`s and (macOS 26) `MTLTensor`s, with byte counts derived from each data type's bit width
- Synchronous executable runs plus `run_async_with_descriptor`, which returns an `AsyncRun` to wait on
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
- Shared constants for `MPSDataType` (including `BFloat16`, the complex types, Int2/Int4/UInt2/UInt4, and the Float8/Float4 types), `MPSGraphTensorNamedDataLayout`, `MPSGraphPaddingStyle`, graph options, optimization levels, deployment platform values, random distributions, random sampling modes, RNN activations, execution stages, reduction / FFT / loss / resize / scatter / sparse / NMS coordinate enums, and pooling return-indices modes

Every top-level interface, category and enum in the SDK headers has at least one wrapper,
but most categories are only partly wrapped. Scaled dot-product attention and the
tensor-scale, per-axis and lookup-table quantization variants are among the missing
methods. See [`COVERAGE.md`](COVERAGE.md) for the per-area status.

## Safety notes

- `TensorData::from_bytes` and `from_buffer` check the byte count against the shape and
  data type with overflow-checked arithmetic, and reads never write past their destination.
  Sub-byte types pack across the whole array.
- Shapes are checked for overflow before they reach the Swift bridge. Every graph builder
  checks the preconditions `MPSGraph` documents or asserts on (ranks, axes, shape
  compatibility and broadcasting, gather and scatter index types, `k` against the last
  dimension, pad widths, stack and concat inputs, RNN weight, state and mask layouts,
  convolution channels, groups and window sizes, supported data types and descriptor
  values) and returns `Err` instead of letting `MPSGraph` abort the process. Dimensions that
  are dynamic (-1) or unranked are checked as far as they are known; `MPSGraph` checks the
  rest when the graph runs and aborts if they do not fit.
- Each tensor and operation remembers the graph and control-flow block it was created in.
  Using one in another graph, or outside the `if`, `while` or `for` block that created it,
  returns `Error::ForeignTensor` or `Error::ForeignOperation`.
- Control-flow blocks must agree: the then and else blocks of `if_then_else` return the
  same number of tensors (at least one) with identical shapes and data types, the `while`
  before block returns a rank-0 bool predicate and at least one tensor, and the `while`
  after block and `for` body return tensors matching the loop's inputs. A block that breaks
  these rules is replaced by placeholders of the expected types, so the graph stays valid,
  and the builder returns an error. `MPSGraph` cannot build an `if` without an else block, so
  there is no `if_then`.
- Runs and compiles need a feed for every placeholder their targets depend on, including
  placeholders captured inside control-flow blocks (`Error::MissingFeed`). A graph with a
  `call` op cannot use `run` or `compile`; `compile_with_descriptor` needs a callable set
  with `CompilationDescriptor::set_callable` whose compiled feed and output types match the
  call (`Error::MissingCallable`).
- `top_k_tensor`, `split_sizes_tensor`, `gather_along_axis_tensor`, `resize_nearest` and the
  shape-tensor random builders are `unsafe`: `MPSGraph` aborts if the values of their tensor
  parameters are out of range, and those values are known only when the graph runs.
- `Graph`, `Executable`, `ShapedType` and the mutable descriptors are `Send` but not `Sync`,
  because their builders and setters mutate the underlying objects.
- An `AsyncRun` hands out its results only from `wait` or `wait_timeout`, after the GPU work
  has finished; dropping it blocks until then.
- `non_maximum_suppression` is `unsafe`: on the Apple-silicon GPU runtime used for testing,
  running a graph that contains it aborts with "Unsupported MPS operation".

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
