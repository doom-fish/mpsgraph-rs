# Changelog

All notable changes to `apple-mpsgraph` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - Unreleased

### Security

- `TensorData::from_buffer` did not check the buffer length against the shape and data
  type, so reads and graph execution could run past the `MTLBuffer`. It now computes the
  required length with overflow-checked arithmetic and returns `Error::BufferTooSmall` or
  `Error::Overflow`.
- The Swift bridge trapped on dimensions above `isize::MAX`. Shapes are now checked
  before they cross the bridge, and oversized ones return `None` or `Err`.
- Out-of-range axes, incompatible operand shapes, invalid reshapes, permutations, slices,
  broadcasts and splits, and feeds that do not match their placeholders reached MPSGraph,
  which aborts the process. Those builders now return errors, and runs and compilation
  return `Err(Error::InvalidShape)`.
- Executables accepted inputs that did not match their compiled feed types, and
  preallocated results of the wrong shape, which MPSGraph writes past. Both are rejected.
- The bridge's byte read ignored the destination length; it now refuses destinations
  shorter than the tensor.
- `ShapedType::shape` read the shape length and the dimensions in two bridge calls, so a
  concurrent `set_shape` with a longer shape made the bridge write past the result
  `Vec`. The copy now takes the destination length and the reader retries.
- The specialized, gather, scatter, random, RNN, control-flow, concat/stack/pad and
  top-K builders passed invalid ranks, axes, shapes, index types and descriptor values
  to MPSGraph, which aborts the process when the op is built or when any later run
  compiles the graph, even if the op is not a target. They now check the documented
  preconditions and return errors; dimensions that are dynamic (-1) or unranked are
  checked as far as they are known.
- Tensors from another graph, and tensors created inside an `if`, `while` or `for`
  block that has ended, aborted MPSGraph when used. Tensors and operations now record
  the graph and block they were created in, and builders, runs and compiles refuse
  foreign ones with `Error::ForeignTensor` or `Error::ForeignOperation`.
- Control-flow blocks whose results disagreed in count, data type or shape (including
  dynamic against static dimensions), `if` blocks without results, `while` predicates
  that are not rank-0 bool tensors, `while` before-blocks without results and `for`
  loops without body arguments aborted MPSGraph. The bridge now checks the block
  results, fills a refused block with placeholders of the expected types so the graph
  stays valid, and the builder returns an error.
- Running or compiling a graph whose targets depend on an unfed placeholder, including
  one captured inside a control-flow block, aborted MPSGraph. Runs and compiles now
  trace the targets' dependencies and return `Error::MissingFeed`.
- A graph containing a call op aborted every run and compile unless a callable with the
  same feed and output types was set. Runs now return `Error::MissingCallable`, and
  `compile_with_descriptor` checks that the descriptor holds a callable set through
  `CompilationDescriptor::set_callable` whose compiled types match the call.
- `Graph`, `Executable`, `ShapedType`, `RandomOpDescriptor`, the RNN descriptors and the
  compilation, execution and serialization descriptors were `Sync`, although builders
  and setters mutate the Objective-C objects through `&self` and their properties are
  nonatomic, so a `ShapedType::shape` racing `set_shape` could read a released array.
  They are now `Send` but not `Sync`.
- Splatted constants with a zero dimension or a sub-byte or `unorm8` type, `isNaN` and
  `isInfinite` on non-float tensors, `softmax` and matrix multiplication on integer
  tensors, and zero strides, kernel sizes, dilation rates or groups in descriptors
  aborted MPSGraph; they now return errors.
- `Executable::output_types` and `specialize` with input types other than the compiled
  feeds, and `serialize_package` with an unknown deployment platform or a minimum
  deployment target below MPSGraph's minimum for the platform (macOS 14, iOS and tvOS
  17, visionOS 1.1), aborted MPSGraph; they now return errors.

### Fixed

- `run_async_with_descriptor` returned result tensors before the GPU had written them.
  Results now come from `AsyncRun::wait` or `wait_timeout` after the work completes, GPU
  errors surface as `Error::ExecutionFailed`, and the caller's descriptor settings,
  including shared-event waits and signals, still apply.
- `TensorData::from_tensor` checked for macOS 16.0 instead of macOS 26 and aborted on
  tensors created without machine-learning usage; it now returns `None` for those.
- `run_with_descriptor` and `run_async_with_descriptor` trapped in the Swift bridge when
  called without preallocated results.
- The Swift while-loop bridge called `fatalError` when the before block returned no
  predicate.

### Changed

- **BREAKING:** depend on `apple-metal` `>=0.10, <0.11` (was `0.8.5`), so the crate shares
  one `apple-metal` with the rest of the family.
- **BREAKING:** `TensorData::from_buffer` returns `Result<TensorData>` instead of `Option`.
- **BREAKING:** `Executable::run_async_with_descriptor` takes preallocated results by value
  and returns `Result<AsyncRun>`; the raw
  `ffi::mpsgraph_executable_run_async_with_descriptor` gains an `out_completion`
  parameter.
- **BREAKING:** the raw `ffi::mpsgraph_shaped_type_copy_shape` and
  `ffi::mpsgraph_tensor_copy_shape` take the destination length and return the shape's
  rank, or -1 when it is unranked.
- **BREAKING:** `Error` is `#[non_exhaustive]` and gains `BufferTooSmall`, `Overflow`,
  `InvalidShape`, `ExecutionFailed` and `Unsupported`.
- **BREAKING:** every graph builder returns `Result` (`Result<Tensor>`,
  `Result<(Tensor, Tensor)>`, `Result<Vec<Tensor>>` or `Result<Operation>`) instead of
  `Option` or an empty `Vec`. `Graph::compile` and `compile_with_descriptor` return
  `Result<Executable>`, and the descriptor constructors that check their parameters
  (`Convolution2DDescriptor`, `Pooling2DDescriptor`, `Convolution3DDescriptor`,
  `DepthwiseConvolution2DDescriptor`, `DepthwiseConvolution3DDescriptor`,
  `FftDescriptor`, `ImToColDescriptor`, `Pooling4DDescriptor`,
  `CreateSparseDescriptor`, `StencilDescriptor` and `RandomOpDescriptor`) return
  `Result<Self>`.
- **BREAKING:** `top_k_tensor`, `split_sizes_tensor`, `gather_along_axis_tensor`,
  `resize_nearest`, `random_tensor_shape_tensor`, `random_tensor_shape_tensor_seed` and
  `random_tensor_shape_tensor_state` are `unsafe`: MPSGraph aborts when the values in
  their tensor parameters (k, split sizes, axis, output size or shape) are out of
  range, and those values are known only when the graph runs. Their data types and
  ranks are still checked.
- **BREAKING:** `non_maximum_suppression` is `unsafe`: on the Apple-silicon runtime used
  for testing, running it aborts with "Unsupported MPS operation". Its operand types
  and shapes are checked.
- **BREAKING:** `Graph`, `Executable`, `ShapedType`, `RandomOpDescriptor`,
  `SingleGateRNNDescriptor`, `LSTMDescriptor`, `GRUDescriptor`, `CompilationDescriptor`,
  `ExecutionDescriptor`, `ExecutableExecutionDescriptor` and
  `ExecutableSerializationDescriptor` are no longer `Sync`.
- **BREAKING:** runs and compiles take only this graph's placeholders as feeds, need a
  feed for every placeholder their targets depend on, refuse targets created inside a
  control-flow block, and refuse to run while a control-flow block is being built.
  `placeholder_tensors` lists only placeholders created through `Graph::placeholder`.
- **BREAKING:** random descriptors accept float16, float32 and int32 for uniform
  distributions and float16 and float32 for normal ones, as the header documents;
  `pad`, `sample_grid` and stencil boundaries refuse `PERIODIC` and `ANTI_PERIODIC`,
  which the GPU runtime does not run; 4D pooling needs a source of rank 4 or more;
  sparse tensors take float32 values with int32 or int64 indices. Convolution weights,
  gradient operands, normalization statistics and every recurrent-layer input need the
  source's data type, although MPSGraph converts some mixed combinations itself.
- **BREAKING:** `Error` also gains `InvalidDataType`, `InvalidArgument`,
  `ForeignTensor`, `ForeignOperation`, `MissingFeed` and `MissingCallable`.
- **BREAKING:** the raw control-flow `ffi` functions take a `failed` out-pointer, and
  `ffi::mpsgraph_compilation_descriptor_callable` is new.
- Data types newer than the running OS are refused instead of reaching MPSGraph.

### Added

- `data_type` constants for BFloat16, complex Float16/Float32/BFloat16,
  Int2/Int4/UInt2/UInt4, Float8 E4M3/E5M2/E8M0 and Float4 E2M1, plus `data_type_bits`.
  Sub-byte types pack across the whole array; `data_type_size` covers every byte-sized
  type.
- `AsyncRun` with `is_complete`, `wait` and `wait_timeout`.

### Removed

- **BREAKING:** `Graph::if_then`. MPSGraph requires an `if` without an else block to
  return no tensors and then aborts on such an `if`, so the method could never succeed.

## [0.2.8] - 2026-05-19

### Added

- Added `TensorData::from_tensor` plus a root `MetalTensor` re-export so `MPSGraphTensorData` can alias `id<MTLTensor>` on macOS 16+.

## [0.2.7] - 2026-05-18

- Add one-line docs across the public safe and FFI surfaces, raising public-item rustdoc coverage to 99.8%.

## [0.2.6] - 2026-05-18

- Widen apple-metal version bound so the 0.x bump dep resolves. No source changes.

## 0.2.5 - 2026-05-17

- Added missing SAFETY comments on context-pointer casts in callback trampolines (`zero_arg_tensor_array_trampoline`, `while_before_trampoline`, `tensor_array_input_trampoline`, `for_body_trampoline`). These unsafe blocks were previously undocumented, improving transparency of the unsafe FFI boundary logic.

## 0.2.4 - 2026-05-17

- Added `@available(macOS 26.0, *)` declaration attributes to the two Swift bridge thunks that reference `MPSGraphReducedPrecisionFastMath` / `reducedPrecisionFastMath` (`mpsgraph_compilation_descriptor_reduced_precision_fast_math` and `mpsgraph_compilation_descriptor_set_reduced_precision_fast_math`), replacing the previous runtime-only `guard #available` pattern. The bridge now correctly signals macOS 26+ availability at compile time, making it portable to older SDK builds.

## 0.2.3 - 2026-05-17

- Fixed the specialized Swift bridge to construct audited descriptors with `init()` plus property assignment, matching current SDK Swift overlays.
- Added specialized wrappers for `MPSGraphObject`, `MPSGraphType`, `MPSGraphVariableOp`, `MPSGraphExecutionStage`, convolution-transpose / 3D / depthwise descriptors and ops, FFT, Im2Col, loss, matrix-inverse, variable read/assign, pooling-4D + return indices, quantization, resize, sample-grid, scatter, sort, sparse, stencil, non-zero/NMS, optimizer SGD, and TopK gradient APIs.
- Added specialized smoke tests and updated the audit/docs to reflect full 90/90 audited SDK coverage.

## 0.2.1 - 2026-05-16

- Added call-op support via `Graph::call` and `CompilationDescriptor::set_callable`
- Added control-flow builders for control dependencies, `if`/`then`/`else`, `while`, and `for`
- Added gather, random/dropout, and recurrent-op bindings plus descriptor wrappers for random and RNN APIs
- Added advanced integration tests and new smoke examples covering call/control-flow and gather/random/RNN surfaces

## 0.2.0 - 2026-05-16

- Added `MPSGraphDevice`, `MPSGraphShapedType`, tensor metadata, and tensor-data device introspection wrappers
- Added descriptor and executable support for `MPSGraphCompilationDescriptor`, `MPSGraphExecutionDescriptor`, `MPSGraphExecutableExecutionDescriptor`, and `MPSGraphExecutableSerializationDescriptor`
- Added executable metadata helpers including placeholder, feed, target, and output-type queries
- Added opcode-driven arithmetic, activation-gradient, reduction, concat/split/stack/pad, and `topK` bindings
- Added coverage documentation, integration tests, and additional smoke examples for descriptors and op families

## 0.1.0 - 2026-05-16

- Initial release of `apple-mpsgraph`
- Added safe wrappers for `MPSGraph`, `MPSGraphTensor`, `MPSGraphTensorData`, and compiled executables
- Added tensor construction, arithmetic, reshape, broadcast, reduction, activation, convolution, pooling, and normalization helpers
- Added direct-run and compiled-executable smoke examples
