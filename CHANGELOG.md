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
  which aborts the process. Those builders now return `None`, and runs and compilation
  return `Err(Error::InvalidShape)`.
- Executables accepted inputs that did not match their compiled feed types, and
  preallocated results of the wrong shape, which MPSGraph writes past. Both are rejected.
- The bridge's byte read ignored the destination length; it now refuses destinations
  shorter than the tensor.
- `ShapedType::shape` read the shape length and the dimensions in two bridge calls, so a
  concurrent `set_shape` with a longer shape made the bridge write past the result
  `Vec`. The copy now takes the destination length and the reader retries.

### Fixed

- `run_async_with_descriptor` returned result tensors before the GPU had written them.
  Results now come from `AsyncRun::wait` or `wait_timeout` after the work completes, GPU
  errors surface as `Error::ExecutionFailed`, and the caller's descriptor settings,
  including shared-event waits and signals, still apply.
- `TensorData::from_tensor` checked for macOS 16.0 instead of macOS 26 and aborted on
  tensors created without machine-learning usage; it now returns `None` for those.
- `run_with_descriptor` and `run_async_with_descriptor` trapped in the Swift bridge when
  called without preallocated results.

### Changed

- BREAKING: depend on `apple-metal` `>=0.10, <0.11` (was `0.8.5`), so the crate shares
  one `apple-metal` with the rest of the family.
- BREAKING: `TensorData::from_buffer` returns `Result<TensorData>` instead of `Option`.
- BREAKING: `Executable::run_async_with_descriptor` takes preallocated results by value
  and returns `Result<AsyncRun>`; the raw
  `ffi::mpsgraph_executable_run_async_with_descriptor` gains an `out_completion`
  parameter.
- BREAKING: the raw `ffi::mpsgraph_shaped_type_copy_shape` and
  `ffi::mpsgraph_tensor_copy_shape` take the destination length and return the shape's
  rank, or -1 when it is unranked.
- BREAKING: `Error` is `#[non_exhaustive]` and gains `BufferTooSmall`, `Overflow`,
  `InvalidShape`, `ExecutionFailed` and `Unsupported`.
- Data types newer than the running OS are refused instead of reaching MPSGraph.

### Added

- `data_type` constants for BFloat16, complex Float16/Float32/BFloat16,
  Int2/Int4/UInt2/UInt4, Float8 E4M3/E5M2/E8M0 and Float4 E2M1, plus `data_type_bits`.
  Sub-byte types pack across the whole array; `data_type_size` covers every byte-sized
  type.
- `AsyncRun` with `is_complete`, `wait` and `wait_timeout`.

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
