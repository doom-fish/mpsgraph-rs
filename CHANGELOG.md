# Changelog

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
