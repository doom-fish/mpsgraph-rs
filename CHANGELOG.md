# Changelog

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
