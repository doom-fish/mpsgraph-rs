# Coverage

Audited against the macOS SDK `MetalPerformanceShadersGraph.framework/Headers` surface and Swift symbol graph extraction.
Each of the 90 top-level interfaces, categories and enums in the macOS 26.5 SDK headers has at least
one wrapper (`90/90`). That count is per type, not per method: most categories are only partly
wrapped, as the table below shows.

Legend:
- ✅ implemented in this crate
- 🟡 partially implemented
- ⏭️ intentionally deferred / currently unsupported

## Header and category status

| Requested area | SDK/header mapping | Status | Notes |
| --- | --- | --- | --- |
| Core graph construction | `MPSGraph.h`, `MPSGraphCore.h`, `MPSGraphTensor.h` | 🟡 | `Graph`, `Tensor`, placeholders, constants, direct run, compile, tensor metadata (`shape`, `data_type`, `operation`), graph options, placeholder introspection, and newer specialized category entry points. Broader convenience overloads remain deferred. |
| Tensor data | `MPSGraphTensorData.h` | 🟡 | CPU bytes / `f32` slices / length-checked `MTLBuffer` / `MTLTensor` aliasing (macOS 26), readback helpers, `graph_device`; matrix/vector/NDArray/image initializers remain deferred. |
| Data types | `MPSDataType` in `MPSCoreTypes.h` | ✅ | `data_type` constants for every type the SDK defines, including BFloat16, complex, Int2/Int4/UInt2/UInt4 and Float8/Float4, with `data_type_bits` / `data_type_size`. Types newer than the running OS are refused at run time. |
| Device and types | `MPSGraphDevice.h` | ✅ | `GraphDevice`, `ShapedType`, `GraphType`, `Object`, `VariableOp`, graph device creation from `MTLDevice`, plus shape/data-type mutation and inspection. |
| Executable / descriptors | `MPSGraphExecutable.h` | ✅ | `CompilationDescriptor`, `ExecutionDescriptor`, `ExecutableExecutionDescriptor`, `ExecutableSerializationDescriptor`, executable feed/target/output-type queries, package load/save, callable registration, raw shared-event wait/signal hooks, and `run_async_with_descriptor` returning a waitable `AsyncRun`. |
| Activation ops | `MPSGraphActivationOps.h` | 🟡 | `reLU`, `sigmoid`, `softMax`, `leakyReLU`, `reLUGradient`, `sigmoidGradient`, `softMaxGradient`; broader activation families remain deferred. |
| Arithmetic ops | `MPSGraphArithmeticOps.h` | 🟡 | Unary arithmetic enum covers identity, exponent/log, square/sqrt/reciprocal, abs/neg/sign, rounding, trig/hyperbolic, `isNaN`, `isInfinite`; binary arithmetic enum covers add/sub/mul/div, `divisionNoNaN`, `power`, min/max, comparisons, logical ops, `atan2`, `floorModulo`, plus `select`. Other arithmetic entry points remain deferred. |
| Tensor shape ops | `MPSGraphTensorShapeOps.h` | 🟡 | Existing reshape / transpose / slice / broadcast plus `concat`, `split`, `stack`, and `pad`. Many advanced slice, gather-like, and indexing variants remain deferred. |
| Reduction ops | `MPSGraphReductionOps.h` | 🟡 | Existing sum/max/min/mean plus axis/axes sum/max/min/product helpers. Arg reductions and the rest of the reduction family remain deferred. |
| Top-K ops | `MPSGraphTopKOps.h` | 🟡 | `topK`, tensor-`k`, and `top_k_gradient`; bottom-K and the remaining axis/tensor overloads remain deferred. |
| Automatic differentiation | `MPSGraphAutomaticDifferentiation.h` | 🟡 | Activation-gradient helpers only. General gradient/JVP/VJP families remain deferred. |
| Pooling ops | `MPSGraphPoolingOps.h` | 🟡 | Max-pooling 2D plus `Pooling4DDescriptor`, `max_pooling4d`, and `max_pooling4d_return_indices`. Average/L2/global/adaptive pooling remain deferred. |
| Convolution ops | convolution APIs on `MPSGraph.h` / related descriptors | 🟡 | `convolution2d`, `convolution_transpose2d`, `Convolution3DDescriptor`/`convolution3d`, and depthwise 2D/3D descriptors + ops. Fused and gradient-heavy convolution families remain deferred. |
| Normalization ops | normalization APIs on `MPSGraph.h` | 🟡 | Existing normalize helper only. Broader normalization families remain deferred. |
| FFT ops | `MPSGraphFourierTransformOps.h` | 🟡 | `fft_scaling_mode`, `FftDescriptor`, and `fast_fourier_transform`; real/Hermitean FFT variants remain deferred. |
| Random ops | `MPSGraphRandomOps.h` | 🟡 | `RandomOpDescriptor`, seeded/stateful descriptor-driven random tensors, Philox state tensors, and dropout. Random-uniform convenience overloads are still deferred. |
| Gather ops | `MPSGraphGatherOps.h` (includes GatherND) | 🟡 | `gather`, `gatherND`, `gatherAlongAxis`, and `gatherAlongAxisTensor`. |
| Scatter ops | `MPSGraphScatterNDOps.h` | 🟡 | `scatter_nd`, `scatter`, and `scatter_along_axis`. Data-tensor and tensor-axis variants remain deferred. |
| Control flow | `MPSGraphControlFlowOps.h` | 🟡 | Control dependencies plus `if`/`then`/`else`, `while`, and `for` builders via Rust callbacks. |
| Call ops | `MPSGraphCallOps.h` | 🟡 | `Graph::call` plus `CompilationDescriptor::set_callable`. |
| Loss ops | `MPSGraphLossOps.h` | 🟡 | `loss_reduction_type` plus `softmax_cross_entropy`; other losses and gradient helpers remain deferred. |
| Matrix multiplication | `MPSGraphMatrixMultiplicationOps.h` | 🟡 | `matrix_multiplication` only (1 of 4 methods); scaled dot-product attention and `HammingDistance` remain deferred. |
| Linear algebra / matrix inverse | `MPSGraphLinearAlgebraOps.h`, `MPSGraphMatrixInverseOps.h` | 🟡 | `band_part` and `matrix_inverse`; broader solve, triangular, and decomposition APIs remain deferred. |
| Memory ops | `MPSGraphMemoryOps.h` | 🟡 | Existing constants plus variable creation, `read_variable`, and `assign_variable`. Higher-level memory convenience APIs remain deferred. |
| Non-zero / sort / one-hot / resize / sample-grid / stencil / sparse / quantization / NMS | multiple dedicated headers in newer SDKs | 🟡 | Wrapped via `src/specialized.rs`; remaining overload families are still deferred. Quantization has only the scalar scale/zero-point `quantize` and `dequantize` (2 of 10 methods); the tensor-scale, per-axis and lookup-table variants are missing. `non_maximum_suppression` builds, but running it aborts on the Apple-silicon GPU runtime used for testing. |
| Optimizer ops | `MPSGraphOptimizerOps.h` | 🟡 | `stochastic_gradient_descent`; Adam/RMSProp/update variants remain deferred. |
| RNN ops | `MPSGraphRNNOps.h` | 🟡 | `SingleGateRNNDescriptor`, `LSTMDescriptor`, `GRUDescriptor`, and forward `singleGateRNN`/`LSTM`/`GRU` helpers. Gradient variants remain deferred. |

## Validation

Shape and dimension lists are overflow-checked before they reach the bridge. These builders also
check axes and static operand shapes, returning `None` instead of letting MPSGraph abort the
process: binary arithmetic, matrix multiplication, reductions, softmax and its gradient, reshape,
transpose, slice, broadcast, and the split family. `Graph::run`, `compile`,
`compile_with_descriptor` and executable runs check feeds, inputs and preallocated results
against the placeholders and compiled feed types.

Not validated: the specialized, gather, scatter, random, RNN, control-flow, concat/stack/pad and
top-K builders, and any tensor whose shape has dynamic dimensions. Invalid shapes there still reach
MPSGraph, which aborts the process.

## Naming notes from the audit

- “Gradient ops” map to `MPSGraphAutomaticDifferentiation.h` in the SDK.
- “GatherND ops” are part of `MPSGraphGatherOps.h`, not a separate header.
- “FFT ops” map to `MPSGraphFourierTransformOps.h`.
- Average-pooling APIs live under `MPSGraphPoolingOps.h`; there is no separate `MPSGraphAvgPoolOps.h` header in this SDK.
- Slice APIs live under `MPSGraphTensorShapeOps.h`; there is no separate `MPSGraphSliceOps.h` header in this SDK.

## Remaining broader-SDK gaps

1. Caller-supplied Swift blocks (`scheduledHandler`, custom `completionHandler`s, compilation callbacks, dispatch queues). Asynchronous executable runs use a completion handler internally to back `AsyncRun`.
2. Safe upstream wrappers for `MTLSharedEvent` / `MPSCommandBuffer`; the audited shared-event surface is currently exposed through raw-handle helpers.
3. The large remaining families in activation, arithmetic, shape, pooling, normalization, optimizer, and domain-specific ops, plus gradient-heavy variants of the newly added RNN/call/control-flow surfaces.
