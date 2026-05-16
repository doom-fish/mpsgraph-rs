# Coverage

Audited against the macOS SDK `MetalPerformanceShadersGraph.framework/Headers` surface and Swift symbol graph extraction.

Legend:
- ✅ implemented in this crate
- 🟡 partially implemented
- ⏭️ intentionally deferred / currently unsupported

## Header and category status

| Requested area | SDK/header mapping | Status | Notes |
| --- | --- | --- | --- |
| Core graph construction | `MPSGraph.h`, `MPSGraphCore.h`, `MPSGraphTensor.h` | 🟡 | `Graph`, `Tensor`, placeholders, constants, direct run, compile, tensor metadata (`shape`, `data_type`, `operation`), graph options, placeholder introspection. |
| Tensor data | `MPSGraphTensorData.h` | 🟡 | CPU bytes / `f32` slices / `MTLBuffer`, readback helpers, `graph_device`; matrix/vector/NDArray/image/MTLTensor initializers remain deferred. |
| Device and types | `MPSGraphDevice.h` | ✅ | `GraphDevice`, `ShapedType`, graph device creation from `MTLDevice`, shape/data-type mutation and inspection. |
| Executable / descriptors | `MPSGraphExecutable.h` | 🟡 | `CompilationDescriptor`, `ExecutionDescriptor`, `ExecutableExecutionDescriptor`, `ExecutableSerializationDescriptor`, executable feed/target/output-type queries, package load/save. Shared-event and `MPSCommandBuffer`-specific paths are deferred pending upstream Metal wrapper support. |
| Activation ops | `MPSGraphActivationOps.h` | 🟡 | `reLU`, `sigmoid`, `softMax`, `leakyReLU`, `reLUGradient`, `sigmoidGradient`, `softMaxGradient`; broader activation families remain deferred. |
| Arithmetic ops | `MPSGraphArithmeticOps.h` | 🟡 | Unary arithmetic enum covers identity, exponent/log, square/sqrt/reciprocal, abs/neg/sign, rounding, trig/hyperbolic, `isNaN`, `isInfinite`; binary arithmetic enum covers add/sub/mul/div, `divisionNoNaN`, `power`, min/max, comparisons, logical ops, `atan2`, `floorModulo`, plus `select`. Other arithmetic entry points remain deferred. |
| Tensor shape ops | `MPSGraphTensorShapeOps.h` | 🟡 | Existing reshape / transpose / slice / broadcast plus new `concat`, `split`, `stack`, and `pad`. Many advanced slice, gather-like, and indexing variants remain deferred. |
| Reduction ops | `MPSGraphReductionOps.h` | 🟡 | Existing sum/max/min/mean plus axis/axes sum/max/min/product helpers. Arg reductions and the rest of the reduction family remain deferred. |
| Top-K ops | `MPSGraphTopKOps.h` | 🟡 | `topK` via scalar `k` and tensor `kTensor`; axis-specific/macOS-14-only variants remain deferred. |
| Automatic differentiation | `MPSGraphAutomaticDifferentiation.h` | 🟡 | Activation-gradient helpers only. General gradient/JVP/VJP families remain deferred. |
| Pooling ops | `MPSGraphPoolingOps.h` | 🟡 | Existing max-pooling helper only. Average/L2/global/adaptive pooling remain deferred. |
| Convolution ops | convolution APIs on `MPSGraph.h` / related descriptors | 🟡 | Existing 2D convolution helper only. Transpose/depthwise/fused families remain deferred. |
| Normalization ops | normalization APIs on `MPSGraph.h` | 🟡 | Existing normalize helper only. Broader normalization families remain deferred. |
| FFT ops | `MPSGraphFourierTransformOps.h` | ⏭️ | Not yet wrapped. |
| Random ops | `MPSGraphRandomOps.h` | ⏭️ | Not yet wrapped. |
| Gather ops | `MPSGraphGatherOps.h` (includes GatherND) | ⏭️ | Not yet wrapped. |
| Scatter ops | `MPSGraphScatterOps.h` | ⏭️ | Not yet wrapped. |
| Control flow | `MPSGraphControlFlowOps.h` | ⏭️ | Not yet wrapped. |
| Call ops | `MPSGraphCallOps.h` | ⏭️ | Not yet wrapped. |
| Loss ops | `MPSGraphLossOps.h` | ⏭️ | Not yet wrapped. |
| Non-zero / sort / one-hot / resize / sample-grid / stencil / sparse / quantization / RNN / NMS | multiple dedicated headers in newer SDKs | ⏭️ | Not yet wrapped in `v0.2.0`. |

## Naming notes from the audit

- “Gradient ops” map to `MPSGraphAutomaticDifferentiation.h` in the SDK.
- “GatherND ops” are part of `MPSGraphGatherOps.h`, not a separate header.
- “FFT ops” map to `MPSGraphFourierTransformOps.h`.
- Average-pooling APIs live under `MPSGraphPoolingOps.h`; there is no separate `MPSGraphAvgPoolOps.h` header in this SDK.
- Slice APIs live under `MPSGraphTensorShapeOps.h`; there is no separate `MPSGraphSliceOps.h` header in this SDK.

## Main gaps for future waves

1. Higher-order APIs that rely on Swift blocks/callbacks (`completionHandler`, `scheduledHandler`, compilation callbacks, dispatch queues).
2. Execution paths that require wrappers not currently exposed by `apple-metal` (`MTLSharedEvent`, `MPSCommandBuffer`).
3. The large remaining families in arithmetic, shape, gather/scatter, random, FFT, control-flow, and domain-specific ops.
