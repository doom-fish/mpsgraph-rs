# mpsgraph-rs coverage audit (vs MacOSX26.2.sdk)

SDK_PUBLIC_SYMBOLS: 90
VERIFIED: 33
GAPS: 57
EXEMPT: 0
COVERAGE_PCT: 36.67%

Scope: top-level Objective-C interfaces/categories and enum types from `MetalPerformanceShadersGraph.framework/Headers`, filtered for macOS availability. This framework exposes no top-level `FOUNDATION_EXPORT`, `extern const`, or free C function declarations in the audited headers.

## 🟢 VERIFIED
| Symbol | Kind | Header | Wrapped by |
| --- | --- | --- | --- |
| `MPSGraphOptions` | enum | `MPSGraph.h` | `graph_options` module; `Graph::options` / `set_options`; `Executable::options` / `set_options`. |
| `MPSGraphOptimization` | enum | `MPSGraph.h` | `optimization` module; `CompilationDescriptor::optimization_level` / `set_optimization_level`. |
| `MPSGraphOptimizationProfile` | enum | `MPSGraph.h` | `optimization_profile` module; `CompilationDescriptor::optimization_profile` / `set_optimization_profile`. |
| `MPSGraphReducedPrecisionFastMath` | enum | `MPSGraph.h` | `reduced_precision_fast_math` module; `CompilationDescriptor::reduced_precision_fast_math` / `set_reduced_precision_fast_math`. |
| `MPSGraphCompilationDescriptor` | interface | `MPSGraph.h` | `CompilationDescriptor` in `src/execution.rs` (disable-type-inference + optimization/profile/wait/reduced-precision knobs). |
| `MPSGraphExecutionDescriptor` | interface | `MPSGraph.h` | `ExecutionDescriptor` in `src/execution.rs` (wait flag plus compilation-descriptor accessors). |
| `MPSGraph` | interface | `MPSGraph.h` | `Graph` in `src/graph.rs` (compile/run/placeholder introspection). |
| `MPSGraph(MPSGraphActivationOps)` | category | `MPSGraphActivationOps.h` | `Graph::{relu,sigmoid,softmax}` in `src/graph.rs` plus `Graph::{leaky_relu,leaky_relu_tensor}` in `src/ops.rs`. |
| `MPSGraph(MPSGraphArithmeticOps)` | category | `MPSGraphArithmeticOps.h` | `Graph::{addition,subtraction,multiplication,division}` in `src/graph.rs` plus `Graph::{unary_arithmetic,binary_arithmetic,select}` in `src/ops.rs`. |
| `MPSGraph(MPSGraphGradientOps)` | category | `MPSGraphAutomaticDifferentiation.h` | `Graph::{relu_gradient,sigmoid_gradient,softmax_gradient,leaky_relu_gradient}` in `src/ops.rs`. |
| `MPSGraphConvolution2DOpDescriptor` | interface | `MPSGraphConvolutionOps.h` | `Convolution2DDescriptor` / `Convolution2DDescriptorInfo` in `src/graph.rs`. |
| `MPSGraph(MPSGraphConvolutionOps)` | category | `MPSGraphConvolutionOps.h` | `Graph::convolution2d` in `src/graph.rs`. |
| `MPSGraphShapedType` | interface | `MPSGraphCore.h` | `ShapedType` in `src/types.rs`. |
| `MPSGraphTensorNamedDataLayout` | enum | `MPSGraphCore.h` | `tensor_named_data_layout` module plus convolution/pooling descriptor config in `src/graph.rs`. |
| `MPSGraphPaddingStyle` | enum | `MPSGraphCore.h` | `padding_style` module plus convolution/pooling descriptor config in `src/graph.rs`. |
| `MPSGraphPaddingMode` | enum | `MPSGraphCore.h` | `Graph::pad(padding_mode: isize, ...)` in `src/ops.rs` forwards raw `MPSGraphPaddingMode` values. |
| `MPSGraphDeviceType` | enum | `MPSGraphDevice.h` | `graph_device_type` module; `GraphDevice::device_type`; `TensorData::graph_device_type`. |
| `MPSGraphDevice` | interface | `MPSGraphDevice.h` | `GraphDevice` in `src/types.rs`. |
| `MPSGraphExecutableExecutionDescriptor` | interface | `MPSGraphExecutable.h` | `ExecutableExecutionDescriptor` in `src/execution.rs`. |
| `MPSGraphDeploymentPlatform` | enum | `MPSGraphExecutable.h` | `deployment_platform` module; `ExecutableSerializationDescriptor::deployment_platform` / `set_deployment_platform`. |
| `MPSGraphExecutableSerializationDescriptor` | interface | `MPSGraphExecutable.h` | `ExecutableSerializationDescriptor` in `src/execution.rs`. |
| `MPSGraphExecutable` | interface | `MPSGraphExecutable.h` | `Executable` in `src/graph.rs` plus specialization/output-type/package APIs in `src/execution.rs`. |
| `MPSGraph(MPSGraphMatrixMultiplicationOps)` | category | `MPSGraphMatrixMultiplicationOps.h` | `Graph::matrix_multiplication` in `src/graph.rs`. |
| `MPSGraph(MemoryOps)` | category | `MPSGraphMemoryOps.h` | `Graph::{placeholder,constant_bytes,constant_f32_slice,constant_scalar,constant_scalar_shaped}` in `src/graph.rs`. |
| `MPSGraph(MPSGraphNormalizationOps)` | category | `MPSGraphNormalizationOps.h` | `Graph::normalize` in `src/graph.rs`. |
| `MPSGraphOperation` | interface | `MPSGraphOperation.h` | `Operation` in `src/types.rs`; `Tensor::operation`. |
| `MPSGraphPooling2DOpDescriptor` | interface | `MPSGraphPoolingOps.h` | `Pooling2DDescriptor` / `Pooling2DDescriptorInfo` in `src/graph.rs`. |
| `MPSGraph(MPSGraphPoolingOps)` | category | `MPSGraphPoolingOps.h` | `Graph::max_pooling2d` in `src/graph.rs`. |
| `MPSGraph(MPSGraphReductionOps)` | category | `MPSGraphReductionOps.h` | `Graph::{reduction_sum,reduction_maximum,reduction_minimum,mean}` in `src/graph.rs` plus `Graph::{reduce_axis,reduce_axes}` in `src/ops.rs`. |
| `MPSGraphTensor` | interface | `MPSGraphTensor.h` | `Tensor` in `src/graph.rs` plus `shape` / `data_type` / `operation` accessors in `src/types.rs`. |
| `MPSGraphTensorData` | interface | `MPSGraphTensorData.h` | `TensorData` in `src/data.rs`. |
| `MPSGraph(MPSGraphTensorShapeOps)` | category | `MPSGraphTensorShapeOps.h` | `Graph::{reshape,transpose,slice,broadcast}` in `src/graph.rs` plus concat/split/stack/pad in `src/ops.rs`. |
| `MPSGraph(MPSGraphTopKOps)` | category | `MPSGraphTopKOps.h` | `Graph::{top_k,top_k_tensor}` in `src/ops.rs`. |

## 🔴 GAPS
| Symbol | Kind | Header | Notes |
| --- | --- | --- | --- |
| `MPSGraphExecutionStage` | enum | `MPSGraph.h` | Shared-event signaling/waiting APIs are not wrapped, so the execution-stage enum is unreachable. |
| `MPSGraph(CallOp)` | category | `MPSGraphCallOps.h` | Callable / symbol-dispatch graph APIs are not wrapped. |
| `MPSGraph(MPSGraphControlFlowOps)` | category | `MPSGraphControlFlowOps.h` | Graph control-flow builders (`if` / `while` / `for`) are not wrapped. |
| `MPSGraphConvolution3DOpDescriptor` | interface | `MPSGraphConvolutionOps.h` | Only the 2D convolution descriptor/path is wrapped. |
| `MPSGraph(MPSGraphConvolutionTransposeOps)` | category | `MPSGraphConvolutionTransposeOps.h` | Transpose-convolution APIs are not wrapped. |
| `MPSGraphObject` | interface | `MPSGraphCore.h` | The common Objective-C base class is not exposed as a standalone Rust type. |
| `MPSGraphType` | interface | `MPSGraphCore.h` | The crate exposes `ShapedType`, but not the generic `MPSGraphType` base class. |
| `MPSGraphReductionMode` | enum | `MPSGraphCore.h` | The crate exposes concrete reduction helpers instead of the generic reduction-mode enum. |
| `MPSGraph(MPSGraphCumulativeOps)` | category | `MPSGraphCumulativeOps.h` | Cumulative-sum / cumulative-product style ops are not wrapped. |
| `MPSGraphDepthwiseConvolution2DOpDescriptor` | interface | `MPSGraphDepthwiseConvolutionOps.h` | Depthwise-convolution APIs are not wrapped. |
| `MPSGraphDepthwiseConvolution3DOpDescriptor` | interface | `MPSGraphDepthwiseConvolutionOps.h` | Depthwise-convolution APIs are not wrapped. |
| `MPSGraph(MPSGraphDepthwiseConvolutionOps)` | category | `MPSGraphDepthwiseConvolutionOps.h` | Depthwise-convolution APIs are not wrapped. |
| `MPSGraphFFTScalingMode` | enum | `MPSGraphFourierTransformOps.h` | FFT APIs are not wrapped. |
| `MPSGraphFFTDescriptor` | interface | `MPSGraphFourierTransformOps.h` | FFT APIs are not wrapped. |
| `MPSGraph(MPSGraphFourierTransformOps)` | category | `MPSGraphFourierTransformOps.h` | FFT APIs are not wrapped. |
| `MPSGraph(GatherNDOps)` | category | `MPSGraphGatherOps.h` | Gather / GatherND APIs are not wrapped. |
| `MPSGraph(GatherOps)` | category | `MPSGraphGatherOps.h` | Gather / GatherND APIs are not wrapped. |
| `MPSGraph(MPSGraphGatherAlongAxisOps)` | category | `MPSGraphGatherOps.h` | Gather / GatherND APIs are not wrapped. |
| `MPSGraphImToColOpDescriptor` | interface | `MPSGraphImToColOps.h` | Im2Col/Col2Im APIs are not wrapped. |
| `MPSGraph(MPSGraphImToColOps)` | category | `MPSGraphImToColOps.h` | Im2Col / Col2Im APIs are not wrapped. |
| `MPSGraph(MPSGraphLinearAlgebraOps)` | category | `MPSGraphLinearAlgebraOps.h` | Linear-algebra helpers beyond plain matmul are not wrapped. |
| `MPSGraphLossReductionType` | enum | `MPSGraphLossOps.h` | Loss APIs are not wrapped. |
| `MPSGraph(MPSGraphLossOps)` | category | `MPSGraphLossOps.h` | Loss APIs are not wrapped. |
| `MPSGraph(MPSGraphMatrixInverseOps)` | category | `MPSGraphMatrixInverseOps.h` | Matrix-inverse / solve APIs are not wrapped. |
| `MPSGraphVariableOp` | interface | `MPSGraphMemoryOps.h` | Variable/read/assign memory ops are not exposed. |
| `MPSGraphNonMaximumSuppressionCoordinateMode` | enum | `MPSGraphNonMaximumSuppressionOps.h` | Non-maximum-suppression APIs are not wrapped. |
| `MPSGraph(MPSGraphNonMaximumSuppressionOps)` | category | `MPSGraphNonMaximumSuppressionOps.h` | Non-maximum-suppression APIs are not wrapped. |
| `MPSGraph(NonZeroOps)` | category | `MPSGraphNonZeroOps.h` | NonZero APIs are not wrapped. |
| `MPSGraph(MPSGraphOneHotOps)` | category | `MPSGraphOneHotOps.h` | One-hot APIs are not wrapped. |
| `MPSGraph(MPSGraphOptimizerOps)` | category | `MPSGraphOptimizerOps.h` | Optimizer update APIs (SGD/Adam/etc.) are not wrapped. |
| `MPSGraphPoolingReturnIndicesMode` | enum | `MPSGraphPoolingOps.h` | Only plain max-pooling is wrapped; return-indices pooling modes are not surfaced. |
| `MPSGraphPooling4DOpDescriptor` | interface | `MPSGraphPoolingOps.h` | Only the 2D pooling descriptor/path is wrapped. |
| `MPSGraph(MPSGraphQuantizationOps)` | category | `MPSGraphQuantizationOps.h` | Quantize/dequantize APIs are not wrapped. |
| `MPSGraphRNNActivation` | enum | `MPSGraphRNNOps.h` | RNN/LSTM/GRU APIs are not wrapped. |
| `MPSGraphSingleGateRNNDescriptor` | interface | `MPSGraphRNNOps.h` | RNN/LSTM/GRU APIs are not wrapped. |
| `MPSGraphLSTMDescriptor` | interface | `MPSGraphRNNOps.h` | RNN/LSTM/GRU APIs are not wrapped. |
| `MPSGraphGRUDescriptor` | interface | `MPSGraphRNNOps.h` | RNN/LSTM/GRU APIs are not wrapped. |
| `MPSGraph(MPSGraphRNNOps)` | category | `MPSGraphRNNOps.h` | RNN/LSTM/GRU APIs are not wrapped. |
| `MPSGraphRandomDistribution` | enum | `MPSGraphRandomOps.h` | Random/dropout APIs are not wrapped. |
| `MPSGraphRandomNormalSamplingMethod` | enum | `MPSGraphRandomOps.h` | Random/dropout APIs are not wrapped. |
| `MPSGraphRandomOpDescriptor` | interface | `MPSGraphRandomOps.h` | Random/dropout APIs are not wrapped. |
| `MPSGraph(MPSGraphRandomOps)` | category | `MPSGraphRandomOps.h` | Random/dropout APIs are not wrapped. |
| `MPSGraphResizeMode` | enum | `MPSGraphResizeOps.h` | Resize APIs are not wrapped. |
| `MPSGraphResizeNearestRoundingMode` | enum | `MPSGraphResizeOps.h` | Resize-nearest APIs are not wrapped. |
| `MPSGraph(MPSGraphResizeOps)` | category | `MPSGraphResizeOps.h` | Resize APIs are not wrapped. |
| `MPSGraph(MPSGraphSampleGrid)` | category | `MPSGraphSampleGridOps.h` | Sample-grid APIs are not wrapped. |
| `MPSGraphScatterMode` | enum | `MPSGraphScatterNDOps.h` | Scatter/ScatterND APIs are not wrapped. |
| `MPSGraph(ScatterNDOps)` | category | `MPSGraphScatterNDOps.h` | Scatter / ScatterND APIs are not wrapped. |
| `MPSGraph(MPSGraphScatterOps)` | category | `MPSGraphScatterNDOps.h` | Scatter / ScatterND APIs are not wrapped. |
| `MPSGraph(MPSGraphScatterAlongAxisOps)` | category | `MPSGraphScatterNDOps.h` | Scatter / ScatterND APIs are not wrapped. |
| `MPSGraph(MPSGraphSortOps)` | category | `MPSGraphSortOps.h` | Sort / argsort APIs are not wrapped. |
| `MPSGraphSparseStorageType` | enum | `MPSGraphSparseOps.h` | Sparse-tensor APIs are not wrapped. |
| `MPSGraphCreateSparseOpDescriptor` | interface | `MPSGraphSparseOps.h` | Sparse-tensor APIs are not wrapped. |
| `MPSGraph(MPSGraphSparseOps)` | category | `MPSGraphSparseOps.h` | Sparse-tensor APIs are not wrapped. |
| `MPSGraphStencilOpDescriptor` | interface | `MPSGraphStencilOps.h` | Stencil APIs are not wrapped. |
| `MPSGraph(MPSGraphStencilOps)` | category | `MPSGraphStencilOps.h` | Stencil APIs are not wrapped. |
| `MPSGraph(MPSGraphTopKGradientOps)` | category | `MPSGraphTopKOps.h` | Only `topK` and `topK(kTensor:)` are wrapped; axis/bottomK/topKGradient variants are missing. |

## ⏭️ EXEMPT
| Symbol | Kind | Header | Reason | SDK attribute |
| --- | --- | --- | --- | --- |
| _(none)_ | - | - | No deprecated top-level macOS symbols were part of the 90-symbol audited surface. | - |

