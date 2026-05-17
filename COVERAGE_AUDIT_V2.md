# mpsgraph-rs coverage audit v2 (vs MacOSX26.2.sdk)

SDK_PUBLIC_SYMBOLS: 90
VERIFIED: 90
GAPS: 0
EXEMPT: 0
COVERAGE_PCT: 100.00

Scope: All 90 Objective-C interfaces/categories and enum types from `MetalPerformanceShadersGraph.framework/Headers` (macOS 26.2), enumerated via @interface/@protocol patterns and typedef enum/NS_ENUM/NS_OPTIONS. All macOS-available top-level symbols are wrapped in v0.2.3 and later.

## 🟢 VERIFIED
| Symbol | Kind | Header | Wrapped by |
| --- | --- | --- | --- |
| `MPSGraphOptions` | enum | `MPSGraph.h` | graph_options module; Graph::options / set_options; Executable::options / set_options |
| `MPSGraphOptimization` | enum | `MPSGraph.h` | optimization module; CompilationDescriptor::optimization_level / set_optimization_level |
| `MPSGraphOptimizationProfile` | enum | `MPSGraph.h` | optimization_profile module; CompilationDescriptor::optimization_profile / set_optimization_profile |
| `MPSGraphExecutionStage` | enum | `MPSGraph.h` | execution_stage module plus raw shared-event wait/signal helpers on ExecutionDescriptor and ExecutableExecutionDescriptor in src/specialized.rs |
| `MPSGraphReducedPrecisionFastMath` | enum | `MPSGraph.h` | CompilationDescriptor::reduced_precision_fast_math / set_reduced_precision_fast_math |
| `MPSGraphTensorNamedDataLayout` | enum | `MPSGraphCore.h` | tensor_named_data_layout module plus convolution/pooling descriptor config |
| `MPSGraphPaddingStyle` | enum | `MPSGraphCore.h` | padding_style module plus convolution/pooling descriptor config |
| `MPSGraphPaddingMode` | enum | `MPSGraphCore.h` | Graph::pad(padding_mode: isize, ...) in src/ops.rs |
| `MPSGraphReductionMode` | enum | `MPSGraphCore.h` | reduction_mode module plus StencilDescriptorInfo::reduction_mode |
| `MPSGraphDeviceType` | enum | `MPSGraphDevice.h` | graph_device_type module; GraphDevice::device_type; TensorData::graph_device_type |
| `MPSGraphDeploymentPlatform` | enum | `MPSGraphExecutable.h` | deployment_platform module; ExecutableSerializationDescriptor::deployment_platform / set_deployment_platform |
| `MPSGraphFFTScalingMode` | enum | `MPSGraphFourierTransformOps.h` | fft_scaling_mode module plus FftDescriptorInfo::scaling_mode |
| `MPSGraphLossReductionType` | enum | `MPSGraphLossOps.h` | loss_reduction_type module plus Graph::softmax_cross_entropy |
| `MPSGraphPoolingReturnIndicesMode` | enum | `MPSGraphPoolingOps.h` | pooling_return_indices_mode module plus Pooling4DDescriptorInfo::return_indices_mode |
| `MPSGraphRandomDistribution` | enum | `MPSGraphRandomOps.h` | random_distribution module plus RandomOpDescriptor::distribution / set_distribution |
| `MPSGraphRandomNormalSamplingMethod` | enum | `MPSGraphRandomOps.h` | random_normal_sampling_method module plus RandomOpDescriptor::sampling_method / set_sampling_method |
| `MPSGraphRNNActivation` | enum | `MPSGraphRNNOps.h` | rnn_activation module plus RNN descriptor activation accessors |
| `MPSGraphResizeMode` | enum | `MPSGraphResizeOps.h` | resize_mode module plus Graph::{resize, sample_grid} |
| `MPSGraphResizeNearestRoundingMode` | enum | `MPSGraphResizeOps.h` | resize_nearest_rounding_mode module plus Graph::resize_nearest |
| `MPSGraphNonMaximumSuppressionCoordinateMode` | enum | `MPSGraphNonMaximumSuppressionOps.h` | non_maximum_suppression_coordinate_mode module plus Graph::non_maximum_suppression |
| `MPSGraphScatterMode` | enum | `MPSGraphScatterNDOps.h` | scatter_mode module plus Graph::{scatter_nd, scatter, scatter_along_axis} |
| `MPSGraphSparseStorageType` | enum | `MPSGraphSparseOps.h` | sparse_storage_type module plus CreateSparseDescriptor::new |
| `MPSGraph` | interface | `MPSGraph.h` | Graph in src/graph.rs (compile/run/placeholder introspection) |
| `MPSGraphCompilationDescriptor` | interface | `MPSGraph.h` | CompilationDescriptor in src/execution.rs |
| `MPSGraphExecutionDescriptor` | interface | `MPSGraph.h` | ExecutionDescriptor in src/execution.rs |
| `MPSGraphObject` | interface | `MPSGraphCore.h` | Object in src/specialized.rs |
| `MPSGraphType` | interface | `MPSGraphCore.h` | GraphType in src/specialized.rs plus ShapedType::as_graph_type |
| `MPSGraphShapedType` | interface | `MPSGraphCore.h` | ShapedType in src/types.rs |
| `MPSGraphOperation` | interface | `MPSGraphOperation.h` | Operation in src/types.rs; Tensor::operation |
| `MPSGraphTensor` | interface | `MPSGraphTensor.h` | Tensor in src/graph.rs plus shape / data_type / operation accessors |
| `MPSGraphTensorData` | interface | `MPSGraphTensorData.h` | TensorData in src/data.rs |
| `MPSGraphVariableOp` | interface | `MPSGraphMemoryOps.h` | VariableOp in src/specialized.rs plus Operation::as_variable |
| `MPSGraphDevice` | interface | `MPSGraphDevice.h` | GraphDevice in src/types.rs |
| `MPSGraphExecutable` | interface | `MPSGraphExecutable.h` | Executable in src/graph.rs plus specialization/output-type/package APIs |
| `MPSGraphExecutableExecutionDescriptor` | interface | `MPSGraphExecutable.h` | ExecutableExecutionDescriptor in src/execution.rs |
| `MPSGraphExecutableSerializationDescriptor` | interface | `MPSGraphExecutable.h` | ExecutableSerializationDescriptor in src/execution.rs |
| `MPSGraphConvolution2DOpDescriptor` | interface | `MPSGraphConvolutionOps.h` | Convolution2DDescriptor / Convolution2DDescriptorInfo |
| `MPSGraphConvolution3DOpDescriptor` | interface | `MPSGraphConvolutionOps.h` | Convolution3DDescriptor / Convolution3DDescriptorInfo in src/specialized.rs |
| `MPSGraphDepthwiseConvolution2DOpDescriptor` | interface | `MPSGraphDepthwiseConvolutionOps.h` | DepthwiseConvolution2DDescriptor / DepthwiseConvolution2DDescriptorInfo |
| `MPSGraphDepthwiseConvolution3DOpDescriptor` | interface | `MPSGraphDepthwiseConvolutionOps.h` | DepthwiseConvolution3DDescriptor / DepthwiseConvolution3DDescriptorInfo |
| `MPSGraphFFTDescriptor` | interface | `MPSGraphFourierTransformOps.h` | FftDescriptor / FftDescriptorInfo in src/specialized.rs |
| `MPSGraphImToColOpDescriptor` | interface | `MPSGraphImToColOps.h` | ImToColDescriptor / ImToColDescriptorInfo in src/specialized.rs |
| `MPSGraphPooling2DOpDescriptor` | interface | `MPSGraphPoolingOps.h` | Pooling2DDescriptor / Pooling2DDescriptorInfo |
| `MPSGraphPooling4DOpDescriptor` | interface | `MPSGraphPoolingOps.h` | Pooling4DDescriptor / Pooling4DDescriptorInfo in src/specialized.rs |
| `MPSGraphRandomOpDescriptor` | interface | `MPSGraphRandomOps.h` | RandomOpDescriptor in src/random.rs |
| `MPSGraphSingleGateRNNDescriptor` | interface | `MPSGraphRNNOps.h` | SingleGateRNNDescriptor in src/rnn.rs |
| `MPSGraphLSTMDescriptor` | interface | `MPSGraphRNNOps.h` | LSTMDescriptor in src/rnn.rs |
| `MPSGraphGRUDescriptor` | interface | `MPSGraphRNNOps.h` | GRUDescriptor in src/rnn.rs |
| `MPSGraphStencilOpDescriptor` | interface | `MPSGraphStencilOps.h` | StencilDescriptor / StencilDescriptorInfo in src/specialized.rs |
| `MPSGraphCreateSparseOpDescriptor` | interface | `MPSGraphSparseOps.h` | CreateSparseDescriptor in src/specialized.rs |
| `MPSGraph(MPSGraphActivationOps)` | category | `MPSGraphActivationOps.h` | Graph::{relu, sigmoid, softmax} in src/graph.rs plus leaky_relu variants |
| `MPSGraph(MPSGraphArithmeticOps)` | category | `MPSGraphArithmeticOps.h` | Graph::{addition, subtraction, multiplication, division} plus unary/binary |
| `MPSGraph(MPSGraphGradientOps)` | category | `MPSGraphAutomaticDifferentiation.h` | Graph::{relu_gradient, sigmoid_gradient, softmax_gradient, leaky_relu_gradient} |
| `MPSGraph(CallOp)` | category | `MPSGraphCallOps.h` | Graph::call in src/call.rs plus CompilationDescriptor::set_callable |
| `MPSGraph(MPSGraphControlFlowOps)` | category | `MPSGraphControlFlowOps.h` | Graph::{control_dependency, if_then, if_then_else, while_loop, for_loop} |
| `MPSGraph(MPSGraphConvolutionOps)` | category | `MPSGraphConvolutionOps.h` | Graph::convolution2d in src/graph.rs |
| `MPSGraph(MPSGraphConvolutionTransposeOps)` | category | `MPSGraphConvolutionTransposeOps.h` | Graph::convolution_transpose2d |
| `MPSGraph(MPSGraphCumulativeOps)` | category | `MPSGraphCumulativeOps.h` | Graph::cumulative_sum in src/specialized.rs |
| `MPSGraph(MPSGraphDepthwiseConvolutionOps)` | category | `MPSGraphDepthwiseConvolutionOps.h` | Graph::{depthwise_convolution2d, depthwise_convolution3d} |
| `MPSGraph(MPSGraphFourierTransformOps)` | category | `MPSGraphFourierTransformOps.h` | Graph::fast_fourier_transform |
| `MPSGraph(GatherNDOps)` | category | `MPSGraphGatherOps.h` | Graph::gather_nd |
| `MPSGraph(GatherOps)` | category | `MPSGraphGatherOps.h` | Graph::gather |
| `MPSGraph(MPSGraphGatherAlongAxisOps)` | category | `MPSGraphGatherOps.h` | Graph::{gather_along_axis, gather_along_axis_tensor} |
| `MPSGraph(MPSGraphImToColOps)` | category | `MPSGraphImToColOps.h` | Graph::im_to_col |
| `MPSGraph(MPSGraphLinearAlgebraOps)` | category | `MPSGraphLinearAlgebraOps.h` | Graph::band_part |
| `MPSGraph(MPSGraphLossOps)` | category | `MPSGraphLossOps.h` | Graph::softmax_cross_entropy |
| `MPSGraph(MPSGraphMatrixInverseOps)` | category | `MPSGraphMatrixInverseOps.h` | Graph::matrix_inverse |
| `MPSGraph(MPSGraphMatrixMultiplicationOps)` | category | `MPSGraphMatrixMultiplicationOps.h` | Graph::matrix_multiplication |
| `MPSGraph(MemoryOps)` | category | `MPSGraphMemoryOps.h` | Graph::{placeholder, constant_*, read_variable, assign_variable} |
| `MPSGraph(MPSGraphNonMaximumSuppressionOps)` | category | `MPSGraphNonMaximumSuppressionOps.h` | Graph::non_maximum_suppression |
| `MPSGraph(NonZeroOps)` | category | `MPSGraphNonZeroOps.h` | Graph::non_zero_indices |
| `MPSGraph(MPSGraphNormalizationOps)` | category | `MPSGraphNormalizationOps.h` | Graph::normalize |
| `MPSGraph(MPSGraphOneHotOps)` | category | `MPSGraphOneHotOps.h` | Graph::one_hot |
| `MPSGraph(MPSGraphOptimizerOps)` | category | `MPSGraphOptimizerOps.h` | Graph::stochastic_gradient_descent |
| `MPSGraph(MPSGraphPoolingOps)` | category | `MPSGraphPoolingOps.h` | Graph::max_pooling2d |
| `MPSGraph(MPSGraphQuantizationOps)` | category | `MPSGraphQuantizationOps.h` | Graph::{quantize, dequantize} |
| `MPSGraph(MPSGraphRandomOps)` | category | `MPSGraphRandomOps.h` | Graph::{random_tensor, dropout, random_philox_state_*} |
| `MPSGraph(MPSGraphReductionOps)` | category | `MPSGraphReductionOps.h` | Graph::{reduction_sum, reduction_maximum, reduce_axis*} |
| `MPSGraph(MPSGraphResizeOps)` | category | `MPSGraphResizeOps.h` | Graph::{resize, resize_nearest} |
| `MPSGraph(MPSGraphRNNOps)` | category | `MPSGraphRNNOps.h` | Graph::{single_gate_rnn, lstm, gru} |
| `MPSGraph(MPSGraphSampleGrid)` | category | `MPSGraphSampleGridOps.h` | Graph::sample_grid |
| `MPSGraph(ScatterNDOps)` | category | `MPSGraphScatterNDOps.h` | Graph::scatter_nd |
| `MPSGraph(MPSGraphScatterOps)` | category | `MPSGraphScatterNDOps.h` | Graph::scatter |
| `MPSGraph(MPSGraphScatterAlongAxisOps)` | category | `MPSGraphScatterNDOps.h` | Graph::scatter_along_axis |
| `MPSGraph(MPSGraphSortOps)` | category | `MPSGraphSortOps.h` | Graph::{sort, arg_sort} |
| `MPSGraph(MPSGraphStencilOps)` | category | `MPSGraphStencilOps.h` | Graph::stencil |
| `MPSGraph(MPSGraphTensorShapeOps)` | category | `MPSGraphTensorShapeOps.h` | Graph::{reshape, transpose, slice, broadcast, concat/split/stack/pad} |
| `MPSGraph(MPSGraphTopKOps)` | category | `MPSGraphTopKOps.h` | Graph::{top_k, top_k_tensor} |
| `MPSGraph(MPSGraphTopKGradientOps)` | category | `MPSGraphTopKOps.h` | Graph::top_k_gradient |
| `MPSGraph (MPSGraphSparseOps)` | category | `MPSGraphSparseOps.h` | Graph::sparse_tensor_with_descriptor |

## 🔴 GAPS
| Symbol | Kind | Header | Notes |
| --- | --- | --- | --- |
| _(none)_ | - | - | All 90 audited macOS-visible top-level symbols are wrapped in v0.2.3. |

## ⏭️ EXEMPT
| Symbol | Kind | Header | Reason | SDK attribute |
| --- | --- | --- | --- | --- |
| _(none)_ | - | - | No unavailable or deprecated top-level macOS symbols were part of the audited surface. | - |