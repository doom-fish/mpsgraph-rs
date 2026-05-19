# mpsgraph-rs coverage audit (vs MacOSX26.2.sdk)

SDK_PUBLIC_SYMBOLS: 90
VERIFIED: 90
GAPS: 0
EXEMPT: 0
COVERAGE_PCT: 100.00%

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
| `MPSGraph(CallOp)` | category | `MPSGraphCallOps.h` | `Graph::call` in `src/call.rs` plus `CompilationDescriptor::set_callable` in `src/execution.rs`. |
| `MPSGraph(MPSGraphControlFlowOps)` | category | `MPSGraphControlFlowOps.h` | `Graph::{control_dependency,if_then,if_then_else,while_loop,for_loop,for_loop_iterations}` in `src/control_flow.rs`. |
| `MPSGraph(GatherNDOps)` | category | `MPSGraphGatherOps.h` | `Graph::gather_nd` in `src/gather.rs`. |
| `MPSGraph(GatherOps)` | category | `MPSGraphGatherOps.h` | `Graph::gather` in `src/gather.rs`. |
| `MPSGraph(MPSGraphGatherAlongAxisOps)` | category | `MPSGraphGatherOps.h` | `Graph::{gather_along_axis,gather_along_axis_tensor}` in `src/gather.rs`. |
| `MPSGraph(MPSGraphMatrixMultiplicationOps)` | category | `MPSGraphMatrixMultiplicationOps.h` | `Graph::matrix_multiplication` in `src/graph.rs`. |
| `MPSGraph(MemoryOps)` | category | `MPSGraphMemoryOps.h` | `Graph::{placeholder,constant_bytes,constant_f32_slice,constant_scalar,constant_scalar_shaped}` in `src/graph.rs`. |
| `MPSGraph(MPSGraphNormalizationOps)` | category | `MPSGraphNormalizationOps.h` | `Graph::normalize` in `src/graph.rs`. |
| `MPSGraphOperation` | interface | `MPSGraphOperation.h` | `Operation` in `src/types.rs`; `Tensor::operation`. |
| `MPSGraphPooling2DOpDescriptor` | interface | `MPSGraphPoolingOps.h` | `Pooling2DDescriptor` / `Pooling2DDescriptorInfo` in `src/graph.rs`. |
| `MPSGraph(MPSGraphPoolingOps)` | category | `MPSGraphPoolingOps.h` | `Graph::max_pooling2d` in `src/graph.rs`. |
| `MPSGraph(MPSGraphReductionOps)` | category | `MPSGraphReductionOps.h` | `Graph::{reduction_sum,reduction_maximum,reduction_minimum,mean}` in `src/graph.rs` plus `Graph::{reduce_axis,reduce_axes}` in `src/ops.rs`. |
| `MPSGraphRandomDistribution` | enum | `MPSGraphRandomOps.h` | `random_distribution` module plus `RandomOpDescriptor::distribution` / `set_distribution` in `src/random.rs`. |
| `MPSGraphRandomNormalSamplingMethod` | enum | `MPSGraphRandomOps.h` | `random_normal_sampling_method` module plus `RandomOpDescriptor::sampling_method` / `set_sampling_method` in `src/random.rs`. |
| `MPSGraphRandomOpDescriptor` | interface | `MPSGraphRandomOps.h` | `RandomOpDescriptor` in `src/random.rs`. |
| `MPSGraph(MPSGraphRandomOps)` | category | `MPSGraphRandomOps.h` | `Graph::{random_philox_state_seed,random_philox_state_counter,random_tensor,random_tensor_shape_tensor,random_tensor_seed,random_tensor_state,dropout,dropout_tensor}` in `src/random.rs`. |
| `MPSGraphRNNActivation` | enum | `MPSGraphRNNOps.h` | `rnn_activation` module plus RNN descriptor activation accessors in `src/rnn.rs`. |
| `MPSGraphSingleGateRNNDescriptor` | interface | `MPSGraphRNNOps.h` | `SingleGateRNNDescriptor` in `src/rnn.rs`. |
| `MPSGraphLSTMDescriptor` | interface | `MPSGraphRNNOps.h` | `LSTMDescriptor` in `src/rnn.rs`. |
| `MPSGraphGRUDescriptor` | interface | `MPSGraphRNNOps.h` | `GRUDescriptor` in `src/rnn.rs`. |
| `MPSGraph(MPSGraphRNNOps)` | category | `MPSGraphRNNOps.h` | `Graph::{single_gate_rnn,lstm,gru}` in `src/rnn.rs`. |
| `MPSGraphTensor` | interface | `MPSGraphTensor.h` | `Tensor` in `src/graph.rs` plus `shape` / `data_type` / `operation` accessors in `src/types.rs`. |
| `MPSGraphTensorData` | interface | `MPSGraphTensorData.h` | `TensorData` in `src/data.rs`, including `from_bytes`, `from_buffer`, and `from_tensor`. |
| `MPSGraph(MPSGraphTensorShapeOps)` | category | `MPSGraphTensorShapeOps.h` | `Graph::{reshape,transpose,slice,broadcast}` in `src/graph.rs` plus concat/split/stack/pad in `src/ops.rs`. |
| `MPSGraph(MPSGraphTopKOps)` | category | `MPSGraphTopKOps.h` | `Graph::{top_k,top_k_tensor}` in `src/ops.rs`. |

## 🟢 VERIFIED IN v0.2.3
| Symbol | Kind | Header | Wrapped by |
| --- | --- | --- | --- |
| `MPSGraphExecutionStage` | enum | `MPSGraph.h` | `execution_stage` module plus raw shared-event wait/signal helpers on `ExecutionDescriptor` and `ExecutableExecutionDescriptor` in `src/specialized.rs`. |
| `MPSGraphConvolution3DOpDescriptor` | interface | `MPSGraphConvolutionOps.h` | `Convolution3DDescriptor` / `Convolution3DDescriptorInfo` in `src/specialized.rs`. |
| `MPSGraph(MPSGraphConvolutionTransposeOps)` | category | `MPSGraphConvolutionTransposeOps.h` | `Graph::convolution_transpose2d` in `src/specialized.rs`. |
| `MPSGraphObject` | interface | `MPSGraphCore.h` | `Object` in `src/specialized.rs`. |
| `MPSGraphType` | interface | `MPSGraphCore.h` | `GraphType` in `src/specialized.rs` plus `ShapedType::as_graph_type`. |
| `MPSGraphReductionMode` | enum | `MPSGraphCore.h` | `reduction_mode` module plus `StencilDescriptorInfo::reduction_mode` in `src/specialized.rs`. |
| `MPSGraph(MPSGraphCumulativeOps)` | category | `MPSGraphCumulativeOps.h` | `Graph::cumulative_sum` in `src/specialized.rs`. |
| `MPSGraphDepthwiseConvolution2DOpDescriptor` | interface | `MPSGraphDepthwiseConvolutionOps.h` | `DepthwiseConvolution2DDescriptor` / `DepthwiseConvolution2DDescriptorInfo` in `src/specialized.rs`. |
| `MPSGraphDepthwiseConvolution3DOpDescriptor` | interface | `MPSGraphDepthwiseConvolutionOps.h` | `DepthwiseConvolution3DDescriptor` / `DepthwiseConvolution3DDescriptorInfo` in `src/specialized.rs`. |
| `MPSGraph(MPSGraphDepthwiseConvolutionOps)` | category | `MPSGraphDepthwiseConvolutionOps.h` | `Graph::{depthwise_convolution2d,depthwise_convolution3d}` in `src/specialized.rs`. |
| `MPSGraphFFTScalingMode` | enum | `MPSGraphFourierTransformOps.h` | `fft_scaling_mode` module plus `FftDescriptorInfo::scaling_mode` in `src/specialized.rs`. |
| `MPSGraphFFTDescriptor` | interface | `MPSGraphFourierTransformOps.h` | `FftDescriptor` / `FftDescriptorInfo` in `src/specialized.rs`. |
| `MPSGraph(MPSGraphFourierTransformOps)` | category | `MPSGraphFourierTransformOps.h` | `Graph::fast_fourier_transform` in `src/specialized.rs`. |
| `MPSGraphImToColOpDescriptor` | interface | `MPSGraphImToColOps.h` | `ImToColDescriptor` / `ImToColDescriptorInfo` in `src/specialized.rs`. |
| `MPSGraph(MPSGraphImToColOps)` | category | `MPSGraphImToColOps.h` | `Graph::im_to_col` in `src/specialized.rs`. |
| `MPSGraph(MPSGraphLinearAlgebraOps)` | category | `MPSGraphLinearAlgebraOps.h` | `Graph::band_part` in `src/specialized.rs`. |
| `MPSGraphLossReductionType` | enum | `MPSGraphLossOps.h` | `loss_reduction_type` module plus `Graph::softmax_cross_entropy` in `src/specialized.rs`. |
| `MPSGraph(MPSGraphLossOps)` | category | `MPSGraphLossOps.h` | `Graph::softmax_cross_entropy` in `src/specialized.rs`. |
| `MPSGraph(MPSGraphMatrixInverseOps)` | category | `MPSGraphMatrixInverseOps.h` | `Graph::matrix_inverse` in `src/specialized.rs`. |
| `MPSGraphVariableOp` | interface | `MPSGraphMemoryOps.h` | `VariableOp` in `src/specialized.rs` plus `Operation::as_variable` and `Graph::{variable_bytes,variable_f32_slice,read_variable,assign_variable}`. |
| `MPSGraphNonMaximumSuppressionCoordinateMode` | enum | `MPSGraphNonMaximumSuppressionOps.h` | `non_maximum_suppression_coordinate_mode` module plus `Graph::non_maximum_suppression` in `src/specialized.rs`. |
| `MPSGraph(MPSGraphNonMaximumSuppressionOps)` | category | `MPSGraphNonMaximumSuppressionOps.h` | `Graph::non_maximum_suppression` in `src/specialized.rs`. |
| `MPSGraph(NonZeroOps)` | category | `MPSGraphNonZeroOps.h` | `Graph::non_zero_indices` in `src/specialized.rs`. |
| `MPSGraph(MPSGraphOneHotOps)` | category | `MPSGraphOneHotOps.h` | `Graph::one_hot` in `src/specialized.rs`. |
| `MPSGraph(MPSGraphOptimizerOps)` | category | `MPSGraphOptimizerOps.h` | `Graph::stochastic_gradient_descent` in `src/specialized.rs`. |
| `MPSGraphPoolingReturnIndicesMode` | enum | `MPSGraphPoolingOps.h` | `pooling_return_indices_mode` module plus `Pooling4DDescriptorInfo::return_indices_mode` in `src/specialized.rs`. |
| `MPSGraphPooling4DOpDescriptor` | interface | `MPSGraphPoolingOps.h` | `Pooling4DDescriptor` / `Pooling4DDescriptorInfo` in `src/specialized.rs`. |
| `MPSGraph(MPSGraphQuantizationOps)` | category | `MPSGraphQuantizationOps.h` | `Graph::{quantize,dequantize}` in `src/specialized.rs`. |
| `MPSGraphResizeMode` | enum | `MPSGraphResizeOps.h` | `resize_mode` module plus `Graph::{resize,sample_grid}` in `src/specialized.rs`. |
| `MPSGraphResizeNearestRoundingMode` | enum | `MPSGraphResizeOps.h` | `resize_nearest_rounding_mode` module plus `Graph::resize_nearest` in `src/specialized.rs`. |
| `MPSGraph(MPSGraphResizeOps)` | category | `MPSGraphResizeOps.h` | `Graph::{resize,resize_nearest}` in `src/specialized.rs`. |
| `MPSGraph(MPSGraphSampleGrid)` | category | `MPSGraphSampleGridOps.h` | `Graph::sample_grid` in `src/specialized.rs`. |
| `MPSGraphScatterMode` | enum | `MPSGraphScatterNDOps.h` | `scatter_mode` module plus `Graph::{scatter_nd,scatter,scatter_along_axis}` in `src/specialized.rs`. |
| `MPSGraph(ScatterNDOps)` | category | `MPSGraphScatterNDOps.h` | `Graph::scatter_nd` in `src/specialized.rs`. |
| `MPSGraph(MPSGraphScatterOps)` | category | `MPSGraphScatterNDOps.h` | `Graph::scatter` in `src/specialized.rs`. |
| `MPSGraph(MPSGraphScatterAlongAxisOps)` | category | `MPSGraphScatterNDOps.h` | `Graph::scatter_along_axis` in `src/specialized.rs`. |
| `MPSGraph(MPSGraphSortOps)` | category | `MPSGraphSortOps.h` | `Graph::{sort,arg_sort}` in `src/specialized.rs`. |
| `MPSGraphSparseStorageType` | enum | `MPSGraphSparseOps.h` | `sparse_storage_type` module plus `CreateSparseDescriptor::new` in `src/specialized.rs`. |
| `MPSGraphCreateSparseOpDescriptor` | interface | `MPSGraphSparseOps.h` | `CreateSparseDescriptor` in `src/specialized.rs`. |
| `MPSGraph(MPSGraphSparseOps)` | category | `MPSGraphSparseOps.h` | `Graph::sparse_tensor_with_descriptor` in `src/specialized.rs`. |
| `MPSGraphStencilOpDescriptor` | interface | `MPSGraphStencilOps.h` | `StencilDescriptor` / `StencilDescriptorInfo` in `src/specialized.rs`. |
| `MPSGraph(MPSGraphStencilOps)` | category | `MPSGraphStencilOps.h` | `Graph::stencil` in `src/specialized.rs`. |
| `MPSGraph(MPSGraphTopKGradientOps)` | category | `MPSGraphTopKOps.h` | `Graph::top_k_gradient` in `src/specialized.rs`. |

## 🔴 GAPS
| Symbol | Kind | Header | Notes |
| --- | --- | --- | --- |
| _(none)_ | - | - | All 90 audited macOS-visible top-level symbols are wrapped in `v0.2.3`. |

## ⏭️ EXEMPT
| Symbol | Kind | Header | Reason | SDK attribute |
| --- | --- | --- | --- | --- |
| _(none)_ | - | - | No deprecated top-level macOS symbols were part of the 90-symbol audited surface. | - |

