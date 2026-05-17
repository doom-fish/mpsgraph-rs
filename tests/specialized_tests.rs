#![allow(clippy::too_many_lines)]

use apple_mpsgraph::{
    data_type, execution_stage, fft_scaling_mode, loss_reduction_type,
    non_maximum_suppression_coordinate_mode, padding_mode, pooling_return_indices_mode,
    reduction_mode, resize_mode, resize_nearest_rounding_mode, scatter_mode,
    sparse_storage_type, tensor_named_data_layout, Convolution2DDescriptor,
    Convolution2DDescriptorInfo, Convolution3DDescriptor, Convolution3DDescriptorInfo,
    CreateSparseDescriptor, DepthwiseConvolution2DDescriptor,
    DepthwiseConvolution2DDescriptorInfo, DepthwiseConvolution3DDescriptor,
    DepthwiseConvolution3DDescriptorInfo, FftDescriptor, FftDescriptorInfo, Graph,
    ImToColDescriptor, ImToColDescriptorInfo, Pooling4DDescriptor,
    Pooling4DDescriptorInfo, ShapedType, StencilDescriptor, StencilDescriptorInfo,
};
use std::process::Command;
use std::sync::OnceLock;

fn macos_version() -> (u64, u64, u64) {
    static VERSION: OnceLock<(u64, u64, u64)> = OnceLock::new();
    *VERSION.get_or_init(|| {
        let output = Command::new("sw_vers")
            .arg("-productVersion")
            .output()
            .expect("sw_vers product version");
        let version = String::from_utf8(output.stdout)
            .expect("utf8 product version")
            .trim()
            .to_owned();
        let mut parts = version.split('.').map(|part| part.parse::<u64>().expect("numeric version part"));
        (
            parts.next().unwrap_or(0),
            parts.next().unwrap_or(0),
            parts.next().unwrap_or(0),
        )
    })
}

fn macos_version_at_least(major: u64, minor: u64) -> bool {
    let version = macos_version();
    (version.0, version.1) >= (major, minor)
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>()
}

fn read_i32(data: &apple_mpsgraph::TensorData) -> Vec<i32> {
    let bytes = data.read_bytes().expect("read bytes");
    bytes
        .chunks_exact(core::mem::size_of::<i32>())
        .map(|chunk| i32::from_ne_bytes(chunk.try_into().expect("i32 chunk")))
        .collect()
}

fn assert_availability<T>(name: &str, value: Option<T>, available: bool) -> Option<T> {
    assert_eq!(value.is_some(), available, "{name} availability mismatch");
    value
}

#[test]
fn specialized_constants_types_and_descriptors_round_trip() {
    assert_eq!(execution_stage::COMPLETED, 0);
    assert_eq!(reduction_mode::SUM, 2);
    assert_eq!(reduction_mode::ARGUMENT_MAX, 5);
    assert_eq!(pooling_return_indices_mode::GLOBAL_FLATTEN_4D, 4);
    assert_eq!(fft_scaling_mode::UNITARY, 2);
    assert_eq!(loss_reduction_type::AXIS, 0);
    assert_eq!(loss_reduction_type::SUM, 1);
    assert_eq!(loss_reduction_type::MEAN, 2);
    assert_eq!(non_maximum_suppression_coordinate_mode::CENTERS_WIDTH_FIRST, 3);
    assert_eq!(resize_mode::BILINEAR, 1);
    assert_eq!(resize_nearest_rounding_mode::ROUND_TO_ODD, 5);
    assert_eq!(scatter_mode::SET, 6);
    assert_eq!(sparse_storage_type::CSR, 2);

    let graph = Graph::new().expect("graph");
    let variable_tensor = graph
        .variable_f32_slice(&[1.0, 2.0], &[2], Some("variable"))
        .expect("variable tensor");
    let variable = variable_tensor
        .operation()
        .and_then(|operation| operation.as_variable())
        .expect("variable op");
    assert_eq!(variable.shape(), vec![2]);
    assert_eq!(variable.data_type(), data_type::FLOAT32);
    let _variable_object = variable.as_object();
    let _variable_operation = variable.as_operation();

    let shaped = ShapedType::new(Some(&[2]), data_type::FLOAT32).expect("shaped type");
    let graph_type = shaped.as_graph_type();
    let _graph_object = graph_type.as_object();

    assert_eq!(
        Convolution3DDescriptor::new(Convolution3DDescriptorInfo {
            data_layout: tensor_named_data_layout::NDHWC,
            weights_layout: tensor_named_data_layout::DHWIO,
            ..Default::default()
        })
        .is_some(),
        macos_version_at_least(13, 2)
    );
    assert!(DepthwiseConvolution2DDescriptor::new(DepthwiseConvolution2DDescriptorInfo::default()).is_some());
    assert_eq!(
        DepthwiseConvolution3DDescriptor::new(DepthwiseConvolution3DDescriptorInfo::default()).is_some(),
        macos_version_at_least(12, 0)
    );
    assert_eq!(
        FftDescriptor::new(FftDescriptorInfo {
            scaling_mode: fft_scaling_mode::UNITARY,
            ..Default::default()
        })
        .is_some(),
        macos_version_at_least(14, 0)
    );
    assert_eq!(
        ImToColDescriptor::new(ImToColDescriptorInfo::default()).is_some(),
        macos_version_at_least(14, 0)
    );
    assert_eq!(
        Pooling4DDescriptor::new(Pooling4DDescriptorInfo::default()).is_some(),
        macos_version_at_least(12, 0)
    );
    assert_eq!(
        Pooling4DDescriptor::new(Pooling4DDescriptorInfo {
            return_indices_mode: pooling_return_indices_mode::GLOBAL_FLATTEN_4D,
            ..Default::default()
        })
        .is_some(),
        macos_version_at_least(12, 2)
    );
    assert_eq!(
        CreateSparseDescriptor::new(sparse_storage_type::COO, data_type::FLOAT32).is_some(),
        macos_version_at_least(12, 0)
    );
    assert_eq!(
        StencilDescriptor::new(StencilDescriptorInfo {
            reduction_mode: reduction_mode::ARGUMENT_MAX,
            ..Default::default()
        })
        .is_some(),
        macos_version_at_least(12, 0)
    );
}

#[test]
fn specialized_ops_smoke_build_all_categories() {
    let graph = Graph::new().expect("graph");

    let vector = graph
        .constant_f32_slice(&[3.0, 1.0, 2.0], &[3])
        .expect("vector");
    let matrix = graph.constant_f32_slice(&[4.0], &[1, 1]).expect("matrix");
    let source4d = graph.constant_f32_slice(&[1.0], &[1, 1, 1, 1]).expect("source4d");
    let weights4d = graph.constant_f32_slice(&[1.0], &[1, 1, 1, 1]).expect("weights4d");
    let source5d = graph
        .constant_f32_slice(&[1.0], &[1, 1, 1, 1, 1])
        .expect("source5d");
    let weights5d = graph
        .constant_f32_slice(&[1.0], &[1, 1, 1, 1, 1])
        .expect("weights5d");
    let logits = graph
        .constant_f32_slice(&[0.25, 0.75], &[1, 2])
        .expect("logits");
    let labels = graph
        .constant_f32_slice(&[0.0, 1.0], &[1, 2])
        .expect("labels");
    let learning_rate = graph
        .constant_f32_slice(&[0.1], &[1])
        .expect("learning rate");
    let updates = graph.constant_f32_slice(&[10.0, 20.0], &[2]).expect("updates");
    let indices_1d = graph
        .constant_bytes(&i32_bytes(&[0, 2]), &[2], data_type::INT32)
        .expect("indices 1d");
    let scatter_nd_indices = graph
        .constant_bytes(&i32_bytes(&[0, 1]), &[2, 1], data_type::INT32)
        .expect("indices nd");
    let size_tensor = graph
        .constant_bytes(&i32_bytes(&[1, 1]), &[2], data_type::INT32)
        .expect("size tensor");
    let coordinate_tensor = graph
        .constant_f32_slice(&[0.0, 0.0], &[1, 1, 1, 2])
        .expect("coordinate tensor");
    let boxes = graph
        .constant_f32_slice(&[0.0, 0.0, 1.0, 1.0], &[1, 1, 4])
        .expect("boxes");
    let scores = graph
        .constant_f32_slice(&[0.9], &[1, 1, 1])
        .expect("scores");
    let sparse_values = graph
        .constant_f32_slice(&[1.0, 2.0], &[2])
        .expect("sparse values");
    let sparse_index0 = graph
        .constant_bytes(&i32_bytes(&[0, 1]), &[2], data_type::INT32)
        .expect("sparse index0");
    let sparse_index1 = graph
        .constant_bytes(&i32_bytes(&[1, 0]), &[2], data_type::INT32)
        .expect("sparse index1");

    let conv2d_descriptor = Convolution2DDescriptor::new(Convolution2DDescriptorInfo::default())
        .expect("conv2d descriptor");
    let conv3d_descriptor = assert_availability(
        "convolution3d descriptor",
        Convolution3DDescriptor::new(Convolution3DDescriptorInfo::default()),
        macos_version_at_least(13, 2),
    );
    let depthwise2d_descriptor = DepthwiseConvolution2DDescriptor::new(
        DepthwiseConvolution2DDescriptorInfo::default(),
    )
    .expect("depthwise2d descriptor");
    let depthwise3d_descriptor = assert_availability(
        "depthwise3d descriptor",
        DepthwiseConvolution3DDescriptor::new(DepthwiseConvolution3DDescriptorInfo::default()),
        macos_version_at_least(12, 0),
    );
    let fft_descriptor = assert_availability(
        "fft descriptor",
        FftDescriptor::new(FftDescriptorInfo::default()),
        macos_version_at_least(14, 0),
    );
    let im_to_col_descriptor = assert_availability(
        "im2col descriptor",
        ImToColDescriptor::new(ImToColDescriptorInfo::default()),
        macos_version_at_least(14, 0),
    );
    let pooling_descriptor = assert_availability(
        "pooling4d descriptor",
        Pooling4DDescriptor::new(Pooling4DDescriptorInfo::default()),
        macos_version_at_least(12, 0),
    );
    let pooling_indices_descriptor = assert_availability(
        "pooling4d indices descriptor",
        Pooling4DDescriptor::new(Pooling4DDescriptorInfo {
            return_indices_mode: pooling_return_indices_mode::GLOBAL_FLATTEN_4D,
            ..Default::default()
        }),
        macos_version_at_least(12, 2),
    );
    let sparse_descriptor = assert_availability(
        "sparse descriptor",
        CreateSparseDescriptor::new(sparse_storage_type::COO, data_type::FLOAT32),
        macos_version_at_least(12, 0),
    );
    let stencil_descriptor = assert_availability(
        "stencil descriptor",
        StencilDescriptor::new(StencilDescriptorInfo::default()),
        macos_version_at_least(12, 0),
    );

    assert!(graph
        .convolution_transpose2d(
            &source4d,
            &weights4d,
            &[1, 1, 1, 1],
            &conv2d_descriptor,
            Some("conv_transpose2d"),
        )
        .is_some());
    assert_availability(
        "cumulative sum",
        graph.cumulative_sum(&vector, 0, false, false, Some("cumulative_sum")),
        macos_version_at_least(13, 0),
    );
    assert!(graph
        .depthwise_convolution2d(
            &source4d,
            &weights4d,
            &depthwise2d_descriptor,
            Some("depthwise2d"),
        )
        .is_some());
    assert!(graph.band_part(&matrix, 0, 0, Some("band_part")).is_some());
    assert!(graph
        .softmax_cross_entropy(
            &logits,
            &labels,
            1,
            loss_reduction_type::SUM,
            Some("softmax_cross_entropy"),
        )
        .is_some());
    assert_availability(
        "matrix inverse",
        graph.matrix_inverse(&matrix, Some("inverse")),
        macos_version_at_least(13, 0),
    );

    let variable = graph
        .variable_f32_slice(&[5.0, 7.0], &[2], Some("variable"))
        .expect("variable");
    assert!(graph.read_variable(&variable, Some("read_variable")).is_some());
    assert!(graph.assign_variable(&variable, &updates, Some("assign_variable")).is_some());
    assert!(graph.one_hot(&indices_1d, 3, data_type::FLOAT32, Some("one_hot")).is_some());
    assert!(graph
        .stochastic_gradient_descent(&learning_rate, &vector, &vector, Some("sgd"))
        .is_some());

    let quantized = assert_availability(
        "quantize",
        graph.quantize(&vector, 1.0, 0.0, data_type::INT8, Some("quantize")),
        macos_version_at_least(13, 1),
    );
    if let Some(tensor) = quantized.as_ref() {
        assert!(graph
            .dequantize(tensor, 1.0, 0.0, data_type::FLOAT32, Some("dequantize"))
            .is_some());
    }

    assert_availability(
        "resize",
        graph.resize(
            &source4d,
            &[1, 1],
            resize_mode::BILINEAR,
            true,
            false,
            tensor_named_data_layout::NHWC,
            Some("resize"),
        ),
        macos_version_at_least(13, 0),
    );
    assert_availability(
        "resize nearest",
        graph.resize_nearest(
            &source4d,
            &size_tensor,
            resize_nearest_rounding_mode::ROUND_PREFER_CEIL,
            true,
            false,
            tensor_named_data_layout::NHWC,
            Some("resize_nearest"),
        ),
        macos_version_at_least(13, 0),
    );
    assert_availability(
        "sample grid",
        graph.sample_grid(
            &source4d,
            &coordinate_tensor,
            tensor_named_data_layout::NHWC,
            true,
            false,
            false,
            padding_mode::ZERO,
            resize_mode::NEAREST,
            0.0,
            Some("sample_grid"),
        ),
        macos_version_at_least(13, 1),
    );

    assert_availability(
        "scatter nd",
        graph.scatter_nd(
            &updates,
            &scatter_nd_indices,
            &[2],
            0,
            scatter_mode::ADD,
            Some("scatter_nd"),
        ),
        macos_version_at_least(12, 0),
    );
    assert_availability(
        "scatter",
        graph.scatter(
            &updates,
            &indices_1d,
            &[3],
            0,
            scatter_mode::SET,
            Some("scatter"),
        ),
        macos_version_at_least(12, 0),
    );
    assert_availability(
        "scatter along axis",
        graph.scatter_along_axis(
            0,
            &updates,
            &indices_1d,
            &[3],
            scatter_mode::ADD,
            Some("scatter_along_axis"),
        ),
        macos_version_at_least(12, 3),
    );

    assert_availability(
        "sort",
        graph.sort(&vector, 0, false, Some("sort")),
        macos_version_at_least(13, 0),
    );
    assert_availability(
        "arg sort",
        graph.arg_sort(&vector, 0, false, Some("arg_sort")),
        macos_version_at_least(13, 0),
    );
    assert_availability(
        "non maximum suppression",
        graph.non_maximum_suppression(
            &boxes,
            &scores,
            0.5,
            0.1,
            false,
            non_maximum_suppression_coordinate_mode::CORNERS_HEIGHT_FIRST,
            Some("nms"),
        ),
        macos_version_at_least(14, 0),
    );
    assert_availability(
        "non zero indices",
        graph.non_zero_indices(&vector, Some("non_zero_indices")),
        macos_version_at_least(14, 0),
    );

    if let Some(descriptor) = conv3d_descriptor.as_ref() {
        assert!(graph
            .convolution3d(&source5d, &weights5d, descriptor, Some("convolution3d"))
            .is_some());
    }
    if let Some(descriptor) = depthwise3d_descriptor.as_ref() {
        assert!(graph
            .depthwise_convolution3d(&source5d, &weights4d, descriptor, Some("depthwise3d"))
            .is_some());
    }
    if let Some(descriptor) = fft_descriptor.as_ref() {
        assert!(graph
            .fast_fourier_transform(&vector, &[0], descriptor, Some("fft"))
            .is_some());
    }
    if let Some(descriptor) = im_to_col_descriptor.as_ref() {
        assert!(graph.im_to_col(&source4d, descriptor, Some("im_to_col")).is_some());
    }
    if let Some(descriptor) = pooling_descriptor.as_ref() {
        assert!(graph.max_pooling4d(&source4d, descriptor, Some("max_pooling4d")).is_some());
    }
    if let Some(descriptor) = pooling_indices_descriptor.as_ref() {
        assert!(graph
            .max_pooling4d_return_indices(&source4d, descriptor, Some("max_pooling4d_indices"))
            .is_some());
    }
    if let Some(descriptor) = sparse_descriptor.as_ref() {
        assert!(graph
            .sparse_tensor_with_descriptor(
                descriptor,
                &[&sparse_values, &sparse_index0, &sparse_index1],
                &[2, 2],
                Some("sparse_tensor"),
            )
            .is_some());
    }
    if let Some(descriptor) = stencil_descriptor.as_ref() {
        assert!(graph
            .stencil(&source4d, &weights4d, descriptor, Some("stencil"))
            .is_some());
    }
    assert_availability(
        "topk gradient",
        graph.top_k_gradient(&vector, &vector, 2, Some("topk_gradient")),
        macos_version_at_least(14, 0),
    );
}

#[test]
fn specialized_ops_execute_selected_results() {
    if !macos_version_at_least(13, 1) {
        return;
    }

    let graph = Graph::new().expect("graph");
    let values = graph
        .constant_f32_slice(&[3.0, 1.0, 2.0], &[3])
        .expect("values");
    let indices = graph
        .constant_bytes(&i32_bytes(&[0, 2]), &[2], data_type::INT32)
        .expect("indices");
    let matrix = graph.constant_f32_slice(&[4.0], &[1, 1]).expect("matrix");
    let variable = graph
        .variable_f32_slice(&[5.0, 7.0], &[2], Some("variable"))
        .expect("variable");

    let cumulative = graph
        .cumulative_sum(&values, 0, false, false, Some("cumulative"))
        .expect("cumulative sum");
    let sorted = graph.sort(&values, 0, false, Some("sorted")).expect("sort");
    let arg_sorted = graph
        .arg_sort(&values, 0, false, Some("arg_sorted"))
        .expect("arg sort");
    let one_hot = graph
        .one_hot(&indices, 3, data_type::FLOAT32, Some("one_hot"))
        .expect("one hot");
    let inverse = graph.matrix_inverse(&matrix, Some("inverse")).expect("inverse");
    let read_variable = graph
        .read_variable(&variable, Some("read_variable"))
        .expect("read variable");
    let quantized = graph
        .quantize(&values, 1.0, 0.0, data_type::INT8, Some("quantized"))
        .expect("quantize");
    let dequantized = graph
        .dequantize(&quantized, 1.0, 0.0, data_type::FLOAT32, Some("dequantized"))
        .expect("dequantize");

    let results = graph
        .run(
            &[],
            &[
                &cumulative,
                &sorted,
                &arg_sorted,
                &one_hot,
                &inverse,
                &read_variable,
                &dequantized,
            ],
        )
        .expect("run graph");

    assert_eq!(results[0].read_f32().expect("cumulative"), vec![3.0, 4.0, 6.0]);
    assert_eq!(results[1].read_f32().expect("sorted"), vec![1.0, 2.0, 3.0]);
    assert_eq!(read_i32(&results[2]), vec![1, 2, 0]);
    assert_eq!(results[3].read_f32().expect("one hot"), vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    assert_eq!(results[4].read_f32().expect("inverse"), vec![0.25]);
    assert_eq!(results[5].read_f32().expect("read variable"), vec![5.0, 7.0]);
    assert_eq!(results[6].read_f32().expect("dequantized"), vec![3.0, 1.0, 2.0]);
}
