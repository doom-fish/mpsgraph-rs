#![allow(clippy::too_many_lines)]

use apple_metal::MetalDevice;
use apple_mpsgraph::{
    data_type, fft_scaling_mode, padding_mode, padding_style, reduction_mode, resize_mode,
    resize_nearest_rounding_mode, sparse_storage_type, tensor_named_data_layout,
    CompilationDescriptor, Convolution3DDescriptor, Convolution3DDescriptorInfo,
    CreateSparseDescriptor, DepthwiseConvolution2DDescriptor, DepthwiseConvolution2DDescriptorInfo,
    DepthwiseConvolution3DDescriptor, DepthwiseConvolution3DDescriptorInfo, Error, Executable,
    ExecutableExecutionDescriptor, ExecutableSerializationDescriptor, ExecutionDescriptor, Feed,
    FftDescriptor, FftDescriptorInfo, GRUDescriptor, Graph, ImToColDescriptor,
    ImToColDescriptorInfo, LSTMDescriptor, Pooling2DDescriptor, Pooling2DDescriptorInfo,
    Pooling4DDescriptor, Pooling4DDescriptorInfo, RandomOpDescriptor, ShapedType,
    SingleGateRNNDescriptor, StencilDescriptor, StencilDescriptorInfo, Tensor, TensorData,
};

fn device() -> MetalDevice {
    MetalDevice::system_default().expect("no Metal device available")
}

fn placeholder(graph: &Graph, shape: &[usize], data: u32) -> Tensor {
    graph
        .placeholder(Some(shape), data, None)
        .expect("placeholder")
}

fn floats(graph: &Graph, shape: &[usize]) -> Tensor {
    placeholder(graph, shape, data_type::FLOAT32)
}

fn still_runs(graph: &Graph) {
    let device = device();
    let input = floats(graph, &[2]);
    let doubled = graph.addition(&input, &input, None).expect("addition");
    let data = TensorData::from_f32_slice(&device, &[1.0, 2.0], &[2]).expect("data");
    let results = graph
        .run(&[Feed::new(&input, &data)], &[&doubled])
        .expect("the graph still runs after refused builders");
    assert_eq!(results[0].read_f32().expect("read"), vec![2.0, 4.0]);
}

macro_rules! refused {
    ($result:expr, $pattern:pat) => {
        match $result {
            Err($pattern) => {}
            Err(other) => panic!("wrong error {other:?} from {}", stringify!($result)),
            Ok(_) => panic!("{} was accepted", stringify!($result)),
        }
    };
}

trait AmbiguousIfSync<A> {
    fn assert_not_sync() {}
}

impl<T: ?Sized> AmbiguousIfSync<()> for T {}
impl<T: ?Sized + Sync> AmbiguousIfSync<u8> for T {}

const fn assert_send<T: Send>() {}

#[test]
fn mutable_handles_are_send_but_not_sync() {
    assert_send::<Graph>();
    assert_send::<Executable>();
    assert_send::<ShapedType>();
    assert_send::<RandomOpDescriptor>();
    assert_send::<CompilationDescriptor>();
    <Graph as AmbiguousIfSync<_>>::assert_not_sync();
    <Executable as AmbiguousIfSync<_>>::assert_not_sync();
    <ShapedType as AmbiguousIfSync<_>>::assert_not_sync();
    <RandomOpDescriptor as AmbiguousIfSync<_>>::assert_not_sync();
    <SingleGateRNNDescriptor as AmbiguousIfSync<_>>::assert_not_sync();
    <LSTMDescriptor as AmbiguousIfSync<_>>::assert_not_sync();
    <GRUDescriptor as AmbiguousIfSync<_>>::assert_not_sync();
    <CompilationDescriptor as AmbiguousIfSync<_>>::assert_not_sync();
    <ExecutionDescriptor as AmbiguousIfSync<_>>::assert_not_sync();
    <ExecutableExecutionDescriptor as AmbiguousIfSync<_>>::assert_not_sync();
    <ExecutableSerializationDescriptor as AmbiguousIfSync<_>>::assert_not_sync();
}

#[test]
fn convolution_family_descriptors_and_operands_are_checked() {
    let graph = Graph::new().expect("graph");
    refused!(
        Convolution3DDescriptor::new(Convolution3DDescriptorInfo {
            stride_in_z: 0,
            ..Default::default()
        }),
        Error::InvalidArgument(_)
    );
    refused!(
        Convolution3DDescriptor::new(Convolution3DDescriptorInfo {
            data_layout: tensor_named_data_layout::NHWC,
            ..Default::default()
        }),
        Error::InvalidArgument(_)
    );
    let same3d = Convolution3DDescriptor::new(Convolution3DDescriptorInfo {
        padding_style: padding_style::TF_SAME,
        ..Default::default()
    })
    .expect("3D convolution");
    let volume = floats(&graph, &[1, 4, 4, 4, 2]);
    refused!(
        graph.convolution3d(&floats(&graph, &[1, 4, 4, 2]), &floats(&graph, &[3, 3, 3, 2, 5]), &same3d, None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.convolution3d(&volume, &floats(&graph, &[3, 3, 3, 3, 5]), &same3d, None),
        Error::InvalidShape(_)
    );
    let valid3d = Convolution3DDescriptor::new(Convolution3DDescriptorInfo {
        padding_style: padding_style::TF_VALID,
        ..Default::default()
    })
    .expect("3D convolution");
    refused!(
        graph.convolution3d(&volume, &floats(&graph, &[5, 3, 3, 2, 5]), &valid3d, None),
        Error::InvalidShape(_)
    );
    assert_eq!(
        graph
            .convolution3d(&volume, &floats(&graph, &[3, 3, 3, 2, 5]), &valid3d, None)
            .expect("3D convolution")
            .shape(),
        Some(vec![1, 2, 2, 2, 5])
    );

    refused!(
        DepthwiseConvolution2DDescriptor::new(DepthwiseConvolution2DDescriptorInfo {
            stride_in_y: 0,
            ..Default::default()
        }),
        Error::InvalidArgument(_)
    );
    let depthwise = DepthwiseConvolution2DDescriptor::new(DepthwiseConvolution2DDescriptorInfo {
        padding_style: padding_style::TF_SAME,
        ..Default::default()
    })
    .expect("depthwise descriptor");
    let image = floats(&graph, &[1, 4, 4, 2]);
    refused!(
        graph.depthwise_convolution2d(&floats(&graph, &[4, 4, 2]), &floats(&graph, &[3, 3, 1, 2]), &depthwise, None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.depthwise_convolution2d(&image, &floats(&graph, &[3, 3, 2]), &depthwise, None),
        Error::InvalidShape(_)
    );
    refused!(
        DepthwiseConvolution3DDescriptor::new(DepthwiseConvolution3DDescriptorInfo {
            strides: [1, 0, 1],
            ..Default::default()
        }),
        Error::InvalidArgument(_)
    );
    let out_of_range = DepthwiseConvolution3DDescriptor::new(DepthwiseConvolution3DDescriptorInfo {
        channel_dimension_index: 9,
        ..Default::default()
    })
    .expect("descriptor");
    refused!(
        graph.depthwise_convolution3d(&volume, &floats(&graph, &[2, 3, 3, 3]), &out_of_range, None),
        Error::InvalidShape(_)
    );
    let depthwise3d = DepthwiseConvolution3DDescriptor::new(DepthwiseConvolution3DDescriptorInfo::default())
        .expect("descriptor");
    refused!(
        graph.depthwise_convolution3d(&floats(&graph, &[4, 4, 2]), &floats(&graph, &[2, 3, 3, 3]), &depthwise3d, None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.depthwise_convolution3d(&volume, &floats(&graph, &[2, 3, 3]), &depthwise3d, None),
        Error::InvalidShape(_)
    );

    refused!(
        ImToColDescriptor::new(ImToColDescriptorInfo {
            kernel_width: 0,
            ..Default::default()
        }),
        Error::InvalidArgument(_)
    );
    refused!(
        ImToColDescriptor::new(ImToColDescriptorInfo {
            data_layout: tensor_named_data_layout::HW,
            ..Default::default()
        }),
        Error::InvalidArgument(_)
    );
    let large = ImToColDescriptor::new(ImToColDescriptorInfo {
        kernel_width: 9,
        kernel_height: 9,
        ..Default::default()
    })
    .expect("im2col descriptor");
    refused!(graph.im_to_col(&image, &large, None), Error::InvalidShape(_));
    refused!(
        graph.im_to_col(&placeholder(&graph, &[1, 4, 4, 2], data_type::FLOAT16), &large, None),
        Error::InvalidDataType(_)
    );
    refused!(graph.im_to_col(&floats(&graph, &[4, 4, 2]), &large, None), Error::InvalidShape(_));
    let padded = ImToColDescriptor::new(ImToColDescriptorInfo {
        kernel_width: 5,
        kernel_height: 5,
        padding_left: 1,
        padding_right: 1,
        padding_top: 1,
        padding_bottom: 1,
        ..Default::default()
    })
    .expect("padded im2col");
    graph.im_to_col(&image, &padded, None).expect("im2col fits with padding");

    refused!(
        Pooling2DDescriptor::new(Pooling2DDescriptorInfo::new(0, 2)),
        Error::InvalidArgument(_)
    );
    refused!(
        Pooling2DDescriptor::new(Pooling2DDescriptorInfo {
            stride_in_x: 0,
            ..Pooling2DDescriptorInfo::new(2, 2)
        }),
        Error::InvalidArgument(_)
    );
    let pooling = Pooling2DDescriptor::new(Pooling2DDescriptorInfo::new(2, 2)).expect("pooling");
    refused!(graph.max_pooling2d(&floats(&graph, &[4, 4]), &pooling, None), Error::InvalidShape(_));
    refused!(
        Pooling4DDescriptor::new(Pooling4DDescriptorInfo {
            kernel_sizes: [1, 1, 0, 2],
            ..Default::default()
        }),
        Error::InvalidArgument(_)
    );
    refused!(
        Pooling4DDescriptor::new(Pooling4DDescriptorInfo {
            strides: [1, 1, 0, 1],
            ..Default::default()
        }),
        Error::InvalidArgument(_)
    );
    refused!(
        Pooling4DDescriptor::new(Pooling4DDescriptorInfo {
            return_indices_mode: 99,
            ..Default::default()
        }),
        Error::InvalidArgument(_)
    );
    let pooling4d = Pooling4DDescriptor::new(Pooling4DDescriptorInfo {
        kernel_sizes: [1, 1, 2, 2],
        padding_style: padding_style::TF_VALID,
        ..Default::default()
    })
    .expect("4D pooling");
    refused!(graph.max_pooling4d(&floats(&graph, &[4, 4]), &pooling4d, None), Error::InvalidShape(_));
    assert_eq!(
        graph
            .max_pooling4d(&floats(&graph, &[1, 1, 4, 4]), &pooling4d, None)
            .expect("4D pooling")
            .shape(),
        Some(vec![1, 1, 3, 3])
    );

    refused!(
        StencilDescriptor::new(StencilDescriptorInfo {
            reduction_mode: 99,
            ..Default::default()
        }),
        Error::InvalidArgument(_)
    );
    refused!(
        StencilDescriptor::new(StencilDescriptorInfo {
            boundary_mode: padding_mode::PERIODIC,
            ..Default::default()
        }),
        Error::InvalidArgument(_)
    );
    let stencil = StencilDescriptor::new(StencilDescriptorInfo {
        reduction_mode: reduction_mode::SUM,
        ..Default::default()
    })
    .expect("stencil");
    let plane = floats(&graph, &[1, 1, 4, 4]);
    refused!(graph.stencil(&floats(&graph, &[4, 4]), &floats(&graph, &[1, 1, 2, 2]), &stencil, None), Error::InvalidShape(_));
    refused!(graph.stencil(&plane, &floats(&graph, &[2, 2]), &stencil, None), Error::InvalidShape(_));
    refused!(graph.stencil(&plane, &floats(&graph, &[1, 1, 9, 9]), &stencil, None), Error::InvalidShape(_));
    let padded_stencil = StencilDescriptor::new(StencilDescriptorInfo {
        explicit_padding: [0, 0, 0, 0, 3, 3, 3, 3],
        ..Default::default()
    })
    .expect("padded stencil");
    graph
        .stencil(&plane, &floats(&graph, &[1, 1, 9, 9]), &padded_stencil, None)
        .expect("stencil fits with padding");
    still_runs(&graph);
}

#[test]
fn transform_loss_and_linear_algebra_ops_are_checked() {
    let graph = Graph::new().expect("graph");
    let matrix = floats(&graph, &[2, 3]);
    let ints = placeholder(&graph, &[2, 3], data_type::INT32);
    refused!(
        FftDescriptor::new(FftDescriptorInfo {
            scaling_mode: 9,
            ..Default::default()
        }),
        Error::InvalidArgument(_)
    );
    let fft = FftDescriptor::new(FftDescriptorInfo {
        scaling_mode: fft_scaling_mode::SIZE,
        ..Default::default()
    })
    .expect("fft");
    refused!(graph.fast_fourier_transform(&ints, &[1], &fft, None), Error::InvalidDataType(_));
    refused!(graph.fast_fourier_transform(&matrix, &[2], &fft, None), Error::InvalidShape(_));
    let complex = placeholder(&graph, &[4, 8], data_type::COMPLEX_FLOAT32);
    graph
        .fast_fourier_transform(&complex, &[1], &fft, None)
        .expect("complex fft");

    refused!(graph.cumulative_sum(&matrix, 2, false, false, None), Error::InvalidShape(_));
    let scalar = graph.constant_scalar(1.0, data_type::FLOAT32).expect("scalar");
    refused!(graph.cumulative_sum(&scalar, 0, false, false, None), Error::InvalidShape(_));
    refused!(graph.sort(&matrix, 2, false, None), Error::InvalidShape(_));
    refused!(graph.arg_sort(&matrix, -3, false, None), Error::InvalidShape(_));
    refused!(graph.sort(&scalar, 0, false, None), Error::InvalidShape(_));
    refused!(graph.non_zero_indices(&scalar, None), Error::InvalidShape(_));

    let labels = floats(&graph, &[2, 3]);
    refused!(
        graph.softmax_cross_entropy(&matrix, &floats(&graph, &[4, 3]), 1, 0, None),
        Error::InvalidShape(_)
    );
    refused!(graph.softmax_cross_entropy(&matrix, &labels, 5, 0, None), Error::InvalidShape(_));
    refused!(
        graph.softmax_cross_entropy(&matrix, &placeholder(&graph, &[2, 3], data_type::FLOAT16), 1, 0, None),
        Error::InvalidDataType(_)
    );
    refused!(graph.softmax_cross_entropy(&matrix, &labels, 1, 99, None), Error::InvalidArgument(_));
    refused!(graph.matrix_inverse(&floats(&graph, &[3]), None), Error::InvalidShape(_));
    refused!(graph.matrix_inverse(&matrix, None), Error::InvalidShape(_));
    refused!(graph.matrix_inverse(&placeholder(&graph, &[3, 3], data_type::INT32), None), Error::InvalidDataType(_));
    refused!(graph.one_hot(&ints, 4, 0x1234, None), Error::UnsupportedDataType(_));
    refused!(graph.one_hot(&ints, 4, data_type::INT4, None), Error::UnsupportedDataType(_));
    let rate = floats(&graph, &[]);
    refused!(
        graph.stochastic_gradient_descent(&rate, &matrix, &floats(&graph, &[4, 3]), None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.stochastic_gradient_descent(&floats(&graph, &[5]), &matrix, &labels, None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.stochastic_gradient_descent(&placeholder(&graph, &[], data_type::FLOAT16), &matrix, &labels, None),
        Error::InvalidDataType(_)
    );
    refused!(graph.quantize(&ints, 0.1, 0.0, data_type::UINT8, None), Error::InvalidDataType(_));
    refused!(graph.quantize(&matrix, 0.1, 0.0, data_type::FLOAT32, None), Error::InvalidDataType(_));
    refused!(graph.quantize(&matrix, 0.1, 0.0, data_type::INT32, None), Error::InvalidDataType(_));
    refused!(graph.dequantize(&matrix, 0.1, 0.0, data_type::FLOAT32, None), Error::InvalidDataType(_));
    let bytes = placeholder(&graph, &[2, 3], data_type::UINT8);
    refused!(graph.dequantize(&bytes, 0.1, 0.0, data_type::INT32, None), Error::InvalidDataType(_));
    graph
        .dequantize(&bytes, 0.1, 0.0, data_type::FLOAT32, None)
        .expect("dequantize");

    let boxes = floats(&graph, &[1, 3, 4]);
    let scores = floats(&graph, &[1, 3, 1]);
    refused!(
        unsafe { graph.non_maximum_suppression(&placeholder(&graph, &[1, 3, 4], data_type::FLOAT16), &scores, 0.5, 0.1, false, 0, None) },
        Error::InvalidDataType(_)
    );
    refused!(
        unsafe { graph.non_maximum_suppression(&floats(&graph, &[3, 4]), &scores, 0.5, 0.1, false, 0, None) },
        Error::InvalidShape(_)
    );
    refused!(
        unsafe { graph.non_maximum_suppression(&boxes, &floats(&graph, &[1, 2, 1]), 0.5, 0.1, false, 0, None) },
        Error::InvalidShape(_)
    );
    refused!(
        unsafe { graph.non_maximum_suppression(&boxes, &scores, 0.5, 0.1, false, 9, None) },
        Error::InvalidArgument(_)
    );
    still_runs(&graph);
}

#[test]
fn resampling_and_sparse_ops_are_checked() {
    let graph = Graph::new().expect("graph");
    let image = floats(&graph, &[1, 4, 4, 2]);
    let nhwc = tensor_named_data_layout::NHWC;
    let bilinear = resize_mode::BILINEAR;
    refused!(graph.resize(&image, &[8, 8, 8], bilinear, true, false, nhwc, None), Error::InvalidShape(_));
    refused!(graph.resize(&floats(&graph, &[4, 4]), &[8, 8], bilinear, true, false, nhwc, None), Error::InvalidShape(_));
    refused!(
        graph.resize(&image, &[8, 8], bilinear, true, false, tensor_named_data_layout::HW, None),
        Error::InvalidArgument(_)
    );
    refused!(graph.resize(&image, &[8, 8], 99, true, false, nhwc, None), Error::InvalidArgument(_));
    refused!(
        graph.resize(&floats(&graph, &[4, 4, 2]), &[8, 8], bilinear, true, false, nhwc, None),
        Error::InvalidShape(_)
    );
    assert_eq!(
        graph
            .resize(&floats(&graph, &[4, 4, 2]), &[8, 8], bilinear, true, false, tensor_named_data_layout::HWC, None)
            .expect("HWC resize")
            .shape(),
        Some(vec![8, 8, 2])
    );
    let float_size = graph
        .constant_scalar_shaped(8.0, &[2], data_type::FLOAT32)
        .expect("size");
    let floor = resize_nearest_rounding_mode::FLOOR;
    refused!(
        unsafe { graph.resize_nearest(&image, &float_size, floor, true, false, nhwc, None) },
        Error::InvalidDataType(_)
    );
    let long_size = graph
        .constant_scalar_shaped(8.0, &[3], data_type::INT32)
        .expect("size");
    refused!(
        unsafe { graph.resize_nearest(&image, &long_size, floor, true, false, nhwc, None) },
        Error::InvalidShape(_)
    );
    let size = graph
        .constant_scalar_shaped(8.0, &[2], data_type::INT32)
        .expect("size");
    refused!(
        unsafe { graph.resize_nearest(&image, &size, 99, true, false, nhwc, None) },
        Error::InvalidArgument(_)
    );

    let coordinates = floats(&graph, &[1, 3, 3, 2]);
    let zero = padding_mode::ZERO;
    let sample = |source: &Tensor, coordinates: &Tensor, layout: usize, padding: isize, sampling: usize| {
        graph.sample_grid(source, coordinates, layout, true, false, false, padding, sampling, 0.0, None)
    };
    refused!(sample(&image, &floats(&graph, &[1, 3, 3, 3]), nhwc, zero, bilinear), Error::InvalidShape(_));
    refused!(sample(&floats(&graph, &[4, 4, 2]), &coordinates, nhwc, zero, bilinear), Error::InvalidShape(_));
    refused!(sample(&image, &floats(&graph, &[2, 3, 3, 2]), nhwc, zero, bilinear), Error::InvalidShape(_));
    refused!(sample(&image, &coordinates, nhwc, 99, bilinear), Error::InvalidArgument(_));
    refused!(sample(&image, &coordinates, nhwc, padding_mode::PERIODIC, bilinear), Error::InvalidArgument(_));
    refused!(sample(&image, &coordinates, nhwc, zero, 99), Error::InvalidArgument(_));
    refused!(sample(&image, &coordinates, tensor_named_data_layout::HW, zero, bilinear), Error::InvalidArgument(_));
    assert_eq!(
        sample(&image, &coordinates, nhwc, padding_mode::CLAMP_TO_EDGE, resize_mode::NEAREST)
            .expect("grid sample")
            .shape(),
        Some(vec![1, 3, 3, 2])
    );

    refused!(
        CreateSparseDescriptor::new(sparse_storage_type::COO, data_type::FLOAT16),
        Error::InvalidDataType(_)
    );
    refused!(CreateSparseDescriptor::new(9, data_type::FLOAT32), Error::InvalidArgument(_));
    let coo = CreateSparseDescriptor::new(sparse_storage_type::COO, data_type::FLOAT32).expect("coo");
    let csr = CreateSparseDescriptor::new(sparse_storage_type::CSR, data_type::FLOAT32).expect("csr");
    let values = floats(&graph, &[2]);
    let index = |length: usize, data: u32| placeholder(&graph, &[length], data);
    let int32 = data_type::INT32;
    refused!(graph.sparse_tensor_with_descriptor(&coo, &[&values], &[3, 3], None), Error::InvalidArgument(_));
    refused!(
        graph.sparse_tensor_with_descriptor(&coo, &[&values, &index(2, data_type::UINT32), &index(2, data_type::UINT32)], &[3, 3], None),
        Error::InvalidDataType(_)
    );
    refused!(
        graph.sparse_tensor_with_descriptor(&coo, &[&placeholder(&graph, &[2], data_type::FLOAT16), &index(2, int32), &index(2, int32)], &[3, 3], None),
        Error::InvalidDataType(_)
    );
    refused!(
        graph.sparse_tensor_with_descriptor(&coo, &[&values, &index(3, int32), &index(2, int32)], &[3, 3], None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.sparse_tensor_with_descriptor(&csr, &[&values, &index(2, int32), &index(2, int32)], &[3, 3], None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.sparse_tensor_with_descriptor(&coo, &[&values, &index(2, int32), &index(2, int32)], &[3, 3, 3], None),
        Error::InvalidShape(_)
    );
    let sparse = graph
        .sparse_tensor_with_descriptor(&csr, &[&values, &index(2, int32), &index(4, int32)], &[3, 3], None)
        .expect("CSR tensor");
    assert_eq!(sparse.shape(), Some(vec![3, 3]));
    still_runs(&graph);
}

#[test]
fn executable_types_and_serialization_targets_are_checked() {
    use apple_mpsgraph::{deployment_platform, FeedDescription};
    let device = device();
    let graph = Graph::new().expect("graph");
    let input = floats(&graph, &[2]);
    let doubled = graph.addition(&input, &input, None).expect("addition");
    let executable = graph
        .compile(
            &device,
            &[FeedDescription::new(&input, &[2], data_type::FLOAT32)],
            &[&doubled],
        )
        .expect("compile");
    let wide = ShapedType::new(Some(&[3]), data_type::FLOAT32).expect("type");
    let unranked = ShapedType::new(None, data_type::FLOAT32).expect("type");
    let pair = ShapedType::new(Some(&[2]), data_type::FLOAT32).expect("type");
    refused!(executable.output_types(Some(&device), &[], None), Error::InvalidShape(_));
    refused!(executable.output_types(Some(&device), &[&wide], None), Error::InvalidShape(_));
    refused!(executable.output_types(Some(&device), &[&unranked], None), Error::InvalidShape(_));
    refused!(executable.specialize(Some(&device), &[&wide], None), Error::InvalidShape(_));
    let output = executable
        .output_types(Some(&device), &[&pair], None)
        .expect("output types");
    assert_eq!(output[0].shape(), Some(vec![2]));

    let descriptor = ExecutableSerializationDescriptor::new().expect("descriptor");
    refused!(descriptor.set_deployment_platform(99), Error::InvalidArgument(_));
    let directory = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    let path = directory.join("serialization-check.mpsgraphpackage");
    let path = path.to_str().expect("utf8 path");
    for (platform, target) in [
        (deployment_platform::MACOS, "13.9"),
        (deployment_platform::MACOS, ""),
        (deployment_platform::MACOS, "x.y"),
        (deployment_platform::IOS, "16.4"),
        (deployment_platform::TVOS, "16"),
        (deployment_platform::VISIONOS, "1.0"),
    ] {
        descriptor.set_deployment_platform(platform).expect("platform");
        descriptor
            .set_minimum_deployment_target(target)
            .expect("target");
        refused!(
            executable.serialize_package(path, Some(&descriptor)),
            Error::InvalidArgument(_)
        );
    }
    descriptor
        .set_deployment_platform(deployment_platform::MACOS)
        .expect("platform");
    descriptor
        .set_minimum_deployment_target("14.0")
        .expect("target");
    executable
        .serialize_package(path, Some(&descriptor))
        .expect("serialize for macOS 14");
    std::fs::remove_dir_all(path).expect("remove package");
}
