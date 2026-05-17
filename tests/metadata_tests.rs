use apple_metal::MetalDevice;
use apple_mpsgraph::{
    data_type, deployment_platform, graph_options, optimization, CompilationDescriptor,
    ExecutableExecutionDescriptor, ExecutableSerializationDescriptor, FeedDescription, Graph,
    GraphDevice, ShapedType,
};

#[test]
fn graph_metadata_and_descriptors_round_trip() {
    let metal = MetalDevice::system_default().expect("no Metal device available");
    let graph_device = GraphDevice::from_metal_device(&metal).expect("graph device");
    assert_eq!(
        graph_device.device_type(),
        apple_mpsgraph::graph_device_type::METAL
    );

    let graph = Graph::new().expect("graph");
    graph
        .set_options(graph_options::VERBOSE)
        .expect("set graph options");
    assert_eq!(graph.options(), graph_options::VERBOSE);

    let input = graph
        .placeholder(Some(&[2, 2]), data_type::FLOAT32, Some("input"))
        .expect("placeholder");
    assert_eq!(graph.placeholder_tensors().len(), 1);
    assert_eq!(input.shape(), Some(vec![2, 2]));
    assert_eq!(input.data_type(), data_type::FLOAT32);
    assert!(input.operation().is_some());

    let shaped = ShapedType::new(Some(&[2, 2]), data_type::FLOAT32).expect("shaped type");
    assert_eq!(shaped.shape(), Some(vec![2, 2]));
    shaped.set_shape(Some(&[4])).expect("set shape");
    assert_eq!(shaped.shape(), Some(vec![4]));
    shaped.set_data_type(data_type::FLOAT16).expect("set dtype");
    assert_eq!(shaped.data_type(), data_type::FLOAT16);

    let compile_desc = CompilationDescriptor::new().expect("compile desc");
    compile_desc
        .disable_type_inference()
        .expect("disable type inference");
    compile_desc
        .set_optimization_level(optimization::LEVEL1)
        .expect("set optimization level");
    compile_desc
        .set_wait_for_compilation_completion(true)
        .expect("set wait");
    assert!(compile_desc.wait_for_compilation_completion());

    let output = graph
        .unary_arithmetic(
            apple_mpsgraph::UnaryArithmeticOp::Square,
            &input,
            Some("square"),
        )
        .expect("square op");
    let executable = graph
        .compile_with_descriptor(
            Some(&metal),
            &[FeedDescription::new(&input, &[2, 2], data_type::FLOAT32)],
            &[&output],
            Some(&compile_desc),
        )
        .expect("compile with descriptor");
    assert_eq!(executable.feed_tensors().len(), 1);
    assert_eq!(executable.target_tensors().len(), 1);

    let input_type = ShapedType::new(Some(&[2, 2]), data_type::FLOAT32).expect("input type");
    let output_types = executable
        .output_types(Some(&metal), &[&input_type], Some(&compile_desc))
        .expect("output types");
    assert_eq!(output_types.len(), 1);
    assert_eq!(output_types[0].shape(), Some(vec![2, 2]));

    let exec_desc = ExecutableExecutionDescriptor::new().expect("exec desc");
    exec_desc
        .set_wait_until_completed(true)
        .expect("set exec wait");
    assert!(exec_desc.wait_until_completed());

    let serialization = ExecutableSerializationDescriptor::new().expect("serialization desc");
    serialization.set_append(true).expect("set append");
    serialization
        .set_deployment_platform(deployment_platform::MACOS)
        .expect("set platform");
    serialization
        .set_minimum_deployment_target("14.0")
        .expect("set target");
    assert!(serialization.append());
    assert_eq!(
        serialization.deployment_platform(),
        deployment_platform::MACOS
    );
    assert_eq!(
        serialization
            .minimum_deployment_target()
            .expect("get target"),
        "14.0"
    );
}
