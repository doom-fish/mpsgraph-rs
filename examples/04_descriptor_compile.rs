use apple_metal::MetalDevice;
use apple_mpsgraph::{
    data_type, optimization, CompilationDescriptor, FeedDescription, Graph, ShapedType,
    UnaryArithmeticOp,
};

fn main() {
    let device = MetalDevice::system_default().expect("no Metal device available");
    let graph = Graph::new().expect("graph");
    let input = graph
        .placeholder(Some(&[4]), data_type::FLOAT32, Some("input"))
        .expect("placeholder");
    let output = graph
        .unary_arithmetic(UnaryArithmeticOp::Absolute, &input, Some("abs"))
        .expect("absolute");

    let descriptor = CompilationDescriptor::new().expect("compilation descriptor");
    descriptor
        .set_optimization_level(optimization::LEVEL1)
        .expect("set optimization level");
    descriptor
        .set_wait_for_compilation_completion(true)
        .expect("set wait");

    let executable = graph
        .compile_with_descriptor(
            Some(&device),
            &[FeedDescription::new(&input, &[4], data_type::FLOAT32)],
            &[&output],
            Some(&descriptor),
        )
        .expect("compile");
    let input_type = ShapedType::new(Some(&[4]), data_type::FLOAT32).expect("shaped type");
    let output_types = executable
        .output_types(Some(&device), &[&input_type], Some(&descriptor))
        .expect("output types");

    println!("feed tensors: {}", executable.feed_tensors().len());
    println!("target tensors: {}", executable.target_tensors().len());
    println!("output type: {:?}", output_types[0].shape());
}
