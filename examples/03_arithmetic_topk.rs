use apple_metal::MetalDevice;
use apple_mpsgraph::{data_type, Feed, Graph, ReductionAxesOp, TensorData, UnaryArithmeticOp};

fn main() {
    let device = MetalDevice::system_default().expect("no Metal device available");
    let graph = Graph::new().expect("graph");
    let input = graph
        .placeholder(Some(&[2, 3]), data_type::FLOAT32, Some("input"))
        .expect("placeholder");
    let squared = graph
        .unary_arithmetic(UnaryArithmeticOp::Square, &input, Some("square"))
        .expect("square");
    let row_sum = graph
        .reduce_axes(ReductionAxesOp::Sum, &squared, &[1], Some("row_sum"))
        .expect("reduce");
    let topk = graph.top_k(&input, 2, Some("topk")).expect("topk");

    let input_data = TensorData::from_f32_slice(&device, &[1.0, 3.0, 2.0, 4.0, 6.0, 5.0], &[2, 3])
        .expect("tensor data");
    let results = graph
        .run(&[Feed::new(&input, &input_data)], &[&row_sum, &topk.0])
        .expect("run");

    println!("row sums: {:?}", results[0].read_f32().expect("row sums"));
    println!(
        "top-k values: {:?}",
        results[1].read_f32().expect("topk values")
    );
}
