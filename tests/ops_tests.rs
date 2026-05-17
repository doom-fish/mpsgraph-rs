use apple_metal::MetalDevice;
use apple_mpsgraph::{
    data_type, BinaryArithmeticOp, Feed, Graph, ReductionAxesOp, TensorData, UnaryArithmeticOp,
};

#[test]
fn unary_arithmetic_and_reduction_execute() {
    let device = MetalDevice::system_default().expect("no Metal device available");
    let graph = Graph::new().expect("graph");
    let input = graph
        .placeholder(Some(&[2, 2]), data_type::FLOAT32, Some("input"))
        .expect("placeholder");
    let squared = graph
        .unary_arithmetic(UnaryArithmeticOp::Square, &input, Some("square"))
        .expect("square");
    let reduced = graph
        .reduce_axes(ReductionAxesOp::Sum, &squared, &[1], Some("row_sum"))
        .expect("reduce");

    let input_data =
        TensorData::from_f32_slice(&device, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("tensor data");
    let results = graph
        .run(&[Feed::new(&input, &input_data)], &[&reduced])
        .expect("run");
    assert_eq!(results[0].read_f32().expect("read"), vec![5.0, 25.0]);
}

#[test]
fn concat_split_and_topk_execute() {
    let device = MetalDevice::system_default().expect("no Metal device available");
    let graph = Graph::new().expect("graph");
    let input = graph
        .placeholder(Some(&[2, 3]), data_type::FLOAT32, Some("input"))
        .expect("placeholder");
    let squared = graph
        .binary_arithmetic(
            BinaryArithmeticOp::Multiplication,
            &input,
            &input,
            Some("square"),
        )
        .expect("square via multiply");
    let padded = graph
        .pad(&squared, 0, &[0, 1], &[0, 1], 0.0, Some("pad"))
        .expect("pad");
    let topk = graph.top_k(&input, 2, Some("topk")).expect("topk");
    let split = graph.split_num(&padded, 2, 1, Some("split"));
    assert_eq!(split.len(), 2);

    let input_data = TensorData::from_f32_slice(&device, &[1.0, 3.0, 2.0, 4.0, 6.0, 5.0], &[2, 3])
        .expect("tensor data");
    let results = graph
        .run(
            &[Feed::new(&input, &input_data)],
            &[&topk.0, &split[0], &split[1]],
        )
        .expect("run");

    assert_eq!(
        results[0].read_f32().expect("topk values"),
        vec![3.0, 2.0, 6.0, 5.0]
    );
    assert_eq!(
        results[1].read_f32().expect("split 0"),
        vec![0.0, 1.0, 9.0, 0.0, 16.0, 36.0]
    );
    assert_eq!(
        results[2].read_f32().expect("split 1"),
        vec![4.0, 0.0, 25.0, 0.0]
    );
}
