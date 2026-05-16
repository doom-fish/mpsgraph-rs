use apple_metal::MetalDevice;
use apple_mpsgraph::{data_type, Feed, Graph, TensorData};

fn main() {
    let device = MetalDevice::system_default().expect("no Metal device available");
    let graph = Graph::new().expect("failed to create MPSGraph");

    let input = graph
        .placeholder(Some(&[2, 2]), data_type::FLOAT32, Some("input"))
        .expect("failed to create placeholder");
    let bias = graph
        .constant_scalar(1.0, data_type::FLOAT32)
        .expect("failed to create scalar constant");
    let added = graph
        .addition(&input, &bias, Some("add"))
        .expect("failed to create addition op");
    let output = graph
        .relu(&added, Some("relu"))
        .expect("failed to create relu op");

    let input_data = TensorData::from_f32_slice(&device, &[1.0, -2.0, 3.0, -4.0], &[2, 2])
        .expect("failed to create tensor data");
    let results = graph
        .run(&[Feed::new(&input, &input_data)], &[&output])
        .expect("failed to execute graph");
    let values = results[0].read_f32().expect("failed to read tensor output");

    assert_eq!(values, vec![2.0, 0.0, 4.0, 0.0]);
    println!("add+relu smoke passed: {values:?}");
}
