use apple_metal::MetalDevice;
use apple_mpsgraph::{data_type, Feed, Graph, TensorData};

fn main() {
    let device = MetalDevice::system_default().expect("no Metal device available");
    let graph = Graph::new().expect("graph");
    let input = graph
        .placeholder(Some(&[2, 2]), data_type::FLOAT32, Some("input"))
        .expect("placeholder");
    let concat = graph
        .concat_pair(&input, &input, 1, Some("concat"))
        .expect("concat");
    let split = graph.split_num(&concat, 2, 1, Some("split"));
    let stacked = graph
        .stack(&[&split[0], &split[1]], 0, Some("stack"))
        .expect("stack");

    let input_data =
        TensorData::from_f32_slice(&device, &[1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("tensor data");
    let results = graph
        .run(&[Feed::new(&input, &input_data)], &[&stacked])
        .expect("run");

    println!(
        "stacked tensor bytes: {}",
        results[0].byte_len().expect("byte len")
    );
}
