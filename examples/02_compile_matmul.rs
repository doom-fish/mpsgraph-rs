use apple_metal::MetalDevice;
use apple_mpsgraph::{data_type, FeedDescription, Graph, TensorData};

fn main() {
    let device = MetalDevice::system_default().expect("no Metal device available");
    let queue = device
        .new_command_queue()
        .expect("failed to create command queue");
    let graph = Graph::new().expect("failed to create MPSGraph");

    let left = graph
        .placeholder(Some(&[2, 3]), data_type::FLOAT32, Some("left"))
        .expect("failed to create left placeholder");
    let right = graph
        .placeholder(Some(&[3, 2]), data_type::FLOAT32, Some("right"))
        .expect("failed to create right placeholder");
    let output = graph
        .matrix_multiplication(&left, &right, Some("matmul"))
        .expect("failed to create matrix multiplication op");

    let executable = graph
        .compile(
            &device,
            &[
                FeedDescription::new(&left, &[2, 3], data_type::FLOAT32),
                FeedDescription::new(&right, &[3, 2], data_type::FLOAT32),
            ],
            &[&output],
        )
        .expect("failed to compile executable");

    let left_data = TensorData::from_f32_slice(&device, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .expect("failed to create left tensor data");
    let right_data =
        TensorData::from_f32_slice(&device, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2])
            .expect("failed to create right tensor data");

    let results = executable
        .run(&queue, &[&left_data, &right_data])
        .expect("failed to run executable");
    let values = results[0].read_f32().expect("failed to read tensor output");
    let expected = [58.0_f32, 64.0, 139.0, 154.0];
    for (actual, expected_value) in values.iter().zip(expected) {
        assert!(
            (actual - expected_value).abs() < 1.0e-4,
            "unexpected matrix multiply result: {values:?}"
        );
    }

    println!("compile+matmul smoke passed: {values:?}");
}
