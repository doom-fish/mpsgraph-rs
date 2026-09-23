use apple_metal::MetalDevice;
use apple_mpsgraph::{
    data_type, execution_stage, Error, ExecutableExecutionDescriptor, Feed, FeedDescription, Graph,
    ReductionAxesOp, ReductionAxisOp, Tensor, TensorData,
};
use std::time::Duration;

fn device() -> MetalDevice {
    MetalDevice::system_default().expect("no Metal device available")
}

fn placeholder(graph: &Graph, shape: &[usize], data: u32) -> Tensor {
    graph
        .placeholder(Some(shape), data, None)
        .expect("placeholder")
}

fn values(count: u8) -> Vec<f32> {
    (1..=count).map(f32::from).collect()
}

#[test]
fn shape_and_axis_mistakes_are_refused_before_mpsgraph() {
    let graph = Graph::new().expect("graph");
    let x = placeholder(&graph, &[2, 3], data_type::FLOAT32);
    assert!(graph.reshape(&x, &[5], None).is_none());
    assert!(graph.reshape(&x, &[0], None).is_none());
    assert!(graph.reshape(&x, &[usize::MAX, 2], None).is_none());
    assert_eq!(
        graph.reshape(&x, &[3, 2], None).expect("reshape").shape(),
        Some(vec![3, 2])
    );
    assert!(graph.reduction_sum(&x, &[5], None).is_none());
    assert_eq!(
        graph.reduction_sum(&x, &[1], None).expect("sum").shape(),
        Some(vec![2, 1])
    );
    assert!(graph
        .reduce_axes(ReductionAxesOp::Sum, &x, &[2], None)
        .is_none());
    assert!(graph
        .reduce_axis(ReductionAxisOp::Maximum, &x, 2, None)
        .is_none());
    assert!(graph.transpose(&x, &[0], None).is_none());
    assert!(graph.transpose(&x, &[0, 0], None).is_none());
    assert!(graph.transpose(&x, &[0, 5], None).is_none());
    assert_eq!(
        graph
            .transpose(&x, &[1, 0], None)
            .expect("transpose")
            .shape(),
        Some(vec![3, 2])
    );
    assert!(graph.slice(&x, 5, 0, 1, None).is_none());
    assert!(graph.slice(&x, 1, -5, 1, None).is_none());
    assert!(graph.slice(&x, 1, 0, -1, None).is_none());
    assert!(graph.slice(&x, 1, 2, 2, None).is_none());
    assert_eq!(
        graph.slice(&x, 1, -1, 1, None).expect("slice").shape(),
        Some(vec![2, 1])
    );
    assert!(graph.softmax(&x, 7, None).is_none());
    assert!(graph.softmax(&x, -3, None).is_none());
    let softmax = graph
        .softmax(&x, -1, None)
        .expect("softmax over the last axis");
    let input = TensorData::from_f32_slice(&device(), &values(6), &[2, 3]).expect("input");
    let probabilities = graph
        .run(&[Feed::new(&x, &input)], &[&softmax])
        .expect("run softmax")[0]
        .read_f32()
        .expect("read softmax");
    let expected = [
        0.090_031, 0.244_728, 0.665_241, 0.090_031, 0.244_728, 0.665_241,
    ];
    assert!(
        probabilities
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() < 1e-5),
        "{probabilities:?}"
    );
    assert!(graph.broadcast(&x, &[4, 5], None).is_none());
    assert!(graph.broadcast(&x, &[3], None).is_none());
    assert_eq!(
        graph
            .broadcast(&x, &[4, 2, 3], None)
            .expect("broadcast")
            .shape(),
        Some(vec![4, 2, 3])
    );
    assert!(graph.split_num(&x, 0, 1, None).is_empty());
    assert!(graph.split_num(&x, 4, 1, None).is_empty());
    let uneven = graph.split_num(&x, 2, 1, None);
    assert_eq!(
        uneven.iter().map(Tensor::shape).collect::<Vec<_>>(),
        vec![Some(vec![2, 2]), Some(vec![2, 1])]
    );
    assert!(graph.split_sizes(&x, &[1, 1], 1, None).is_empty());
    assert!(graph.split_sizes(&x, &[0, 3], 1, None).is_empty());
    let sized = graph.split_sizes(&x, &[1, 2], 1, None);
    assert_eq!(
        sized.iter().map(Tensor::shape).collect::<Vec<_>>(),
        vec![Some(vec![2, 1]), Some(vec![2, 2])]
    );
}

#[test]
fn binary_and_matrix_operands_must_be_compatible() {
    let graph = Graph::new().expect("graph");
    let x = placeholder(&graph, &[2, 3], data_type::FLOAT32);
    let wide = placeholder(&graph, &[4, 5], data_type::FLOAT32);
    let half = placeholder(&graph, &[2, 3], data_type::FLOAT16);
    let row = placeholder(&graph, &[3], data_type::FLOAT32);
    assert!(graph.addition(&x, &wide, None).is_none());
    assert!(graph.multiplication(&x, &half, None).is_none());
    assert_eq!(
        graph
            .addition(&x, &row, None)
            .expect("broadcast add")
            .shape(),
        Some(vec![2, 3])
    );
    assert!(graph.matrix_multiplication(&x, &x, None).is_none());
    assert!(graph.matrix_multiplication(&x, &row, None).is_none());
    let batched = placeholder(&graph, &[2, 2, 3], data_type::FLOAT32);
    let other_batch = placeholder(&graph, &[3, 3, 4], data_type::FLOAT32);
    assert!(graph
        .matrix_multiplication(&batched, &other_batch, None)
        .is_none());
    let tall = placeholder(&graph, &[3, 4], data_type::FLOAT32);
    assert_eq!(
        graph
            .matrix_multiplication(&x, &tall, None)
            .expect("matmul")
            .shape(),
        Some(vec![2, 4])
    );
}

#[test]
fn feeds_must_match_their_placeholders() {
    let device = device();
    let graph = Graph::new().expect("graph");
    let x = placeholder(&graph, &[2, 3], data_type::FLOAT32);
    let doubled = graph.addition(&x, &x, None).expect("addition");
    let flat = TensorData::from_f32_slice(&device, &values(6), &[6]).expect("flat");
    assert!(matches!(
        graph.run(&[Feed::new(&x, &flat)], &[&doubled]),
        Err(Error::InvalidShape(_))
    ));
    let half = TensorData::from_bytes(&device, &[0; 12], &[2, 3], data_type::FLOAT16).expect("f16");
    assert!(matches!(
        graph.run(&[Feed::new(&x, &half)], &[&doubled]),
        Err(Error::InvalidShape(_))
    ));
    let queue = device.new_command_queue().expect("queue");
    assert!(matches!(
        graph.run_with_command_queue(&queue, &[Feed::new(&x, &flat)], &[&doubled]),
        Err(Error::InvalidShape(_))
    ));
    assert!(graph
        .compile(
            &device,
            &[FeedDescription::new(&x, &[5], data_type::FLOAT32)],
            &[&doubled]
        )
        .is_none());
    let good = TensorData::from_f32_slice(&device, &values(6), &[2, 3]).expect("data");
    let results = graph
        .run(&[Feed::new(&x, &good)], &[&doubled])
        .expect("run");
    assert_eq!(
        results[0].read_f32().expect("read"),
        vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    );
}

#[test]
fn executables_check_inputs_and_preallocated_results() {
    let device = device();
    let queue = device.new_command_queue().expect("queue");
    let graph = Graph::new().expect("graph");
    let x = placeholder(&graph, &[2, 3], data_type::FLOAT32);
    let doubled = graph.addition(&x, &x, None).expect("addition");
    let executable = graph
        .compile(
            &device,
            &[FeedDescription::new(&x, &[2, 3], data_type::FLOAT32)],
            &[&doubled],
        )
        .expect("compile");
    let input = TensorData::from_f32_slice(&device, &values(6), &[2, 3]).expect("input");
    let wrong = TensorData::from_f32_slice(&device, &values(4), &[4]).expect("wrong");
    assert!(matches!(
        executable.run(&queue, &[]),
        Err(Error::InvalidShape(_))
    ));
    assert!(matches!(
        executable.run(&queue, &[&wrong]),
        Err(Error::InvalidShape(_))
    ));
    let small = TensorData::from_f32_slice(&device, &[0.0], &[1]).expect("small");
    assert!(matches!(
        executable.run_with_descriptor(&queue, &[&input], Some(&[&small]), None),
        Err(Error::InvalidShape(_))
    ));
    let output = TensorData::from_f32_slice(&device, &[0.0; 6], &[2, 3]).expect("output");
    let results = executable
        .run_with_descriptor(&queue, &[&input], Some(&[&output]), None)
        .expect("run into preallocated output");
    assert_eq!(
        results[0].read_f32().expect("read"),
        vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    );
    assert_eq!(
        output.read_f32().expect("preallocated output"),
        vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    );
}

#[test]
fn async_runs_expose_results_only_after_completion() {
    let device = device();
    let queue = device.new_command_queue().expect("queue");
    let graph = Graph::new().expect("graph");
    let x = placeholder(&graph, &[2, 3], data_type::FLOAT32);
    let tripled = graph
        .multiplication(
            &x,
            &graph
                .constant_scalar(3.0, data_type::FLOAT32)
                .expect("constant"),
            None,
        )
        .expect("multiplication");
    let executable = graph
        .compile(
            &device,
            &[FeedDescription::new(&x, &[2, 3], data_type::FLOAT32)],
            &[&tripled],
        )
        .expect("compile");
    let input = TensorData::from_f32_slice(&device, &values(6), &[2, 3]).expect("input");
    let expected = vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0];

    let run = executable
        .run_async_with_descriptor(&queue, &[&input], None, None)
        .expect("run async");
    let results = run.wait().expect("completed");
    assert_eq!(results[0].read_f32().expect("read"), expected);

    let descriptor = ExecutableExecutionDescriptor::new().expect("descriptor");
    let event = device.new_shared_event().expect("shared event");
    unsafe { descriptor.signal_shared_event_raw(event.as_ptr(), execution_stage::COMPLETED, 5) }
        .expect("signal");
    let output = TensorData::from_f32_slice(&device, &[0.0; 6], &[2, 3]).expect("output");
    let run = executable
        .run_async_with_descriptor(&queue, &[&input], Some(vec![output]), Some(&descriptor))
        .expect("run async into output");
    let results = match run.wait_timeout(Duration::from_secs(30)) {
        Ok(results) => results.expect("completed"),
        Err(run) => {
            assert!(!run.is_complete());
            run.wait().expect("completed after the timeout")
        }
    };
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].read_f32().expect("read"), expected);
    assert!(event.wait_until_signaled_value(5, 10_000));
    assert_eq!(event.signaled_value(), 5);
    assert!(!descriptor.wait_until_completed());

    let wrong = TensorData::from_f32_slice(&device, &[0.0; 4], &[4]).expect("wrong");
    assert!(matches!(
        executable.run_async_with_descriptor(&queue, &[&input], Some(vec![wrong]), None),
        Err(Error::InvalidShape(_))
    ));
    drop(
        executable
            .run_async_with_descriptor(&queue, &[&input], None, None)
            .expect("dropped run"),
    );
}
