#![allow(clippy::too_many_lines)]

use apple_metal::MetalDevice;
use apple_mpsgraph::{
    data_type, padding_mode, padding_style, random_distribution, random_normal_sampling_method,
    rnn_activation, scatter_mode, tensor_named_data_layout, BinaryArithmeticOp,
    CompilationDescriptor, Convolution2DDescriptor, Convolution2DDescriptorInfo, Error, Feed,
    FeedDescription, GRUDescriptor, Graph, LSTMDescriptor, RandomOpDescriptor, ShapedType,
    SingleGateRNNDescriptor, Tensor, TensorData, UnaryArithmeticOp, WhileBeforeResult,
};
use std::cell::RefCell;

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

fn constant(graph: &Graph, value: f64, shape: &[usize], data: u32) -> Tensor {
    graph
        .constant_scalar_shaped(value, shape, data)
        .expect("constant")
}

fn scalar(graph: &Graph, value: f64, data: u32) -> Tensor {
    graph.constant_scalar(value, data).expect("scalar")
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

fn run_f32(graph: &Graph, target: &Tensor) -> Vec<f32> {
    graph.run(&[], &[target]).expect("run")[0]
        .read_f32()
        .expect("read")
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

#[test]
fn tensors_and_operations_from_other_graphs_are_refused() {
    let device = device();
    let graph = Graph::new().expect("graph");
    let other = Graph::new().expect("other graph");
    let foreign = floats(&other, &[2]);
    let local = floats(&graph, &[2]);
    refused!(graph.addition(&foreign, &local, None), Error::ForeignTensor);
    refused!(graph.relu(&foreign, None), Error::ForeignTensor);
    refused!(
        graph.concat_tensors(&[&local, &foreign], 0, false, None),
        Error::ForeignTensor
    );
    refused!(
        graph.gather_along_axis(0, &foreign, &placeholder(&graph, &[2], data_type::INT32), None),
        Error::ForeignTensor
    );
    let operation = foreign.operation().expect("operation");
    refused!(
        graph.control_dependency(&[&operation], Vec::new, None),
        Error::ForeignOperation
    );
    let data = TensorData::from_f32_slice(&device, &[1.0, 2.0], &[2]).expect("data");
    refused!(
        graph.run(&[Feed::new(&foreign, &data)], &[&local]),
        Error::ForeignTensor
    );
    refused!(graph.run(&[], &[&foreign]), Error::ForeignTensor);
    refused!(
        graph.compile(&device, &[], &[&foreign]),
        Error::ForeignTensor
    );
    still_runs(&graph);
}

#[test]
fn tensors_cannot_escape_control_flow_blocks() {
    let graph = Graph::new().expect("graph");
    let predicate = scalar(&graph, 1.0, data_type::BOOL);
    let leaked = RefCell::new(None);
    let results = graph
        .if_then_else(
            &predicate,
            || {
                let value = constant(&graph, 1.0, &[2], data_type::FLOAT32);
                leaked.replace(Some(graph.relu(&value, None).expect("relu")));
                vec![value]
            },
            || vec![constant(&graph, 2.0, &[2], data_type::FLOAT32)],
            None,
        )
        .expect("if");
    let leaked = leaked.take().expect("leaked tensor");
    refused!(graph.addition(&leaked, &results[0], None), Error::ForeignTensor);
    refused!(graph.run(&[], &[&leaked]), Error::ForeignTensor);

    let sibling = RefCell::new(None);
    refused!(
        graph.if_then_else(
            &predicate,
            || {
                let value = constant(&graph, 1.0, &[2], data_type::FLOAT32);
                sibling.replace(Some(graph.relu(&value, None).expect("relu")));
                vec![value]
            },
            || vec![sibling.take().expect("then tensor")],
            None,
        ),
        Error::ForeignTensor
    );

    let index_leak = RefCell::new(None);
    let zero = scalar(&graph, 0.0, data_type::FLOAT32);
    graph
        .for_loop_iterations(
            &scalar(&graph, 3.0, data_type::INT32),
            &[&zero],
            |index, arguments| {
                index_leak.replace(Some(
                    graph
                        .unary_arithmetic(UnaryArithmeticOp::Identity, index, None)
                        .expect("identity"),
                ));
                vec![graph
                    .addition(&arguments[0], &scalar(&graph, 1.0, data_type::FLOAT32), None)
                    .expect("add")]
            },
            None,
        )
        .expect("for");
    let index_leak = index_leak.take().expect("index");
    refused!(
        graph.unary_arithmetic(UnaryArithmeticOp::Identity, &index_leak, None),
        Error::ForeignTensor
    );

    let outer = constant(&graph, 5.0, &[2], data_type::FLOAT32);
    let nested = graph
        .if_then_else(
            &predicate,
            || {
                graph
                    .if_then_else(
                        &predicate,
                        || vec![graph.addition(&outer, &outer, None).expect("inner add")],
                        || vec![constant(&graph, 3.0, &[2], data_type::FLOAT32)],
                        None,
                    )
                    .expect("inner if")
            },
            || vec![constant(&graph, 2.0, &[2], data_type::FLOAT32)],
            None,
        )
        .expect("nested if");
    assert_eq!(run_f32(&graph, &nested[0]), vec![10.0, 10.0]);
    assert_eq!(run_f32(&graph, &results[0]), vec![1.0, 1.0]);
    still_runs(&graph);
}

#[test]
fn if_blocks_must_agree() {
    let graph = Graph::new().expect("graph");
    let predicate = scalar(&graph, 1.0, data_type::BOOL);
    let pair = || vec![constant(&graph, 1.0, &[2], data_type::FLOAT32)];
    refused!(
        graph.if_then_else(&predicate, pair, Vec::new, None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.if_then_else(
            &predicate,
            || {
                vec![
                    constant(&graph, 1.0, &[2], data_type::FLOAT32),
                    constant(&graph, 1.0, &[2], data_type::FLOAT32),
                ]
            },
            pair,
            None,
        ),
        Error::InvalidShape(_)
    );
    refused!(
        graph.if_then_else(
            &predicate,
            pair,
            || vec![constant(&graph, 2.0, &[2], data_type::INT32)],
            None
        ),
        Error::InvalidShape(_)
    );
    refused!(
        graph.if_then_else(
            &predicate,
            pair,
            || vec![constant(&graph, 2.0, &[3], data_type::FLOAT32)],
            None
        ),
        Error::InvalidShape(_)
    );
    refused!(
        graph.if_then_else(
            &predicate,
            pair,
            || vec![constant(&graph, 2.0, &[2, 1], data_type::FLOAT32)],
            None
        ),
        Error::InvalidShape(_)
    );
    refused!(
        graph.if_then_else(&predicate, Vec::new, Vec::new, None),
        Error::InvalidShape(_)
    );
    let unranked = graph
        .placeholder(None, data_type::FLOAT32, None)
        .expect("unranked");
    refused!(
        graph.if_then_else(
            &predicate,
            || vec![graph.relu(&unranked, None).expect("relu")],
            || vec![constant(&graph, 2.0, &[2], data_type::FLOAT32)],
            None
        ),
        Error::InvalidShape(_)
    );
    refused!(
        graph.if_then_else(
            &constant(&graph, 1.0, &[3], data_type::BOOL),
            pair,
            pair,
            None
        ),
        Error::InvalidShape(_)
    );
    let chosen = graph
        .if_then_else(
            &predicate,
            pair,
            || vec![constant(&graph, 2.0, &[2], data_type::FLOAT32)],
            None,
        )
        .expect("valid if");
    assert_eq!(run_f32(&graph, &chosen[0]), vec![1.0, 1.0]);
    still_runs(&graph);
}

#[test]
fn loop_blocks_must_keep_their_types() {
    let graph = Graph::new().expect("graph");
    let zero = scalar(&graph, 0.0, data_type::INT32);
    let one = scalar(&graph, 1.0, data_type::INT32);
    let three = scalar(&graph, 3.0, data_type::INT32);
    let before = |inputs: &[Tensor]| WhileBeforeResult {
        predicate: graph
            .binary_arithmetic(BinaryArithmeticOp::LessThan, &inputs[0], &three, None)
            .expect("less than"),
        results: vec![graph
            .unary_arithmetic(UnaryArithmeticOp::Identity, &inputs[0], None)
            .expect("identity")],
    };
    refused!(
        graph.while_loop(
            &[&zero],
            before,
            |arguments| {
                vec![
                    graph.addition(&arguments[0], &one, None).expect("add"),
                    scalar(&graph, 5.0, data_type::INT32),
                ]
            },
            None
        ),
        Error::InvalidShape(_)
    );
    refused!(
        graph.while_loop(
            &[&zero],
            before,
            |_| vec![scalar(&graph, 1.0, data_type::FLOAT32)],
            None
        ),
        Error::InvalidShape(_)
    );
    refused!(
        graph.while_loop(
            &[&zero],
            |inputs| WhileBeforeResult {
                predicate: scalar(&graph, 1.0, data_type::FLOAT32),
                results: vec![graph
                    .unary_arithmetic(UnaryArithmeticOp::Identity, &inputs[0], None)
                    .expect("identity")],
            },
            |arguments| vec![graph.addition(&arguments[0], &one, None).expect("add")],
            None
        ),
        Error::InvalidShape(_)
    );
    refused!(
        graph.while_loop(
            &[&zero],
            |_| WhileBeforeResult {
                predicate: constant(&graph, 1.0, &[2], data_type::BOOL),
                results: vec![scalar(&graph, 1.0, data_type::INT32)],
            },
            |arguments| vec![graph.addition(&arguments[0], &one, None).expect("add")],
            None
        ),
        Error::InvalidShape(_)
    );
    refused!(
        graph.while_loop(
            &[&zero],
            |inputs| WhileBeforeResult {
                predicate: graph
                    .binary_arithmetic(BinaryArithmeticOp::LessThan, &inputs[0], &three, None)
                    .expect("less than"),
                results: Vec::new(),
            },
            |_| vec![scalar(&graph, 1.0, data_type::INT32)],
            None
        ),
        Error::InvalidShape(_)
    );
    let counted = graph
        .while_loop(
            &[&zero],
            before,
            |arguments| vec![graph.addition(&arguments[0], &one, None).expect("add")],
            None,
        )
        .expect("valid while");
    let count = graph.run(&[], &[&counted[0]]).expect("run while")[0]
        .read_bytes()
        .expect("read");
    assert_eq!(count, 3_i32.to_ne_bytes().to_vec());

    let start = scalar(&graph, 0.0, data_type::FLOAT32);
    let step = scalar(&graph, 1.0, data_type::FLOAT32);
    refused!(
        graph.for_loop(&zero, &three, &one, &[&start], |_, _| Vec::new(), None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.for_loop(
            &zero,
            &three,
            &one,
            &[&start],
            |_, _| vec![scalar(&graph, 1.0, data_type::INT32)],
            None
        ),
        Error::InvalidShape(_)
    );
    refused!(
        graph.for_loop(
            &zero,
            &three,
            &one,
            &[&start],
            |_, _| vec![constant(&graph, 1.0, &[3], data_type::FLOAT32)],
            None
        ),
        Error::InvalidShape(_)
    );
    refused!(
        graph.for_loop(&zero, &three, &one, &[], |_, _| Vec::new(), None),
        Error::InvalidArgument(_)
    );
    refused!(
        graph.for_loop(
            &constant(&graph, 0.0, &[2], data_type::INT32),
            &three,
            &one,
            &[&start],
            |_, arguments| vec![graph.addition(&arguments[0], &step, None).expect("add")],
            None
        ),
        Error::InvalidShape(_)
    );
    refused!(
        graph.for_loop_iterations(
            &constant(&graph, 3.0, &[2], data_type::INT32),
            &[&start],
            |_, arguments| vec![graph.addition(&arguments[0], &step, None).expect("add")],
            None
        ),
        Error::InvalidShape(_)
    );
    refused!(
        graph.for_loop_iterations(&three, &[], |_, _| Vec::new(), None),
        Error::InvalidArgument(_)
    );
    let summed = graph
        .for_loop_iterations(
            &three,
            &[&start],
            |_, arguments| vec![graph.addition(&arguments[0], &step, None).expect("add")],
            None,
        )
        .expect("valid for");
    assert_eq!(run_f32(&graph, &summed[0]), vec![3.0]);
    still_runs(&graph);
}

#[test]
fn every_needed_placeholder_must_be_fed() {
    let device = device();
    let graph = Graph::new().expect("graph");
    let unfed = floats(&graph, &[2]);
    let doubled = graph.addition(&unfed, &unfed, None).expect("addition");
    refused!(graph.run(&[], &[&doubled]), Error::MissingFeed);
    refused!(
        graph.compile(&device, &[], &[&doubled]),
        Error::MissingFeed
    );
    let predicate = scalar(&graph, 1.0, data_type::BOOL);
    let captured = graph
        .if_then_else(
            &predicate,
            || vec![graph.relu(&unfed, None).expect("relu")],
            || vec![constant(&graph, 0.0, &[2], data_type::FLOAT32)],
            None,
        )
        .expect("if");
    refused!(graph.run(&[], &[&captured[0]]), Error::MissingFeed);
    let data = TensorData::from_f32_slice(&device, &[1.0, 2.0], &[2]).expect("data");
    refused!(
        graph.run(&[Feed::new(&doubled, &data)], &[&doubled]),
        Error::InvalidArgument(_)
    );
    let results = graph
        .run(&[Feed::new(&unfed, &data)], &[&captured[0], &doubled])
        .expect("run with the feed");
    assert_eq!(results[0].read_f32().expect("read"), vec![1.0, 2.0]);
    let inside = graph
        .if_then_else(
            &predicate,
            || {
                refused!(graph.run(&[], &[&predicate]), Error::InvalidArgument(_));
                vec![constant(&graph, 1.0, &[1], data_type::FLOAT32)]
            },
            || vec![constant(&graph, 2.0, &[1], data_type::FLOAT32)],
            None,
        )
        .expect("if with a nested run");
    assert_eq!(run_f32(&graph, &inside[0]), vec![1.0]);
    refused!(
        graph.if_then_else(
            &predicate,
            || vec![constant(&graph, 1.0, &[2], data_type::FLOAT32)],
            Vec::new,
            None
        ),
        Error::InvalidShape(_)
    );
    let placeholders = graph.placeholder_tensors();
    assert_eq!(placeholders.len(), 1);
    assert_eq!(placeholders[0].as_ptr(), unfed.as_ptr());
    still_runs(&graph);
}

#[test]
fn calls_need_a_matching_callable() {
    let device = device();
    let callee_graph = Graph::new().expect("callee graph");
    let callee_input = floats(&callee_graph, &[2]);
    let callee_output = callee_graph
        .addition(&callee_input, &callee_input, None)
        .expect("callee");
    let callee = callee_graph
        .compile(
            &device,
            &[FeedDescription::new(&callee_input, &[2], data_type::FLOAT32)],
            &[&callee_output],
        )
        .expect("callee executable");

    let graph = Graph::new().expect("graph");
    let wide = floats(&graph, &[3]);
    let wide_type = ShapedType::new(Some(&[3]), data_type::FLOAT32).expect("type");
    refused!(
        graph.call(
            "double",
            &[&wide],
            &[&ShapedType::new(None, data_type::FLOAT32).expect("unranked")],
            None
        ),
        Error::InvalidArgument(_)
    );
    refused!(graph.call("double", &[&wide], &[], None), Error::InvalidArgument(_));
    let results = graph
        .call("double", &[&wide], &[&wide_type], None)
        .expect("call op");
    let data = TensorData::from_f32_slice(&device, &[1.0, 2.0, 3.0], &[3]).expect("data");
    refused!(
        graph.run(&[Feed::new(&wide, &data)], &[&results[0]]),
        Error::MissingCallable(_)
    );
    let feeds = [FeedDescription::new(&wide, &[3], data_type::FLOAT32)];
    refused!(
        graph.compile(&device, &feeds, &[&results[0]]),
        Error::MissingCallable(_)
    );
    refused!(
        graph.compile_with_descriptor(Some(&device), &feeds, &[&results[0]], None),
        Error::MissingCallable(_)
    );
    let descriptor = CompilationDescriptor::new().expect("descriptor");
    refused!(
        graph.compile_with_descriptor(Some(&device), &feeds, &[&results[0]], Some(&descriptor)),
        Error::MissingCallable(_)
    );
    descriptor
        .set_callable("double", Some(&callee))
        .expect("set callable");
    refused!(
        graph.compile_with_descriptor(Some(&device), &feeds, &[&results[0]], Some(&descriptor)),
        Error::MissingCallable(_)
    );

    let matching = Graph::new().expect("matching graph");
    let input = floats(&matching, &[2]);
    let pair_type = ShapedType::new(Some(&[2]), data_type::FLOAT32).expect("type");
    let call_results = matching
        .call("double", &[&input], &[&pair_type], None)
        .expect("call op");
    let feeds = [FeedDescription::new(&input, &[2], data_type::FLOAT32)];
    let executable = matching
        .compile_with_descriptor(Some(&device), &feeds, &[&call_results[0]], Some(&descriptor))
        .expect("compile with the callable");
    let queue = device.new_command_queue().expect("queue");
    let pair = TensorData::from_f32_slice(&device, &[1.0, 2.0], &[2]).expect("data");
    assert_eq!(
        executable.run(&queue, &[&pair]).expect("run")[0]
            .read_f32()
            .expect("read"),
        vec![2.0, 4.0]
    );
    descriptor.set_callable("double", None).expect("clear callable");
    refused!(
        matching.compile_with_descriptor(Some(&device), &feeds, &[&call_results[0]], Some(&descriptor)),
        Error::MissingCallable(_)
    );
}

#[test]
fn concat_stack_pad_and_top_k_check_their_operands() {
    let graph = Graph::new().expect("graph");
    let matrix = floats(&graph, &[2, 3]);
    let tall = floats(&graph, &[4, 3]);
    refused!(
        graph.concat_pair(&matrix, &floats(&graph, &[2, 3, 4]), 0, None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.concat_pair(&matrix, &floats(&graph, &[4, 5]), 0, None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.concat_pair(&matrix, &floats(&graph, &[4, 1]), 0, None),
        Error::InvalidShape(_)
    );
    refused!(graph.concat_pair(&matrix, &matrix, 5, None), Error::InvalidShape(_));
    refused!(graph.concat_pair(&matrix, &matrix, -3, None), Error::InvalidShape(_));
    refused!(
        graph.concat_pair(&matrix, &placeholder(&graph, &[2, 3], data_type::INT32), 0, None),
        Error::InvalidDataType(_)
    );
    refused!(graph.concat_tensors(&[], 0, false, None), Error::InvalidArgument(_));
    refused!(
        graph.concat_tensors(&[&matrix, &floats(&graph, &[2, 4])], 1, true, None),
        Error::InvalidShape(_)
    );
    assert_eq!(
        graph
            .concat_pair(&matrix, &tall, 0, None)
            .expect("concat")
            .shape(),
        Some(vec![6, 3])
    );

    refused!(
        graph.stack(&[&matrix, &floats(&graph, &[3, 2])], 0, None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.stack(&[&matrix, &floats(&graph, &[1, 3])], 0, None),
        Error::InvalidShape(_)
    );
    refused!(graph.stack(&[&matrix, &matrix], 3, None), Error::InvalidShape(_));
    refused!(graph.stack(&[&matrix, &matrix], -4, None), Error::InvalidShape(_));
    refused!(
        graph.stack(&[&matrix, &placeholder(&graph, &[2, 3], data_type::INT32)], 0, None),
        Error::InvalidDataType(_)
    );
    refused!(graph.stack(&[], 0, None), Error::InvalidArgument(_));
    assert_eq!(
        graph.stack(&[&matrix, &matrix], -1, None).expect("stack").shape(),
        Some(vec![2, 3, 2])
    );

    let constant_mode = padding_mode::CONSTANT;
    refused!(
        graph.pad(&matrix, constant_mode, &[1], &[1], 0.0, None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.pad(&matrix, constant_mode, &[1, 1, 1], &[1, 1, 1], 0.0, None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.pad(&matrix, constant_mode, &[1, 1], &[1], 0.0, None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.pad(&matrix, constant_mode, &[-5, 0], &[0, 0], 0.0, None),
        Error::InvalidShape(_)
    );
    for mode in [99, padding_mode::PERIODIC, padding_mode::ANTI_PERIODIC] {
        refused!(
            graph.pad(&matrix, mode, &[1, 1], &[1, 1], 0.0, None),
            Error::InvalidArgument(_)
        );
    }
    refused!(
        graph.pad(&matrix, padding_mode::REFLECT, &[2, 0], &[0, 0], 0.0, None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.pad(&matrix, padding_mode::SYMMETRIC, &[3, 0], &[0, 0], 0.0, None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.pad(&scalar(&graph, 1.0, data_type::FLOAT32), constant_mode, &[], &[], 0.0, None),
        Error::InvalidShape(_)
    );
    assert_eq!(
        graph
            .pad(&matrix, padding_mode::REFLECT, &[1, 2], &[1, 2], 0.0, None)
            .expect("reflect")
            .shape(),
        Some(vec![4, 7])
    );
    assert_eq!(
        graph
            .pad(&matrix, constant_mode, &[-1, 0], &[-1, 1], 0.0, None)
            .expect("crop")
            .shape(),
        Some(vec![0, 4])
    );

    refused!(graph.top_k(&matrix, 5, None), Error::InvalidShape(_));
    refused!(graph.top_k(&matrix, 0, None), Error::InvalidArgument(_));
    refused!(
        graph.top_k(&scalar(&graph, 1.0, data_type::FLOAT32), 1, None),
        Error::InvalidShape(_)
    );
    let (values, indices) = graph.top_k(&matrix, 3, None).expect("top-k");
    assert_eq!(values.shape(), Some(vec![2, 3]));
    assert_eq!(indices.data_type(), data_type::INT32);
    refused!(
        graph.top_k_gradient(&floats(&graph, &[2, 3]), &matrix, 2, None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.top_k_gradient(&floats(&graph, &[2, 5]), &matrix, 5, None),
        Error::InvalidShape(_)
    );
    let k = placeholder(&graph, &[], data_type::FLOAT32);
    refused!(
        unsafe { graph.top_k_tensor(&matrix, &k, None) },
        Error::InvalidDataType(_)
    );
    let k = constant(&graph, 2.0, &[2], data_type::INT32);
    refused!(
        unsafe { graph.top_k_tensor(&matrix, &k, None) },
        Error::InvalidShape(_)
    );
    still_runs(&graph);
}

#[test]
fn gathers_and_scatters_check_indices_and_shapes() {
    let graph = Graph::new().expect("graph");
    let updates = floats(&graph, &[3, 4]);
    let int_indices = |shape: &[usize]| placeholder(&graph, shape, data_type::INT32);
    refused!(
        graph.gather_nd(&updates, &floats(&graph, &[2, 1]), 0, None),
        Error::InvalidDataType(_)
    );
    refused!(
        graph.gather_nd(&updates, &placeholder(&graph, &[2, 1], data_type::BOOL), 0, None),
        Error::InvalidDataType(_)
    );
    refused!(graph.gather_nd(&updates, &int_indices(&[2, 5]), 0, None), Error::InvalidShape(_));
    refused!(graph.gather_nd(&updates, &int_indices(&[2, 1]), 5, None), Error::InvalidShape(_));
    refused!(graph.gather_nd(&updates, &int_indices(&[2, 1]), 1, None), Error::InvalidShape(_));
    refused!(
        graph.gather_nd(&floats(&graph, &[3]), &int_indices(&[1]), 0, None),
        Error::InvalidShape(_)
    );
    refused!(graph.gather_nd(&updates, &int_indices(&[]), 0, None), Error::InvalidShape(_));
    assert_eq!(
        graph
            .gather_nd(&updates, &int_indices(&[5, 2]), 0, None)
            .expect("gather nd")
            .shape(),
        Some(vec![5])
    );

    refused!(graph.gather(&updates, &int_indices(&[2]), 5, 0, None), Error::InvalidShape(_));
    refused!(graph.gather(&updates, &int_indices(&[2]), 1, 3, None), Error::InvalidShape(_));
    refused!(
        graph.gather(&updates, &int_indices(&[2, 2]), 1, 1, None),
        Error::InvalidShape(_)
    );
    refused!(graph.gather(&updates, &floats(&graph, &[2]), 1, 0, None), Error::InvalidDataType(_));
    refused!(
        graph.gather(&updates, &int_indices(&[3, 2]), 0, 1, None),
        Error::InvalidShape(_)
    );
    assert_eq!(
        graph
            .gather(&floats(&graph, &[2, 3, 4]), &int_indices(&[2, 5]), 1, 1, None)
            .expect("batched gather")
            .shape(),
        Some(vec![2, 5, 4])
    );

    refused!(graph.gather_along_axis(4, &updates, &int_indices(&[3, 2]), None), Error::InvalidShape(_));
    refused!(graph.gather_along_axis(-3, &updates, &int_indices(&[3, 2]), None), Error::InvalidShape(_));
    refused!(graph.gather_along_axis(1, &updates, &int_indices(&[5, 2]), None), Error::InvalidShape(_));
    refused!(
        graph.gather_along_axis(1, &updates, &int_indices(&[3, 2, 1]), None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.gather_along_axis(1, &updates, &floats(&graph, &[3, 2]), None),
        Error::InvalidDataType(_)
    );
    let float_axis = scalar(&graph, 1.0, data_type::FLOAT32);
    refused!(
        unsafe { graph.gather_along_axis_tensor(&float_axis, &updates, &int_indices(&[3, 2]), None) },
        Error::InvalidDataType(_)
    );
    let vector_axis = constant(&graph, 1.0, &[2], data_type::INT32);
    refused!(
        unsafe { graph.gather_along_axis_tensor(&vector_axis, &updates, &int_indices(&[3, 2]), None) },
        Error::InvalidShape(_)
    );

    let rows = floats(&graph, &[2, 4]);
    let add = scatter_mode::ADD;
    refused!(graph.scatter_nd(&rows, &int_indices(&[2, 5]), &[3, 4], 0, add, None), Error::InvalidShape(_));
    refused!(
        graph.scatter_nd(&floats(&graph, &[2, 7]), &int_indices(&[2, 1]), &[3, 4], 0, add, None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.scatter_nd(&rows, &floats(&graph, &[2, 1]), &[3, 4], 0, add, None),
        Error::InvalidDataType(_)
    );
    refused!(graph.scatter_nd(&rows, &int_indices(&[2, 1]), &[3, 4], 4, add, None), Error::InvalidShape(_));
    refused!(graph.scatter_nd(&rows, &int_indices(&[2, 1]), &[3, 4], 0, 99, None), Error::InvalidArgument(_));
    refused!(graph.scatter_nd(&rows, &int_indices(&[2, 1]), &[], 0, add, None), Error::InvalidShape(_));
    assert_eq!(
        graph
            .scatter_nd(&floats(&graph, &[2, 5, 4]), &int_indices(&[2, 5, 1]), &[2, 3, 4], 1, add, None)
            .expect("batched scatter nd")
            .shape(),
        Some(vec![2, 3, 4])
    );

    refused!(graph.scatter(&rows, &int_indices(&[2, 1]), &[3, 4], 0, add, None), Error::InvalidShape(_));
    refused!(graph.scatter(&rows, &int_indices(&[2]), &[3, 4], 5, add, None), Error::InvalidShape(_));
    refused!(
        graph.scatter(&floats(&graph, &[2, 5]), &int_indices(&[2]), &[3, 4], 0, add, None),
        Error::InvalidShape(_)
    );
    refused!(graph.scatter(&rows, &int_indices(&[3]), &[3, 4], 0, add, None), Error::InvalidShape(_));
    refused!(graph.scatter(&rows, &floats(&graph, &[2]), &[3, 4], 0, add, None), Error::InvalidDataType(_));
    refused!(graph.scatter(&rows, &int_indices(&[2]), &[3, 4, 5], 0, add, None), Error::InvalidShape(_));
    assert_eq!(
        graph
            .scatter(&rows, &int_indices(&[2]), &[3, 4], 0, add, None)
            .expect("scatter")
            .shape(),
        Some(vec![3, 4])
    );

    let columns = floats(&graph, &[3, 2]);
    refused!(
        graph.scatter_along_axis(4, &columns, &int_indices(&[3, 2]), &[3, 4], add, None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.scatter_along_axis(1, &columns, &int_indices(&[3, 3]), &[3, 4], add, None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.scatter_along_axis(1, &columns, &int_indices(&[3, 2]), &[5, 4], add, None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.scatter_along_axis(1, &columns, &int_indices(&[3, 2]), &[3, 4, 1], add, None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.scatter_along_axis(1, &columns, &floats(&graph, &[3, 2]), &[3, 4], add, None),
        Error::InvalidDataType(_)
    );
    assert_eq!(
        graph
            .scatter_along_axis(1, &columns, &int_indices(&[3, 2]), &[3, 4], add, None)
            .expect("scatter along axis")
            .shape(),
        Some(vec![3, 4])
    );
    still_runs(&graph);
}

#[test]
fn random_descriptors_and_ops_check_their_types() {
    let uniform = random_distribution::UNIFORM;
    let normal = random_distribution::NORMAL;
    refused!(RandomOpDescriptor::new(normal, data_type::INT32), Error::InvalidDataType(_));
    refused!(
        RandomOpDescriptor::new(random_distribution::TRUNCATED_NORMAL, data_type::INT32),
        Error::InvalidDataType(_)
    );
    refused!(RandomOpDescriptor::new(uniform, data_type::INT8), Error::InvalidDataType(_));
    refused!(RandomOpDescriptor::new(uniform, data_type::BOOL), Error::InvalidDataType(_));
    refused!(RandomOpDescriptor::new(99, data_type::FLOAT32), Error::InvalidArgument(_));
    let integers = RandomOpDescriptor::new(uniform, data_type::INT32).expect("int32 uniform");
    refused!(integers.set_distribution(normal), Error::InvalidDataType(_));
    refused!(integers.set_distribution(99), Error::InvalidArgument(_));
    let floats_descriptor = RandomOpDescriptor::new(normal, data_type::FLOAT32).expect("normal");
    refused!(floats_descriptor.set_data_type(data_type::INT32), Error::InvalidDataType(_));
    refused!(floats_descriptor.set_data_type(data_type::INT8), Error::InvalidDataType(_));
    refused!(floats_descriptor.set_sampling_method(99), Error::InvalidArgument(_));
    floats_descriptor
        .set_sampling_method(random_normal_sampling_method::BOX_MULLER)
        .expect("sampling method");

    let graph = Graph::new().expect("graph");
    let descriptor = RandomOpDescriptor::new(uniform, data_type::FLOAT32).expect("uniform");
    refused!(
        graph.random_tensor_state(&[2, 3], &descriptor, &floats(&graph, &[7]), None),
        Error::InvalidDataType(_)
    );
    refused!(
        graph.random_tensor_state(&[2, 3], &descriptor, &placeholder(&graph, &[3], data_type::INT32), None),
        Error::InvalidShape(_)
    );
    let float_shape = constant(&graph, 2.0, &[2], data_type::FLOAT32);
    refused!(
        unsafe { graph.random_tensor_shape_tensor(&float_shape, &descriptor, None) },
        Error::InvalidDataType(_)
    );
    let matrix_shape = constant(&graph, 2.0, &[2, 2], data_type::INT32);
    refused!(
        unsafe { graph.random_tensor_shape_tensor(&matrix_shape, &descriptor, None) },
        Error::InvalidShape(_)
    );
    let source = floats(&graph, &[2, 3]);
    refused!(
        graph.dropout_tensor(&source, &scalar(&graph, 1.0, data_type::INT32), None),
        Error::InvalidDataType(_)
    );
    refused!(
        graph.dropout_tensor(&source, &constant(&graph, 0.5, &[4], data_type::FLOAT32), None),
        Error::InvalidShape(_)
    );
    let state = graph.random_philox_state_seed(7, None).expect("state");
    let (random, next_state) = graph
        .random_tensor_state(&[2, 3], &descriptor, &state, None)
        .expect("random with state");
    assert_eq!(random.shape(), Some(vec![2, 3]));
    assert_eq!(next_state.shape(), Some(vec![7]));
    let values = run_f32(&graph, &random);
    assert!(values.iter().all(|value| (0.0..1.0).contains(value)), "{values:?}");
    still_runs(&graph);
}

#[test]
fn recurrent_layers_check_weights_states_and_masks() {
    let graph = Graph::new().expect("graph");
    let source = floats(&graph, &[2, 1, 3]);
    let rnn = SingleGateRNNDescriptor::new().expect("rnn descriptor");
    refused!(rnn.set_activation(99), Error::InvalidArgument(_));
    rnn.set_activation(rnn_activation::TANH).expect("activation");
    let tensor = |shape: &[usize]| floats(&graph, shape);
    let single = |recurrent: &Tensor, input: Option<&Tensor>, bias: Option<&Tensor>, state: Option<&Tensor>, mask: Option<&Tensor>| {
        graph.single_gate_rnn(&source, recurrent, input, bias, state, mask, &rnn, None)
    };
    let square = tensor(&[4, 4]);
    let input_weight = tensor(&[4, 3]);
    refused!(single(&tensor(&[4, 5]), Some(&input_weight), None, None, None), Error::InvalidShape(_));
    refused!(single(&square, Some(&tensor(&[4, 2])), None, None, None), Error::InvalidShape(_));
    refused!(single(&square, Some(&input_weight), Some(&tensor(&[5])), None, None), Error::InvalidShape(_));
    refused!(single(&square, Some(&input_weight), None, Some(&tensor(&[1, 5])), None), Error::InvalidShape(_));
    refused!(single(&square, Some(&input_weight), None, Some(&tensor(&[3, 4])), None), Error::InvalidShape(_));
    refused!(single(&square, None, None, None, None), Error::InvalidShape(_));
    refused!(single(&square, Some(&input_weight), None, None, Some(&tensor(&[2, 1, 3]))), Error::InvalidShape(_));
    refused!(
        single(&square, Some(&input_weight), Some(&placeholder(&graph, &[4], data_type::FLOAT16)), None, None),
        Error::InvalidDataType(_)
    );
    refused!(
        graph.single_gate_rnn(&tensor(&[2, 3]), &square, Some(&input_weight), None, None, None, &rnn, None),
        Error::InvalidShape(_)
    );
    let int_source = placeholder(&graph, &[2, 1, 3], data_type::INT32);
    refused!(
        graph.single_gate_rnn(&int_source, &square, None, None, None, None, &rnn, None),
        Error::InvalidDataType(_)
    );
    rnn.set_bidirectional(true).expect("bidirectional");
    refused!(single(&square, Some(&tensor(&[8, 3])), None, None, None), Error::InvalidShape(_));
    refused!(
        single(&tensor(&[2, 4, 4]), Some(&tensor(&[8, 3])), None, None, Some(&tensor(&[2, 1, 4]))),
        Error::InvalidShape(_)
    );
    let bidirectional = single(
        &tensor(&[2, 4, 4]),
        Some(&tensor(&[8, 3])),
        Some(&tensor(&[8])),
        Some(&tensor(&[1, 8])),
        Some(&tensor(&[2, 1, 8])),
    )
    .expect("bidirectional rnn");
    assert_eq!(bidirectional[0].shape(), Some(vec![2, 1, 8]));

    let lstm = LSTMDescriptor::new().expect("lstm descriptor");
    let lstm_weight = tensor(&[16, 4]);
    let lstm_input = tensor(&[16, 3]);
    let cell = tensor(&[1, 4]);
    let run_lstm = |recurrent: &Tensor, init_cell: Option<&Tensor>, peephole: Option<&Tensor>| {
        graph.lstm(&source, recurrent, Some(&lstm_input), None, None, init_cell, None, peephole, &lstm, None)
    };
    refused!(run_lstm(&tensor(&[12, 4]), None, None), Error::InvalidShape(_));
    refused!(run_lstm(&lstm_weight, Some(&tensor(&[1, 9])), None), Error::InvalidShape(_));
    refused!(run_lstm(&lstm_weight, Some(&cell), Some(&tensor(&[12]))), Error::InvalidShape(_));
    refused!(run_lstm(&lstm_weight, None, Some(&tensor(&[16]))), Error::InvalidArgument(_));
    let peephole = run_lstm(&lstm_weight, Some(&cell), Some(&tensor(&[16]))).expect("peephole lstm");
    assert_eq!(peephole[0].shape(), Some(vec![2, 1, 4]));
    lstm.set_bidirectional(true).expect("bidirectional");
    refused!(
        graph.lstm(&source, &tensor(&[2, 16, 4]), Some(&tensor(&[32, 3])), None, None, Some(&tensor(&[1, 8])), None, Some(&tensor(&[16])), &lstm, None),
        Error::Unsupported(_)
    );

    let gru = GRUDescriptor::new().expect("gru descriptor");
    let gru_weight = tensor(&[12, 4]);
    let gru_input = tensor(&[12, 3]);
    let run_gru = |recurrent: &Tensor, bias: Option<&Tensor>, secondary: Option<&Tensor>| {
        graph.gru(&source, recurrent, Some(&gru_input), bias, None, None, secondary, &gru, None)
    };
    refused!(run_gru(&tensor(&[4, 4]), None, None), Error::InvalidShape(_));
    refused!(run_gru(&gru_weight, Some(&tensor(&[5])), None), Error::InvalidShape(_));
    refused!(run_gru(&gru_weight, None, Some(&tensor(&[4]))), Error::InvalidArgument(_));
    gru.set_reset_after(true).expect("reset after");
    refused!(run_gru(&gru_weight, None, Some(&tensor(&[12]))), Error::InvalidShape(_));
    let reset_after = run_gru(&gru_weight, Some(&tensor(&[12])), Some(&tensor(&[4]))).expect("gru");
    assert_eq!(reset_after[0].shape(), Some(vec![2, 1, 4]));
    still_runs(&graph);
}

#[test]
fn graph_level_ops_check_shapes_and_data_types() {
    let graph = Graph::new().expect("graph");
    let matrix = floats(&graph, &[2, 3]);
    let ints = placeholder(&graph, &[2, 3], data_type::INT32);
    refused!(graph.constant_scalar_shaped(1.0, &[0, 3], data_type::FLOAT32), Error::InvalidShape(_));
    refused!(graph.constant_scalar_shaped(1.0, &[2], data_type::INT4), Error::UnsupportedDataType(_));
    refused!(graph.constant_scalar(1.0, 0x1234), Error::UnsupportedDataType(_));
    refused!(graph.constant_scalar(1.0, data_type::UNORM8), Error::UnsupportedDataType(_));
    refused!(graph.softmax(&ints, 0, None), Error::InvalidDataType(_));
    refused!(graph.matrix_multiplication(&ints, &ints, None), Error::InvalidDataType(_));
    refused!(
        graph.unary_arithmetic(UnaryArithmeticOp::IsNaN, &ints, None),
        Error::InvalidDataType(_)
    );
    refused!(
        graph.unary_arithmetic(
            UnaryArithmeticOp::IsInfinite,
            &placeholder(&graph, &[2], data_type::BOOL),
            None
        ),
        Error::InvalidDataType(_)
    );
    refused!(graph.reduction_sum(&scalar(&graph, 1.0, data_type::FLOAT32), &[0], None), Error::InvalidShape(_));
    let wide = floats(&graph, &[4, 3]);
    refused!(graph.select(&placeholder(&graph, &[2, 3], data_type::BOOL), &wide, &matrix, None), Error::InvalidShape(_));
    refused!(graph.select(&matrix, &matrix, &ints, None), Error::InvalidDataType(_));
    refused!(graph.select(&placeholder(&graph, &[5], data_type::BOOL), &matrix, &matrix, None), Error::InvalidShape(_));
    refused!(graph.relu_gradient(&wide, &matrix, None), Error::InvalidShape(_));
    refused!(graph.sigmoid_gradient(&wide, &matrix, None), Error::InvalidShape(_));
    refused!(graph.softmax_gradient(&wide, &matrix, 1, None), Error::InvalidShape(_));
    refused!(graph.leaky_relu_tensor(&matrix, &floats(&graph, &[4]), None), Error::InvalidShape(_));
    refused!(graph.leaky_relu_tensor(&matrix, &ints, None), Error::InvalidDataType(_));
    refused!(graph.leaky_relu_gradient(&wide, &matrix, &matrix, None), Error::InvalidShape(_));
    let mean = floats(&graph, &[3]);
    refused!(graph.normalize(&matrix, &floats(&graph, &[4]), &mean, None, None, 1e-3, None), Error::InvalidShape(_));
    refused!(
        graph.normalize(&matrix, &mean, &mean, Some(&floats(&graph, &[5])), None, 1e-3, None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.normalize(&matrix, &placeholder(&graph, &[3], data_type::FLOAT16), &mean, None, None, 1e-3, None),
        Error::InvalidDataType(_)
    );

    let convolution = |info: Convolution2DDescriptorInfo| Convolution2DDescriptor::new(info);
    refused!(
        convolution(Convolution2DDescriptorInfo { stride_in_x: 0, ..Default::default() }),
        Error::InvalidArgument(_)
    );
    refused!(
        convolution(Convolution2DDescriptorInfo { groups: 0, ..Default::default() }),
        Error::InvalidArgument(_)
    );
    refused!(
        convolution(Convolution2DDescriptorInfo { data_layout: tensor_named_data_layout::HW, ..Default::default() }),
        Error::InvalidArgument(_)
    );
    refused!(
        convolution(Convolution2DDescriptorInfo { padding_style: 9, ..Default::default() }),
        Error::InvalidArgument(_)
    );
    let same = convolution(Convolution2DDescriptorInfo {
        padding_style: padding_style::TF_SAME,
        ..Default::default()
    })
    .expect("same convolution");
    let image = floats(&graph, &[1, 4, 4, 2]);
    let weights = floats(&graph, &[3, 3, 2, 5]);
    refused!(graph.convolution2d(&floats(&graph, &[4, 4, 2]), &weights, &same, None), Error::InvalidShape(_));
    refused!(graph.convolution2d(&image, &floats(&graph, &[3, 3, 3, 5]), &same, None), Error::InvalidShape(_));
    refused!(
        graph.convolution2d(&image, &placeholder(&graph, &[3, 3, 2, 5], data_type::FLOAT16), &same, None),
        Error::InvalidDataType(_)
    );
    let grouped = convolution(Convolution2DDescriptorInfo {
        groups: 2,
        padding_style: padding_style::TF_SAME,
        ..Default::default()
    })
    .expect("grouped convolution");
    let four_channels = floats(&graph, &[1, 4, 4, 4]);
    refused!(graph.convolution2d(&four_channels, &floats(&graph, &[3, 3, 2, 5]), &grouped, None), Error::InvalidShape(_));
    refused!(graph.convolution2d(&image, &weights, &grouped, None), Error::InvalidShape(_));
    let valid = convolution(Convolution2DDescriptorInfo {
        padding_style: padding_style::TF_VALID,
        ..Default::default()
    })
    .expect("valid convolution");
    refused!(graph.convolution2d(&floats(&graph, &[1, 2, 2, 2]), &weights, &valid, None), Error::InvalidShape(_));
    assert_eq!(
        graph
            .convolution2d(&four_channels, &floats(&graph, &[3, 3, 2, 6]), &grouped, None)
            .expect("grouped convolution")
            .shape(),
        Some(vec![1, 4, 4, 6])
    );
    refused!(
        graph.convolution_transpose2d(&floats(&graph, &[1, 4, 4, 5]), &weights, &[1, 4, 4], &same, None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.convolution_transpose2d(&floats(&graph, &[1, 4, 4, 3]), &weights, &[1, 4, 4, 2], &same, None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.convolution_transpose2d(&floats(&graph, &[1, 4, 4, 5]), &weights, &[1, 4, 4, 7], &same, None),
        Error::InvalidShape(_)
    );
    refused!(
        graph.convolution_transpose2d(&floats(&graph, &[2, 4, 4, 5]), &weights, &[1, 4, 4, 2], &same, None),
        Error::InvalidShape(_)
    );
    still_runs(&graph);
}
