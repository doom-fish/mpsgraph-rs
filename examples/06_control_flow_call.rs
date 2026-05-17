#![allow(clippy::too_many_lines)]

use apple_metal::MetalDevice;
use apple_mpsgraph::{
    data_type, BinaryArithmeticOp, CompilationDescriptor, FeedDescription, Graph, ShapedType,
    TensorData, UnaryArithmeticOp, WhileBeforeResult,
};

fn read_i32(data: &TensorData) -> Vec<i32> {
    let bytes = data.read_bytes().expect("read bytes");
    bytes
        .chunks_exact(core::mem::size_of::<i32>())
        .map(|chunk| i32::from_ne_bytes(chunk.try_into().expect("i32 chunk")))
        .collect()
}

fn main() {
    let device = MetalDevice::system_default().expect("no Metal device available");
    let queue = device
        .new_command_queue()
        .expect("failed to create command queue");

    let callee_graph = Graph::new().expect("callee graph");
    let callee_input = callee_graph
        .placeholder(Some(&[2]), data_type::FLOAT32, Some("callee_input"))
        .expect("callee placeholder");
    let callee_output = callee_graph
        .addition(&callee_input, &callee_input, Some("callee_double"))
        .expect("callee output");
    let callee_executable = callee_graph
        .compile(
            &device,
            &[FeedDescription::new(
                &callee_input,
                &[2],
                data_type::FLOAT32,
            )],
            &[&callee_output],
        )
        .expect("callee executable");

    let graph = Graph::new().expect("graph");
    let input = graph
        .placeholder(Some(&[2]), data_type::FLOAT32, Some("input"))
        .expect("input placeholder");
    let predicate = graph
        .placeholder(Some(&[]), data_type::BOOL, Some("predicate"))
        .expect("predicate placeholder");
    let bias = graph
        .constant_f32_slice(&[1.0, 1.0], &[2])
        .expect("bias constant");

    let output_type = ShapedType::new(Some(&[2]), data_type::FLOAT32).expect("output type");
    let call_results = graph
        .call("double", &[&input], &[&output_type], Some("call"))
        .expect("call op");
    let if_results = graph
        .if_then_else(
            &predicate,
            || vec![graph.addition(&input, &bias, None).expect("then add")],
            || vec![graph.subtraction(&input, &bias, None).expect("else sub")],
            Some("branch"),
        )
        .expect("if/then/else");

    let call_operation = call_results[0].operation().expect("call operation");
    let dependency = graph
        .control_dependency(
            &[&call_operation],
            || {
                vec![graph
                    .unary_arithmetic(UnaryArithmeticOp::Identity, &call_results[0], None)
                    .expect("identity")]
            },
            Some("dependency"),
        )
        .expect("control dependency");

    let number_of_iterations = graph
        .constant_scalar(4.0, data_type::INT32)
        .expect("iteration count");
    let zero = graph
        .constant_scalar(0.0, data_type::INT32)
        .expect("zero constant");
    let one = graph
        .constant_scalar(1.0, data_type::INT32)
        .expect("one constant");
    let limit = graph
        .constant_scalar(3.0, data_type::INT32)
        .expect("limit constant");

    let for_results = graph
        .for_loop_iterations(
            &number_of_iterations,
            &[&zero],
            |_index, args| vec![graph.addition(&args[0], &one, None).expect("for-loop add")],
            Some("for_loop"),
        )
        .expect("for loop");
    let while_results = graph
        .while_loop(
            &[&zero],
            |inputs| {
                let condition = graph
                    .binary_arithmetic(BinaryArithmeticOp::LessThan, &inputs[0], &limit, None)
                    .expect("while predicate");
                let passthrough = graph
                    .unary_arithmetic(UnaryArithmeticOp::Identity, &inputs[0], None)
                    .expect("while passthrough");
                WhileBeforeResult {
                    predicate: condition,
                    results: vec![passthrough],
                }
            },
            |inputs| vec![graph.addition(&inputs[0], &one, None).expect("while add")],
            Some("while_loop"),
        )
        .expect("while loop");

    let compile_descriptor = CompilationDescriptor::new().expect("compile descriptor");
    compile_descriptor
        .set_callable("double", Some(&callee_executable))
        .expect("set callable");
    let executable = graph
        .compile_with_descriptor(
            Some(&device),
            &[
                FeedDescription::new(&input, &[2], data_type::FLOAT32),
                FeedDescription::new(&predicate, &[], data_type::BOOL),
            ],
            &[
                &call_results[0],
                &if_results[0],
                &dependency[0],
                &for_results[0],
                &while_results[0],
            ],
            Some(&compile_descriptor),
        )
        .expect("compile executable");

    let input_data = TensorData::from_f32_slice(&device, &[3.0, 4.0], &[2]).expect("input data");
    let predicate_data =
        TensorData::from_bytes(&device, &[1_u8], &[], data_type::BOOL).expect("predicate data");
    let results = executable
        .run(&queue, &[&input_data, &predicate_data])
        .expect("run executable");

    println!(
        "call output: {:?}",
        results[0].read_f32().expect("call output")
    );
    println!("if output: {:?}", results[1].read_f32().expect("if output"));
    println!(
        "dependency output: {:?}",
        results[2].read_f32().expect("dependency output")
    );
    println!("for output: {:?}", read_i32(&results[3]));
    println!("while output: {:?}", read_i32(&results[4]));
}
