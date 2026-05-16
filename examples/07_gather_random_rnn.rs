#![allow(clippy::too_many_lines)]

use apple_mpsgraph::{
    data_type, random_distribution, rnn_activation, Graph, GRUDescriptor, LSTMDescriptor,
    RandomOpDescriptor, SingleGateRNNDescriptor,
};

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>()
}

fn main() {
    let graph = Graph::new().expect("graph");
    let updates = graph
        .constant_f32_slice(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3])
        .expect("updates");
    let gather_indices = graph
        .constant_bytes(&i32_bytes(&[2, 0]), &[2], data_type::INT32)
        .expect("gather indices");
    let gather_nd_indices = graph
        .constant_bytes(&i32_bytes(&[0, 1, 1, 0]), &[2, 2], data_type::INT32)
        .expect("gather nd indices");
    let along_indices = graph
        .constant_bytes(&i32_bytes(&[2, 1, 0, 0, 1, 2]), &[2, 3], data_type::INT32)
        .expect("gather along indices");
    let axis_tensor = graph
        .constant_scalar(1.0, data_type::INT32)
        .expect("axis tensor");

    let gather = graph
        .gather(&updates, &gather_indices, 1, 0, Some("gather"))
        .expect("gather");
    let gather_nd = graph
        .gather_nd(&updates, &gather_nd_indices, 0, Some("gather_nd"))
        .expect("gather nd");
    let gather_axis = graph
        .gather_along_axis(1, &updates, &along_indices, Some("gather_axis"))
        .expect("gather along axis");
    let gather_axis_tensor = graph
        .gather_along_axis_tensor(&axis_tensor, &updates, &along_indices, Some("gather_axis_tensor"))
        .expect("gather along axis tensor");

    let descriptor = RandomOpDescriptor::new(random_distribution::UNIFORM, data_type::FLOAT32)
        .expect("random descriptor");
    descriptor.set_min(0.0).expect("random min");
    descriptor.set_max(1.0).expect("random max");
    let random = graph
        .random_tensor_seed(&[4], &descriptor, 7, Some("random"))
        .expect("random tensor");
    let dropout = graph.dropout(&updates, 1.0, Some("dropout")).expect("dropout");

    let single_gate_descriptor = SingleGateRNNDescriptor::new().expect("single gate descriptor");
    single_gate_descriptor
        .set_activation(rnn_activation::RELU)
        .expect("single gate activation");
    let single_gate_source = graph
        .constant_f32_slice(&[0.5], &[1, 1, 1])
        .expect("single gate source");
    let single_gate_recurrent = graph
        .constant_f32_slice(&[0.0], &[1, 1])
        .expect("single gate recurrent");
    let single_gate = graph
        .single_gate_rnn(
            &single_gate_source,
            &single_gate_recurrent,
            None,
            None,
            None,
            None,
            &single_gate_descriptor,
            Some("single_gate"),
        )
        .expect("single gate rnn");

    let lstm_descriptor = LSTMDescriptor::new().expect("lstm descriptor");
    lstm_descriptor
        .set_produce_cell(true)
        .expect("set produce cell");
    let lstm_source = graph
        .constant_f32_slice(&[0.0; 4], &[1, 1, 4])
        .expect("lstm source");
    let lstm_recurrent = graph
        .constant_f32_slice(&[0.0; 4], &[4, 1])
        .expect("lstm recurrent");
    let lstm = graph
        .lstm(
            &lstm_source,
            &lstm_recurrent,
            None,
            None,
            None,
            None,
            None,
            None,
            &lstm_descriptor,
            Some("lstm"),
        )
        .expect("lstm");

    let gru_descriptor = GRUDescriptor::new().expect("gru descriptor");
    gru_descriptor.set_training(true).expect("set gru training");
    gru_descriptor
        .set_reset_after(true)
        .expect("set gru reset_after");
    let gru_source = graph
        .constant_f32_slice(&[0.0; 3], &[1, 1, 3])
        .expect("gru source");
    let gru_recurrent = graph
        .constant_f32_slice(&[0.0; 3], &[3, 1])
        .expect("gru recurrent");
    let gru_secondary_bias = graph
        .constant_f32_slice(&[0.0], &[1])
        .expect("gru secondary bias");
    let gru = graph
        .gru(
            &gru_source,
            &gru_recurrent,
            None,
            None,
            None,
            None,
            Some(&gru_secondary_bias),
            &gru_descriptor,
            Some("gru"),
        )
        .expect("gru");

    let results = graph
        .run(
            &[],
            &[
                &gather,
                &gather_nd,
                &gather_axis,
                &gather_axis_tensor,
                &random,
                &dropout,
                &single_gate[0],
                &lstm[0],
                &lstm[1],
                &gru[0],
                &gru[1],
            ],
        )
        .expect("run graph");

    println!("gather: {:?}", results[0].read_f32().expect("gather"));
    println!("gather_nd: {:?}", results[1].read_f32().expect("gather_nd"));
    println!("gather_axis: {:?}", results[2].read_f32().expect("gather_axis"));
    println!(
        "gather_axis_tensor: {:?}",
        results[3].read_f32().expect("gather_axis_tensor")
    );
    println!("random: {:?}", results[4].read_f32().expect("random"));
    println!("dropout: {:?}", results[5].read_f32().expect("dropout"));
    println!("single_gate: {:?}", results[6].read_f32().expect("single_gate"));
    println!("lstm state: {:?}", results[7].read_f32().expect("lstm state"));
    println!("lstm cell: {:?}", results[8].read_f32().expect("lstm cell"));
    println!("gru state: {:?}", results[9].read_f32().expect("gru state"));
    println!("gru training: {:?}", results[10].read_f32().expect("gru training"));
}
