import Foundation
import MetalPerformanceShadersGraph

@inline(__always)
func mpsgraph_rnn_activation(_ rawValue: UInt) -> MPSGraphRNNActivation? {
    MPSGraphRNNActivation(rawValue: rawValue)
}

@inline(__always)
func mpsgraph_optional_tensor(_ handle: UnsafeMutableRawPointer?) -> MPSGraphTensor? {
    guard let handle else {
        return nil
    }
    let tensor: MPSGraphTensor = mpsgraph_borrow(handle)
    return tensor
}

@_cdecl("mpsgraph_single_gate_rnn_descriptor_new")
public func mpsgraph_single_gate_rnn_descriptor_new() -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.3, *) else {
        return nil
    }
    return mpsgraph_retain(MPSGraphSingleGateRNNDescriptor())
}

@_cdecl("mpsgraph_single_gate_rnn_descriptor_reverse")
public func mpsgraph_single_gate_rnn_descriptor_reverse(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphSingleGateRNNDescriptor = mpsgraph_borrow(handle)
    return descriptor.reverse
}

@_cdecl("mpsgraph_single_gate_rnn_descriptor_set_reverse")
public func mpsgraph_single_gate_rnn_descriptor_set_reverse(_ handle: UnsafeMutableRawPointer?, _ value: Bool) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphSingleGateRNNDescriptor = mpsgraph_borrow(handle)
    descriptor.reverse = value
    return true
}

@_cdecl("mpsgraph_single_gate_rnn_descriptor_bidirectional")
public func mpsgraph_single_gate_rnn_descriptor_bidirectional(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphSingleGateRNNDescriptor = mpsgraph_borrow(handle)
    return descriptor.bidirectional
}

@_cdecl("mpsgraph_single_gate_rnn_descriptor_set_bidirectional")
public func mpsgraph_single_gate_rnn_descriptor_set_bidirectional(_ handle: UnsafeMutableRawPointer?, _ value: Bool) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphSingleGateRNNDescriptor = mpsgraph_borrow(handle)
    descriptor.bidirectional = value
    return true
}

@_cdecl("mpsgraph_single_gate_rnn_descriptor_training")
public func mpsgraph_single_gate_rnn_descriptor_training(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphSingleGateRNNDescriptor = mpsgraph_borrow(handle)
    return descriptor.training
}

@_cdecl("mpsgraph_single_gate_rnn_descriptor_set_training")
public func mpsgraph_single_gate_rnn_descriptor_set_training(_ handle: UnsafeMutableRawPointer?, _ value: Bool) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphSingleGateRNNDescriptor = mpsgraph_borrow(handle)
    descriptor.training = value
    return true
}

@_cdecl("mpsgraph_single_gate_rnn_descriptor_activation")
public func mpsgraph_single_gate_rnn_descriptor_activation(_ handle: UnsafeMutableRawPointer?) -> UInt {
    guard #available(macOS 12.3, *) else {
        return 0
    }
    guard let handle else {
        return 0
    }
    let descriptor: MPSGraphSingleGateRNNDescriptor = mpsgraph_borrow(handle)
    return descriptor.activation.rawValue
}

@_cdecl("mpsgraph_single_gate_rnn_descriptor_set_activation")
public func mpsgraph_single_gate_rnn_descriptor_set_activation(_ handle: UnsafeMutableRawPointer?, _ rawValue: UInt) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle, let activation = mpsgraph_rnn_activation(rawValue) else {
        return false
    }
    let descriptor: MPSGraphSingleGateRNNDescriptor = mpsgraph_borrow(handle)
    descriptor.activation = activation
    return true
}

@_cdecl("mpsgraph_lstm_descriptor_new")
public func mpsgraph_lstm_descriptor_new() -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.3, *) else {
        return nil
    }
    return mpsgraph_retain(MPSGraphLSTMDescriptor())
}

@_cdecl("mpsgraph_lstm_descriptor_reverse")
public func mpsgraph_lstm_descriptor_reverse(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphLSTMDescriptor = mpsgraph_borrow(handle)
    return descriptor.reverse
}

@_cdecl("mpsgraph_lstm_descriptor_set_reverse")
public func mpsgraph_lstm_descriptor_set_reverse(_ handle: UnsafeMutableRawPointer?, _ value: Bool) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphLSTMDescriptor = mpsgraph_borrow(handle)
    descriptor.reverse = value
    return true
}

@_cdecl("mpsgraph_lstm_descriptor_bidirectional")
public func mpsgraph_lstm_descriptor_bidirectional(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphLSTMDescriptor = mpsgraph_borrow(handle)
    return descriptor.bidirectional
}

@_cdecl("mpsgraph_lstm_descriptor_set_bidirectional")
public func mpsgraph_lstm_descriptor_set_bidirectional(_ handle: UnsafeMutableRawPointer?, _ value: Bool) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphLSTMDescriptor = mpsgraph_borrow(handle)
    descriptor.bidirectional = value
    return true
}

@_cdecl("mpsgraph_lstm_descriptor_produce_cell")
public func mpsgraph_lstm_descriptor_produce_cell(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphLSTMDescriptor = mpsgraph_borrow(handle)
    return descriptor.produceCell
}

@_cdecl("mpsgraph_lstm_descriptor_set_produce_cell")
public func mpsgraph_lstm_descriptor_set_produce_cell(_ handle: UnsafeMutableRawPointer?, _ value: Bool) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphLSTMDescriptor = mpsgraph_borrow(handle)
    descriptor.produceCell = value
    return true
}

@_cdecl("mpsgraph_lstm_descriptor_training")
public func mpsgraph_lstm_descriptor_training(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphLSTMDescriptor = mpsgraph_borrow(handle)
    return descriptor.training
}

@_cdecl("mpsgraph_lstm_descriptor_set_training")
public func mpsgraph_lstm_descriptor_set_training(_ handle: UnsafeMutableRawPointer?, _ value: Bool) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphLSTMDescriptor = mpsgraph_borrow(handle)
    descriptor.training = value
    return true
}

@_cdecl("mpsgraph_lstm_descriptor_forget_gate_last")
public func mpsgraph_lstm_descriptor_forget_gate_last(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphLSTMDescriptor = mpsgraph_borrow(handle)
    return descriptor.forgetGateLast
}

@_cdecl("mpsgraph_lstm_descriptor_set_forget_gate_last")
public func mpsgraph_lstm_descriptor_set_forget_gate_last(_ handle: UnsafeMutableRawPointer?, _ value: Bool) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphLSTMDescriptor = mpsgraph_borrow(handle)
    descriptor.forgetGateLast = value
    return true
}

@_cdecl("mpsgraph_lstm_descriptor_input_gate_activation")
public func mpsgraph_lstm_descriptor_input_gate_activation(_ handle: UnsafeMutableRawPointer?) -> UInt {
    guard #available(macOS 12.3, *) else {
        return 0
    }
    guard let handle else {
        return 0
    }
    let descriptor: MPSGraphLSTMDescriptor = mpsgraph_borrow(handle)
    return descriptor.inputGateActivation.rawValue
}

@_cdecl("mpsgraph_lstm_descriptor_set_input_gate_activation")
public func mpsgraph_lstm_descriptor_set_input_gate_activation(_ handle: UnsafeMutableRawPointer?, _ rawValue: UInt) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle, let activation = mpsgraph_rnn_activation(rawValue) else {
        return false
    }
    let descriptor: MPSGraphLSTMDescriptor = mpsgraph_borrow(handle)
    descriptor.inputGateActivation = activation
    return true
}

@_cdecl("mpsgraph_lstm_descriptor_forget_gate_activation")
public func mpsgraph_lstm_descriptor_forget_gate_activation(_ handle: UnsafeMutableRawPointer?) -> UInt {
    guard #available(macOS 12.3, *) else {
        return 0
    }
    guard let handle else {
        return 0
    }
    let descriptor: MPSGraphLSTMDescriptor = mpsgraph_borrow(handle)
    return descriptor.forgetGateActivation.rawValue
}

@_cdecl("mpsgraph_lstm_descriptor_set_forget_gate_activation")
public func mpsgraph_lstm_descriptor_set_forget_gate_activation(_ handle: UnsafeMutableRawPointer?, _ rawValue: UInt) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle, let activation = mpsgraph_rnn_activation(rawValue) else {
        return false
    }
    let descriptor: MPSGraphLSTMDescriptor = mpsgraph_borrow(handle)
    descriptor.forgetGateActivation = activation
    return true
}

@_cdecl("mpsgraph_lstm_descriptor_cell_gate_activation")
public func mpsgraph_lstm_descriptor_cell_gate_activation(_ handle: UnsafeMutableRawPointer?) -> UInt {
    guard #available(macOS 12.3, *) else {
        return 0
    }
    guard let handle else {
        return 0
    }
    let descriptor: MPSGraphLSTMDescriptor = mpsgraph_borrow(handle)
    return descriptor.cellGateActivation.rawValue
}

@_cdecl("mpsgraph_lstm_descriptor_set_cell_gate_activation")
public func mpsgraph_lstm_descriptor_set_cell_gate_activation(_ handle: UnsafeMutableRawPointer?, _ rawValue: UInt) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle, let activation = mpsgraph_rnn_activation(rawValue) else {
        return false
    }
    let descriptor: MPSGraphLSTMDescriptor = mpsgraph_borrow(handle)
    descriptor.cellGateActivation = activation
    return true
}

@_cdecl("mpsgraph_lstm_descriptor_output_gate_activation")
public func mpsgraph_lstm_descriptor_output_gate_activation(_ handle: UnsafeMutableRawPointer?) -> UInt {
    guard #available(macOS 12.3, *) else {
        return 0
    }
    guard let handle else {
        return 0
    }
    let descriptor: MPSGraphLSTMDescriptor = mpsgraph_borrow(handle)
    return descriptor.outputGateActivation.rawValue
}

@_cdecl("mpsgraph_lstm_descriptor_set_output_gate_activation")
public func mpsgraph_lstm_descriptor_set_output_gate_activation(_ handle: UnsafeMutableRawPointer?, _ rawValue: UInt) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle, let activation = mpsgraph_rnn_activation(rawValue) else {
        return false
    }
    let descriptor: MPSGraphLSTMDescriptor = mpsgraph_borrow(handle)
    descriptor.outputGateActivation = activation
    return true
}

@_cdecl("mpsgraph_lstm_descriptor_activation")
public func mpsgraph_lstm_descriptor_activation(_ handle: UnsafeMutableRawPointer?) -> UInt {
    guard #available(macOS 12.3, *) else {
        return 0
    }
    guard let handle else {
        return 0
    }
    let descriptor: MPSGraphLSTMDescriptor = mpsgraph_borrow(handle)
    return descriptor.activation.rawValue
}

@_cdecl("mpsgraph_lstm_descriptor_set_activation")
public func mpsgraph_lstm_descriptor_set_activation(_ handle: UnsafeMutableRawPointer?, _ rawValue: UInt) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle, let activation = mpsgraph_rnn_activation(rawValue) else {
        return false
    }
    let descriptor: MPSGraphLSTMDescriptor = mpsgraph_borrow(handle)
    descriptor.activation = activation
    return true
}

@_cdecl("mpsgraph_gru_descriptor_new")
public func mpsgraph_gru_descriptor_new() -> UnsafeMutableRawPointer? {
    guard #available(macOS 13.0, *) else {
        return nil
    }
    return mpsgraph_retain(MPSGraphGRUDescriptor())
}

@_cdecl("mpsgraph_gru_descriptor_reverse")
public func mpsgraph_gru_descriptor_reverse(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard #available(macOS 13.0, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphGRUDescriptor = mpsgraph_borrow(handle)
    return descriptor.reverse
}

@_cdecl("mpsgraph_gru_descriptor_set_reverse")
public func mpsgraph_gru_descriptor_set_reverse(_ handle: UnsafeMutableRawPointer?, _ value: Bool) -> Bool {
    guard #available(macOS 13.0, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphGRUDescriptor = mpsgraph_borrow(handle)
    descriptor.reverse = value
    return true
}

@_cdecl("mpsgraph_gru_descriptor_bidirectional")
public func mpsgraph_gru_descriptor_bidirectional(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard #available(macOS 13.0, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphGRUDescriptor = mpsgraph_borrow(handle)
    return descriptor.bidirectional
}

@_cdecl("mpsgraph_gru_descriptor_set_bidirectional")
public func mpsgraph_gru_descriptor_set_bidirectional(_ handle: UnsafeMutableRawPointer?, _ value: Bool) -> Bool {
    guard #available(macOS 13.0, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphGRUDescriptor = mpsgraph_borrow(handle)
    descriptor.bidirectional = value
    return true
}

@_cdecl("mpsgraph_gru_descriptor_training")
public func mpsgraph_gru_descriptor_training(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard #available(macOS 13.0, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphGRUDescriptor = mpsgraph_borrow(handle)
    return descriptor.training
}

@_cdecl("mpsgraph_gru_descriptor_set_training")
public func mpsgraph_gru_descriptor_set_training(_ handle: UnsafeMutableRawPointer?, _ value: Bool) -> Bool {
    guard #available(macOS 13.0, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphGRUDescriptor = mpsgraph_borrow(handle)
    descriptor.training = value
    return true
}

@_cdecl("mpsgraph_gru_descriptor_reset_gate_first")
public func mpsgraph_gru_descriptor_reset_gate_first(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard #available(macOS 13.0, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphGRUDescriptor = mpsgraph_borrow(handle)
    return descriptor.resetGateFirst
}

@_cdecl("mpsgraph_gru_descriptor_set_reset_gate_first")
public func mpsgraph_gru_descriptor_set_reset_gate_first(_ handle: UnsafeMutableRawPointer?, _ value: Bool) -> Bool {
    guard #available(macOS 13.0, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphGRUDescriptor = mpsgraph_borrow(handle)
    descriptor.resetGateFirst = value
    return true
}

@_cdecl("mpsgraph_gru_descriptor_reset_after")
public func mpsgraph_gru_descriptor_reset_after(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard #available(macOS 13.0, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphGRUDescriptor = mpsgraph_borrow(handle)
    return descriptor.resetAfter
}

@_cdecl("mpsgraph_gru_descriptor_set_reset_after")
public func mpsgraph_gru_descriptor_set_reset_after(_ handle: UnsafeMutableRawPointer?, _ value: Bool) -> Bool {
    guard #available(macOS 13.0, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphGRUDescriptor = mpsgraph_borrow(handle)
    descriptor.resetAfter = value
    return true
}

@_cdecl("mpsgraph_gru_descriptor_flip_z")
public func mpsgraph_gru_descriptor_flip_z(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard #available(macOS 13.0, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphGRUDescriptor = mpsgraph_borrow(handle)
    return descriptor.flipZ
}

@_cdecl("mpsgraph_gru_descriptor_set_flip_z")
public func mpsgraph_gru_descriptor_set_flip_z(_ handle: UnsafeMutableRawPointer?, _ value: Bool) -> Bool {
    guard #available(macOS 13.0, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphGRUDescriptor = mpsgraph_borrow(handle)
    descriptor.flipZ = value
    return true
}

@_cdecl("mpsgraph_gru_descriptor_update_gate_activation")
public func mpsgraph_gru_descriptor_update_gate_activation(_ handle: UnsafeMutableRawPointer?) -> UInt {
    guard #available(macOS 13.0, *) else {
        return 0
    }
    guard let handle else {
        return 0
    }
    let descriptor: MPSGraphGRUDescriptor = mpsgraph_borrow(handle)
    return descriptor.updateGateActivation.rawValue
}

@_cdecl("mpsgraph_gru_descriptor_set_update_gate_activation")
public func mpsgraph_gru_descriptor_set_update_gate_activation(_ handle: UnsafeMutableRawPointer?, _ rawValue: UInt) -> Bool {
    guard #available(macOS 13.0, *) else {
        return false
    }
    guard let handle, let activation = mpsgraph_rnn_activation(rawValue) else {
        return false
    }
    let descriptor: MPSGraphGRUDescriptor = mpsgraph_borrow(handle)
    descriptor.updateGateActivation = activation
    return true
}

@_cdecl("mpsgraph_gru_descriptor_reset_gate_activation")
public func mpsgraph_gru_descriptor_reset_gate_activation(_ handle: UnsafeMutableRawPointer?) -> UInt {
    guard #available(macOS 13.0, *) else {
        return 0
    }
    guard let handle else {
        return 0
    }
    let descriptor: MPSGraphGRUDescriptor = mpsgraph_borrow(handle)
    return descriptor.resetGateActivation.rawValue
}

@_cdecl("mpsgraph_gru_descriptor_set_reset_gate_activation")
public func mpsgraph_gru_descriptor_set_reset_gate_activation(_ handle: UnsafeMutableRawPointer?, _ rawValue: UInt) -> Bool {
    guard #available(macOS 13.0, *) else {
        return false
    }
    guard let handle, let activation = mpsgraph_rnn_activation(rawValue) else {
        return false
    }
    let descriptor: MPSGraphGRUDescriptor = mpsgraph_borrow(handle)
    descriptor.resetGateActivation = activation
    return true
}

@_cdecl("mpsgraph_gru_descriptor_output_gate_activation")
public func mpsgraph_gru_descriptor_output_gate_activation(_ handle: UnsafeMutableRawPointer?) -> UInt {
    guard #available(macOS 13.0, *) else {
        return 0
    }
    guard let handle else {
        return 0
    }
    let descriptor: MPSGraphGRUDescriptor = mpsgraph_borrow(handle)
    return descriptor.outputGateActivation.rawValue
}

@_cdecl("mpsgraph_gru_descriptor_set_output_gate_activation")
public func mpsgraph_gru_descriptor_set_output_gate_activation(_ handle: UnsafeMutableRawPointer?, _ rawValue: UInt) -> Bool {
    guard #available(macOS 13.0, *) else {
        return false
    }
    guard let handle, let activation = mpsgraph_rnn_activation(rawValue) else {
        return false
    }
    let descriptor: MPSGraphGRUDescriptor = mpsgraph_borrow(handle)
    descriptor.outputGateActivation = activation
    return true
}

@_cdecl("mpsgraph_graph_single_gate_rnn")
public func mpsgraph_graph_single_gate_rnn(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ sourceHandle: UnsafeMutableRawPointer?,
    _ recurrentWeightHandle: UnsafeMutableRawPointer?,
    _ inputWeightHandle: UnsafeMutableRawPointer?,
    _ biasHandle: UnsafeMutableRawPointer?,
    _ initStateHandle: UnsafeMutableRawPointer?,
    _ maskHandle: UnsafeMutableRawPointer?,
    _ descriptorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.3, *) else {
        return nil
    }
    guard let graphHandle, let sourceHandle, let recurrentWeightHandle, let descriptorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let source: MPSGraphTensor = mpsgraph_borrow(sourceHandle)
    let recurrentWeight: MPSGraphTensor = mpsgraph_borrow(recurrentWeightHandle)
    let descriptor: MPSGraphSingleGateRNNDescriptor = mpsgraph_borrow(descriptorHandle)
    let result = graph.singleGateRNN(
        source,
        recurrentWeight: recurrentWeight,
        inputWeight: mpsgraph_optional_tensor(inputWeightHandle),
        bias: mpsgraph_optional_tensor(biasHandle),
        initState: mpsgraph_optional_tensor(initStateHandle),
        mask: mpsgraph_optional_tensor(maskHandle),
        descriptor: descriptor,
        name: mpsgraph_optional_name(name)
    )
    return mpsgraph_tensor_array_box(result)
}

@_cdecl("mpsgraph_graph_lstm")
public func mpsgraph_graph_lstm(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ sourceHandle: UnsafeMutableRawPointer?,
    _ recurrentWeightHandle: UnsafeMutableRawPointer?,
    _ inputWeightHandle: UnsafeMutableRawPointer?,
    _ biasHandle: UnsafeMutableRawPointer?,
    _ initStateHandle: UnsafeMutableRawPointer?,
    _ initCellHandle: UnsafeMutableRawPointer?,
    _ maskHandle: UnsafeMutableRawPointer?,
    _ peepholeHandle: UnsafeMutableRawPointer?,
    _ descriptorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.3, *) else {
        return nil
    }
    guard let graphHandle, let sourceHandle, let recurrentWeightHandle, let descriptorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let source: MPSGraphTensor = mpsgraph_borrow(sourceHandle)
    let recurrentWeight: MPSGraphTensor = mpsgraph_borrow(recurrentWeightHandle)
    let descriptor: MPSGraphLSTMDescriptor = mpsgraph_borrow(descriptorHandle)
    let result = graph.LSTM(
        source,
        recurrentWeight: recurrentWeight,
        inputWeight: mpsgraph_optional_tensor(inputWeightHandle),
        bias: mpsgraph_optional_tensor(biasHandle),
        initState: mpsgraph_optional_tensor(initStateHandle),
        initCell: mpsgraph_optional_tensor(initCellHandle),
        mask: mpsgraph_optional_tensor(maskHandle),
        peephole: mpsgraph_optional_tensor(peepholeHandle),
        descriptor: descriptor,
        name: mpsgraph_optional_name(name)
    )
    return mpsgraph_tensor_array_box(result)
}

@_cdecl("mpsgraph_graph_gru")
public func mpsgraph_graph_gru(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ sourceHandle: UnsafeMutableRawPointer?,
    _ recurrentWeightHandle: UnsafeMutableRawPointer?,
    _ inputWeightHandle: UnsafeMutableRawPointer?,
    _ biasHandle: UnsafeMutableRawPointer?,
    _ initStateHandle: UnsafeMutableRawPointer?,
    _ maskHandle: UnsafeMutableRawPointer?,
    _ secondaryBiasHandle: UnsafeMutableRawPointer?,
    _ descriptorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 13.0, *) else {
        return nil
    }
    guard let graphHandle, let sourceHandle, let recurrentWeightHandle, let descriptorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let source: MPSGraphTensor = mpsgraph_borrow(sourceHandle)
    let recurrentWeight: MPSGraphTensor = mpsgraph_borrow(recurrentWeightHandle)
    let descriptor: MPSGraphGRUDescriptor = mpsgraph_borrow(descriptorHandle)
    let result = graph.GRU(
        source,
        recurrentWeight: recurrentWeight,
        inputWeight: mpsgraph_optional_tensor(inputWeightHandle),
        bias: mpsgraph_optional_tensor(biasHandle),
        initState: mpsgraph_optional_tensor(initStateHandle),
        mask: mpsgraph_optional_tensor(maskHandle),
        secondaryBias: mpsgraph_optional_tensor(secondaryBiasHandle),
        descriptor: descriptor,
        name: mpsgraph_optional_name(name)
    )
    return mpsgraph_tensor_array_box(result)
}
