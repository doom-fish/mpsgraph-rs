import Foundation
import MetalPerformanceShadersGraph

@_cdecl("mpsgraph_graph_call_symbol")
public func mpsgraph_graph_call_symbol(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ symbolName: UnsafePointer<CChar>?,
    _ inputHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ inputCount: Int,
    _ outputTypeHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ outputTypeCount: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 15.0, *) else {
        return nil
    }
    guard
        let graphHandle,
        let symbolName,
        let inputTensors = mpsgraph_tensor_array(inputHandles, count: inputCount),
        let outputTypes = mpsgraph_graph_type_array(outputTypeHandles, count: outputTypeCount)
    else {
        return nil
    }

    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let result = graph.call(
        symbolName: String(cString: symbolName),
        inputTensors: inputTensors,
        outputTypes: outputTypes,
        name: mpsgraph_optional_name(name)
    )
    return mpsgraph_tensor_array_box(result)
}
