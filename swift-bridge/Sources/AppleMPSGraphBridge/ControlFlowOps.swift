import Foundation
import MetalPerformanceShadersGraph

public typealias MPSGraphRustTensorArrayCallback = @convention(c) (_ context: UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer?
public typealias MPSGraphRustWhileBeforeCallback = @convention(c) (
    _ context: UnsafeMutableRawPointer?,
    _ inputBoxHandle: UnsafeMutableRawPointer?,
    _ outResultBoxHandle: UnsafeMutablePointer<UnsafeMutableRawPointer?>?
) -> UnsafeMutableRawPointer?
public typealias MPSGraphRustTensorArrayInputCallback = @convention(c) (
    _ context: UnsafeMutableRawPointer?,
    _ inputBoxHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer?
public typealias MPSGraphRustForBodyCallback = @convention(c) (
    _ context: UnsafeMutableRawPointer?,
    _ indexHandle: UnsafeMutableRawPointer?,
    _ inputBoxHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer?

private struct MPSGraphTensorType: Equatable {
    let shape: [Int]?
    let dataType: MPSDataType

    init(_ tensor: MPSGraphTensor) {
        shape = tensor.shape?.map { $0.intValue }
        dataType = tensor.dataType
    }

    init(shape: [Int]?, dataType: MPSDataType) {
        self.shape = shape
        self.dataType = dataType
    }
}

private func mpsgraph_types(_ tensors: [MPSGraphTensor]) -> [MPSGraphTensorType] {
    tensors.map(MPSGraphTensorType.init)
}

private func mpsgraph_placeholders(_ graph: MPSGraph, _ types: [MPSGraphTensorType]) -> [MPSGraphTensor] {
    types.map { type in
        graph.placeholder(
            shape: type.shape.map { $0.map { NSNumber(value: $0) } },
            dataType: type.dataType,
            name: nil
        )
    }
}

private let mpsgraph_scalar_type = MPSGraphTensorType(shape: [], dataType: .float32)

@_cdecl("mpsgraph_graph_control_dependency")
public func mpsgraph_graph_control_dependency(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ operationHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ operationCount: Int,
    _ dependentCallback: MPSGraphRustTensorArrayCallback?,
    _ dependentContext: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?,
    _ outFailed: UnsafeMutablePointer<Bool>?
) -> UnsafeMutableRawPointer? {
    outFailed?.pointee = true
    guard #available(macOS 12.0, *) else {
        return nil
    }
    guard
        let graphHandle,
        let dependentCallback,
        let operations = mpsgraph_operation_array(operationHandles, count: operationCount)
    else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    var failed = false
    let result = graph.controlDependency(with: operations, dependentBlock: {
        guard let tensors = mpsgraph_take_tensor_array_box(dependentCallback(dependentContext)) else {
            failed = true
            return []
        }
        return tensors
    }, name: mpsgraph_optional_name(name))
    outFailed?.pointee = failed
    return failed ? nil : mpsgraph_tensor_array_box(result)
}

@_cdecl("mpsgraph_graph_if_then_else")
public func mpsgraph_graph_if_then_else(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ predicateHandle: UnsafeMutableRawPointer?,
    _ thenCallback: MPSGraphRustTensorArrayCallback?,
    _ thenContext: UnsafeMutableRawPointer?,
    _ elseCallback: MPSGraphRustTensorArrayCallback?,
    _ elseContext: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?,
    _ outFailed: UnsafeMutablePointer<Bool>?
) -> UnsafeMutableRawPointer? {
    outFailed?.pointee = true
    guard #available(macOS 12.0, *) else {
        return nil
    }
    guard let graphHandle, let predicateHandle, let thenCallback, let elseCallback else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let predicate: MPSGraphTensor = mpsgraph_borrow(predicateHandle)
    var failed = false
    var thenTypes = [mpsgraph_scalar_type]
    let result = graph.`if`(predicate, then: {
        guard
            let tensors = mpsgraph_take_tensor_array_box(thenCallback(thenContext)),
            !tensors.isEmpty
        else {
            failed = true
            return mpsgraph_placeholders(graph, thenTypes)
        }
        thenTypes = mpsgraph_types(tensors)
        return tensors
    }, else: {
        if !failed,
           let tensors = mpsgraph_take_tensor_array_box(elseCallback(elseContext)),
           mpsgraph_types(tensors) == thenTypes
        {
            return tensors
        }
        failed = true
        return mpsgraph_placeholders(graph, thenTypes)
    }, name: mpsgraph_optional_name(name))
    outFailed?.pointee = failed
    return failed ? nil : mpsgraph_tensor_array_box(result)
}

@_cdecl("mpsgraph_graph_while_loop")
public func mpsgraph_graph_while_loop(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ inputHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ inputCount: Int,
    _ beforeCallback: MPSGraphRustWhileBeforeCallback?,
    _ beforeContext: UnsafeMutableRawPointer?,
    _ afterCallback: MPSGraphRustTensorArrayInputCallback?,
    _ afterContext: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?,
    _ outFailed: UnsafeMutablePointer<Bool>?
) -> UnsafeMutableRawPointer? {
    outFailed?.pointee = true
    guard #available(macOS 12.0, *) else {
        return nil
    }
    guard
        let graphHandle,
        let beforeCallback,
        let afterCallback,
        let initialInputs = mpsgraph_tensor_array(inputHandles, count: inputCount)
    else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let inputTypes = mpsgraph_types(initialInputs)
    var failed = false
    let result = graph.`while`(initialInputs: initialInputs, before: { inputTensors, resultTensors in
        var resultBox: UnsafeMutableRawPointer?
        let predicateHandle = beforeCallback(beforeContext, mpsgraph_tensor_array_box(inputTensors), &resultBox)
        let results = mpsgraph_take_tensor_array_box(resultBox)
        var predicate: MPSGraphTensor?
        if let predicateHandle {
            let borrowed: MPSGraphTensor = mpsgraph_borrow(predicateHandle)
            predicate = borrowed
            mpsgraph_object_release(predicateHandle)
        }
        if let predicate, let results, !results.isEmpty,
           predicate.dataType == .bool, predicate.shape?.isEmpty == true
        {
            resultTensors.addObjects(from: results)
            return predicate
        }
        failed = true
        resultTensors.addObjects(
            from: mpsgraph_placeholders(graph, inputTypes.isEmpty ? [mpsgraph_scalar_type] : inputTypes)
        )
        return graph.placeholder(shape: [], dataType: .bool, name: nil)
    }, after: { bodyBlockArguments in
        if !failed,
           let tensors = mpsgraph_take_tensor_array_box(
               afterCallback(afterContext, mpsgraph_tensor_array_box(bodyBlockArguments))
           ),
           mpsgraph_types(tensors) == inputTypes
        {
            return tensors
        }
        failed = true
        return mpsgraph_placeholders(graph, inputTypes)
    }, name: mpsgraph_optional_name(name))
    outFailed?.pointee = failed
    return failed ? nil : mpsgraph_tensor_array_box(result)
}

private func mpsgraph_for_body(
    _ graph: MPSGraph,
    _ argumentTypes: [MPSGraphTensorType],
    _ failed: inout Bool,
    _ bodyCallback: MPSGraphRustForBodyCallback,
    _ bodyContext: UnsafeMutableRawPointer?,
    _ index: MPSGraphTensor,
    _ iterationArguments: [MPSGraphTensor]
) -> [MPSGraphTensor] {
    if !failed,
       let tensors = mpsgraph_take_tensor_array_box(
           bodyCallback(bodyContext, mpsgraph_retain(index), mpsgraph_tensor_array_box(iterationArguments))
       ),
       mpsgraph_types(tensors) == argumentTypes
    {
        return tensors
    }
    failed = true
    return mpsgraph_placeholders(graph, argumentTypes)
}

@_cdecl("mpsgraph_graph_for_loop")
public func mpsgraph_graph_for_loop(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ lowerBoundHandle: UnsafeMutableRawPointer?,
    _ upperBoundHandle: UnsafeMutableRawPointer?,
    _ stepHandle: UnsafeMutableRawPointer?,
    _ argumentHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ argumentCount: Int,
    _ bodyCallback: MPSGraphRustForBodyCallback?,
    _ bodyContext: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?,
    _ outFailed: UnsafeMutablePointer<Bool>?
) -> UnsafeMutableRawPointer? {
    outFailed?.pointee = true
    guard #available(macOS 12.0, *) else {
        return nil
    }
    guard
        let graphHandle,
        let lowerBoundHandle,
        let upperBoundHandle,
        let stepHandle,
        let bodyCallback,
        let initialArguments = mpsgraph_tensor_array(argumentHandles, count: argumentCount),
        !initialArguments.isEmpty
    else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let lowerBound: MPSGraphTensor = mpsgraph_borrow(lowerBoundHandle)
    let upperBound: MPSGraphTensor = mpsgraph_borrow(upperBoundHandle)
    let step: MPSGraphTensor = mpsgraph_borrow(stepHandle)
    let argumentTypes = mpsgraph_types(initialArguments)
    var failed = false
    let result = graph.`for`(
        lowerBound: lowerBound,
        upperBound: upperBound,
        step: step,
        initialBodyArguments: initialArguments,
        body: { index, iterationArguments in
            mpsgraph_for_body(graph, argumentTypes, &failed, bodyCallback, bodyContext, index, iterationArguments)
        },
        name: mpsgraph_optional_name(name)
    )
    outFailed?.pointee = failed
    return failed ? nil : mpsgraph_tensor_array_box(result)
}

@_cdecl("mpsgraph_graph_for_loop_iterations")
public func mpsgraph_graph_for_loop_iterations(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ numberOfIterationsHandle: UnsafeMutableRawPointer?,
    _ argumentHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ argumentCount: Int,
    _ bodyCallback: MPSGraphRustForBodyCallback?,
    _ bodyContext: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?,
    _ outFailed: UnsafeMutablePointer<Bool>?
) -> UnsafeMutableRawPointer? {
    outFailed?.pointee = true
    guard #available(macOS 12.0, *) else {
        return nil
    }
    guard
        let graphHandle,
        let numberOfIterationsHandle,
        let bodyCallback,
        let initialArguments = mpsgraph_tensor_array(argumentHandles, count: argumentCount),
        !initialArguments.isEmpty
    else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let numberOfIterations: MPSGraphTensor = mpsgraph_borrow(numberOfIterationsHandle)
    let argumentTypes = mpsgraph_types(initialArguments)
    var failed = false
    let result = graph.`for`(numberOfIterations: numberOfIterations, initialBodyArguments: initialArguments, body: { index, iterationArguments in
        mpsgraph_for_body(graph, argumentTypes, &failed, bodyCallback, bodyContext, index, iterationArguments)
    }, name: mpsgraph_optional_name(name))
    outFailed?.pointee = failed
    return failed ? nil : mpsgraph_tensor_array_box(result)
}
