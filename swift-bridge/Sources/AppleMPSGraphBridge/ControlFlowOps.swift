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

@_cdecl("mpsgraph_graph_control_dependency")
public func mpsgraph_graph_control_dependency(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ operationHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ operationCount: Int,
    _ dependentCallback: MPSGraphRustTensorArrayCallback?,
    _ dependentContext: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
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
    let result = graph.controlDependency(with: operations, dependentBlock: {
        guard let boxHandle = dependentCallback(dependentContext) else {
            return []
        }
        return mpsgraph_take_tensor_array_box(boxHandle) ?? []
    }, name: mpsgraph_optional_name(name))
    return mpsgraph_tensor_array_box(result)
}

@_cdecl("mpsgraph_graph_if_then_else")
public func mpsgraph_graph_if_then_else(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ predicateHandle: UnsafeMutableRawPointer?,
    _ thenCallback: MPSGraphRustTensorArrayCallback?,
    _ thenContext: UnsafeMutableRawPointer?,
    _ elseCallback: MPSGraphRustTensorArrayCallback?,
    _ elseContext: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.0, *) else {
        return nil
    }
    guard let graphHandle, let predicateHandle, let thenCallback else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let predicate: MPSGraphTensor = mpsgraph_borrow(predicateHandle)
    let elseBlock: (() -> [MPSGraphTensor])? = elseCallback.map { callback in
        {
            guard let boxHandle = callback(elseContext) else {
                return []
            }
            return mpsgraph_take_tensor_array_box(boxHandle) ?? []
        }
    }
    let result = graph.`if`(predicate, then: {
        guard let boxHandle = thenCallback(thenContext) else {
            return []
        }
        return mpsgraph_take_tensor_array_box(boxHandle) ?? []
    }, else: elseBlock, name: mpsgraph_optional_name(name))
    return mpsgraph_tensor_array_box(result)
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
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
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
    let result = graph.`while`(initialInputs: initialInputs, before: { inputTensors, resultTensors in
        let inputBox = mpsgraph_tensor_array_box(inputTensors)
        var resultBox: UnsafeMutableRawPointer?
        guard let predicateHandle = beforeCallback(beforeContext, inputBox, &resultBox) else {
            fatalError("while before callback returned nil predicate")
        }
        let predicate: MPSGraphTensor = mpsgraph_borrow(predicateHandle)
        mpsgraph_object_release(predicateHandle)
        if let produced = mpsgraph_take_tensor_array_box(resultBox) {
            resultTensors.addObjects(from: produced)
        }
        return predicate
    }, after: { bodyBlockArguments in
        let inputBox = mpsgraph_tensor_array_box(bodyBlockArguments)
        guard let boxHandle = afterCallback(afterContext, inputBox) else {
            return []
        }
        return mpsgraph_take_tensor_array_box(boxHandle) ?? []
    }, name: mpsgraph_optional_name(name))
    return mpsgraph_tensor_array_box(result)
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
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.0, *) else {
        return nil
    }
    guard
        let graphHandle,
        let lowerBoundHandle,
        let upperBoundHandle,
        let stepHandle,
        let bodyCallback,
        let initialArguments = mpsgraph_tensor_array(argumentHandles, count: argumentCount)
    else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let lowerBound: MPSGraphTensor = mpsgraph_borrow(lowerBoundHandle)
    let upperBound: MPSGraphTensor = mpsgraph_borrow(upperBoundHandle)
    let step: MPSGraphTensor = mpsgraph_borrow(stepHandle)
    let result = graph.`for`(
        lowerBound: lowerBound,
        upperBound: upperBound,
        step: step,
        initialBodyArguments: initialArguments,
        body: { index, iterationArguments in
            let indexHandle = mpsgraph_retain(index)
            let inputBox = mpsgraph_tensor_array_box(iterationArguments)
            guard let boxHandle = bodyCallback(bodyContext, indexHandle, inputBox) else {
                return []
            }
            return mpsgraph_take_tensor_array_box(boxHandle) ?? []
        },
        name: mpsgraph_optional_name(name)
    )
    return mpsgraph_tensor_array_box(result)
}

@_cdecl("mpsgraph_graph_for_loop_iterations")
public func mpsgraph_graph_for_loop_iterations(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ numberOfIterationsHandle: UnsafeMutableRawPointer?,
    _ argumentHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ argumentCount: Int,
    _ bodyCallback: MPSGraphRustForBodyCallback?,
    _ bodyContext: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.0, *) else {
        return nil
    }
    guard
        let graphHandle,
        let numberOfIterationsHandle,
        let bodyCallback,
        let initialArguments = mpsgraph_tensor_array(argumentHandles, count: argumentCount)
    else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let numberOfIterations: MPSGraphTensor = mpsgraph_borrow(numberOfIterationsHandle)
    let result = graph.`for`(numberOfIterations: numberOfIterations, initialBodyArguments: initialArguments, body: { index, iterationArguments in
        let indexHandle = mpsgraph_retain(index)
        let inputBox = mpsgraph_tensor_array_box(iterationArguments)
        guard let boxHandle = bodyCallback(bodyContext, indexHandle, inputBox) else {
            return []
        }
        return mpsgraph_take_tensor_array_box(boxHandle) ?? []
    }, name: mpsgraph_optional_name(name))
    return mpsgraph_tensor_array_box(result)
}
