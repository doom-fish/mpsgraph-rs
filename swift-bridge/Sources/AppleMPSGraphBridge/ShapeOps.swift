import Foundation
import MetalPerformanceShadersGraph

@_cdecl("mpsgraph_graph_concat_pair")
public func mpsgraph_graph_concat_pair(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ firstHandle: UnsafeMutableRawPointer?,
    _ secondHandle: UnsafeMutableRawPointer?,
    _ dimension: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let firstHandle, let secondHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let first: MPSGraphTensor = mpsgraph_borrow(firstHandle)
    let second: MPSGraphTensor = mpsgraph_borrow(secondHandle)
    return mpsgraph_retain(graph.concatTensor(first, with: second, dimension: dimension, name: mpsgraph_optional_name(name)))
}

@_cdecl("mpsgraph_graph_concat_tensors")
public func mpsgraph_graph_concat_tensors(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ tensorCount: Int,
    _ dimension: Int,
    _ interleave: Bool,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let tensors = mpsgraph_tensor_array(tensorHandles, count: tensorCount) else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let result = interleave
        ? graph.concatTensors(tensors, dimension: dimension, interleave: true, name: mpsgraph_optional_name(name))
        : graph.concatTensors(tensors, dimension: dimension, name: mpsgraph_optional_name(name))
    return mpsgraph_retain(result)
}

@_cdecl("mpsgraph_graph_split_sizes")
public func mpsgraph_graph_split_sizes(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ splitSizes: UnsafePointer<UInt>?,
    _ splitCount: Int,
    _ axis: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.3, *) else {
        return nil
    }
    guard let graphHandle, let tensorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let tensor: MPSGraphTensor = mpsgraph_borrow(tensorHandle)
    let result = graph.split(tensor, splitSizes: mpsgraph_shape(splitSizes, splitCount), axis: axis, name: mpsgraph_optional_name(name))
    return mpsgraph_tensor_array_box(result)
}

@_cdecl("mpsgraph_graph_split_sizes_tensor")
public func mpsgraph_graph_split_sizes_tensor(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ splitSizesTensorHandle: UnsafeMutableRawPointer?,
    _ axis: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.3, *) else {
        return nil
    }
    guard let graphHandle, let tensorHandle, let splitSizesTensorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let tensor: MPSGraphTensor = mpsgraph_borrow(tensorHandle)
    let splitSizesTensor: MPSGraphTensor = mpsgraph_borrow(splitSizesTensorHandle)
    let result = graph.split(tensor, splitSizesTensor: splitSizesTensor, axis: axis, name: mpsgraph_optional_name(name))
    return mpsgraph_tensor_array_box(result)
}

@_cdecl("mpsgraph_graph_split_num")
public func mpsgraph_graph_split_num(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ numSplits: Int,
    _ axis: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.3, *) else {
        return nil
    }
    guard let graphHandle, let tensorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let tensor: MPSGraphTensor = mpsgraph_borrow(tensorHandle)
    let result = graph.split(tensor, numSplits: numSplits, axis: axis, name: mpsgraph_optional_name(name))
    return mpsgraph_tensor_array_box(result)
}

@_cdecl("mpsgraph_graph_stack")
public func mpsgraph_graph_stack(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ tensorCount: Int,
    _ axis: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.3, *) else {
        return nil
    }
    guard let graphHandle, let tensors = mpsgraph_tensor_array(tensorHandles, count: tensorCount) else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    return mpsgraph_retain(graph.stack(tensors, axis: axis, name: mpsgraph_optional_name(name)))
}

@_cdecl("mpsgraph_graph_pad")
public func mpsgraph_graph_pad(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ paddingModeRaw: Int,
    _ leftPadding: UnsafePointer<Int>?,
    _ leftPaddingLen: Int,
    _ rightPadding: UnsafePointer<Int>?,
    _ rightPaddingLen: Int,
    _ constantValue: Double,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let tensorHandle, let paddingMode = MPSGraphPaddingMode(rawValue: paddingModeRaw) else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let tensor: MPSGraphTensor = mpsgraph_borrow(tensorHandle)
    let left = mpsgraph_optional_signed_shape(leftPadding, leftPaddingLen) ?? []
    let right = mpsgraph_optional_signed_shape(rightPadding, rightPaddingLen) ?? []
    return mpsgraph_retain(
        graph.padTensor(tensor, with: paddingMode, leftPadding: left, rightPadding: right, constantValue: constantValue, name: mpsgraph_optional_name(name))
    )
}

@_cdecl("mpsgraph_graph_top_k")
public func mpsgraph_graph_top_k(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ sourceHandle: UnsafeMutableRawPointer?,
    _ k: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.0, *) else {
        return nil
    }
    guard let graphHandle, let sourceHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let source: MPSGraphTensor = mpsgraph_borrow(sourceHandle)
    let result = graph.topK(source, k: k, name: mpsgraph_optional_name(name))
    return mpsgraph_tensor_array_box(result)
}

@_cdecl("mpsgraph_graph_top_k_tensor")
public func mpsgraph_graph_top_k_tensor(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ sourceHandle: UnsafeMutableRawPointer?,
    _ kTensorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.0, *) else {
        return nil
    }
    guard let graphHandle, let sourceHandle, let kTensorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let source: MPSGraphTensor = mpsgraph_borrow(sourceHandle)
    let kTensor: MPSGraphTensor = mpsgraph_borrow(kTensorHandle)
    let result = graph.topK(source, kTensor: kTensor, name: mpsgraph_optional_name(name))
    return mpsgraph_tensor_array_box(result)
}
