import Foundation
import MetalPerformanceShadersGraph

@_cdecl("mpsgraph_graph_gather_nd")
public func mpsgraph_graph_gather_nd(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ updatesTensorHandle: UnsafeMutableRawPointer?,
    _ indicesTensorHandle: UnsafeMutableRawPointer?,
    _ batchDimensions: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let updatesTensorHandle, let indicesTensorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let updatesTensor: MPSGraphTensor = mpsgraph_borrow(updatesTensorHandle)
    let indicesTensor: MPSGraphTensor = mpsgraph_borrow(indicesTensorHandle)
    return mpsgraph_retain(
        graph.gatherND(
            withUpdatesTensor: updatesTensor,
            indicesTensor: indicesTensor,
            batchDimensions: batchDimensions,
            name: mpsgraph_optional_name(name)
        )
    )
}

@_cdecl("mpsgraph_graph_gather")
public func mpsgraph_graph_gather(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ updatesTensorHandle: UnsafeMutableRawPointer?,
    _ indicesTensorHandle: UnsafeMutableRawPointer?,
    _ axis: Int,
    _ batchDimensions: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let updatesTensorHandle, let indicesTensorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let updatesTensor: MPSGraphTensor = mpsgraph_borrow(updatesTensorHandle)
    let indicesTensor: MPSGraphTensor = mpsgraph_borrow(indicesTensorHandle)
    return mpsgraph_retain(
        graph.gather(
            withUpdatesTensor: updatesTensor,
            indicesTensor: indicesTensor,
            axis: axis,
            batchDimensions: batchDimensions,
            name: mpsgraph_optional_name(name)
        )
    )
}

@_cdecl("mpsgraph_graph_gather_along_axis")
public func mpsgraph_graph_gather_along_axis(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ axis: Int,
    _ updatesTensorHandle: UnsafeMutableRawPointer?,
    _ indicesTensorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.3, *) else {
        return nil
    }
    guard let graphHandle, let updatesTensorHandle, let indicesTensorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let updatesTensor: MPSGraphTensor = mpsgraph_borrow(updatesTensorHandle)
    let indicesTensor: MPSGraphTensor = mpsgraph_borrow(indicesTensorHandle)
    return mpsgraph_retain(
        graph.gatherAlongAxis(axis, updates: updatesTensor, indices: indicesTensor, name: mpsgraph_optional_name(name))
    )
}

@_cdecl("mpsgraph_graph_gather_along_axis_tensor")
public func mpsgraph_graph_gather_along_axis_tensor(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ axisTensorHandle: UnsafeMutableRawPointer?,
    _ updatesTensorHandle: UnsafeMutableRawPointer?,
    _ indicesTensorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.3, *) else {
        return nil
    }
    guard let graphHandle, let axisTensorHandle, let updatesTensorHandle, let indicesTensorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let axisTensor: MPSGraphTensor = mpsgraph_borrow(axisTensorHandle)
    let updatesTensor: MPSGraphTensor = mpsgraph_borrow(updatesTensorHandle)
    let indicesTensor: MPSGraphTensor = mpsgraph_borrow(indicesTensorHandle)
    return mpsgraph_retain(
        graph.gatherAlongAxisTensor(axisTensor, updates: updatesTensor, indices: indicesTensor, name: mpsgraph_optional_name(name))
    )
}
