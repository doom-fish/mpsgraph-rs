import Foundation
import MetalPerformanceShadersGraph

final class MPSGraphTensorArrayBox: NSObject {
    let values: [MPSGraphTensor]

    init(_ values: [MPSGraphTensor]) {
        self.values = values
    }
}

final class MPSGraphTensorDataArrayBox: NSObject {
    let values: [MPSGraphTensorData]

    init(_ values: [MPSGraphTensorData]) {
        self.values = values
    }
}

final class MPSGraphShapedTypeArrayBox: NSObject {
    let values: [MPSGraphShapedType]

    init(_ values: [MPSGraphShapedType]) {
        self.values = values
    }
}

@inline(__always)
func mpsgraph_tensor_array_box(_ values: [MPSGraphTensor]?) -> UnsafeMutableRawPointer? {
    guard let values else {
        return nil
    }
    return mpsgraph_retain(MPSGraphTensorArrayBox(values))
}

@inline(__always)
func mpsgraph_tensor_data_array_box(_ values: [MPSGraphTensorData]?) -> UnsafeMutableRawPointer? {
    guard let values else {
        return nil
    }
    return mpsgraph_retain(MPSGraphTensorDataArrayBox(values))
}

@inline(__always)
func mpsgraph_shaped_type_array_box(_ values: [MPSGraphShapedType]?) -> UnsafeMutableRawPointer? {
    guard let values else {
        return nil
    }
    return mpsgraph_retain(MPSGraphShapedTypeArrayBox(values))
}

@_cdecl("mpsgraph_tensor_array_box_len")
public func mpsgraph_tensor_array_box_len(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let handle else {
        return 0
    }
    let box: MPSGraphTensorArrayBox = mpsgraph_borrow(handle)
    return box.values.count
}

@_cdecl("mpsgraph_tensor_array_box_get")
public func mpsgraph_tensor_array_box_get(
    _ handle: UnsafeMutableRawPointer?,
    _ index: Int
) -> UnsafeMutableRawPointer? {
    guard let handle else {
        return nil
    }
    let box: MPSGraphTensorArrayBox = mpsgraph_borrow(handle)
    guard box.values.indices.contains(index) else {
        return nil
    }
    return mpsgraph_retain(box.values[index])
}

@_cdecl("mpsgraph_tensor_data_array_box_len")
public func mpsgraph_tensor_data_array_box_len(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let handle else {
        return 0
    }
    let box: MPSGraphTensorDataArrayBox = mpsgraph_borrow(handle)
    return box.values.count
}

@_cdecl("mpsgraph_tensor_data_array_box_get")
public func mpsgraph_tensor_data_array_box_get(
    _ handle: UnsafeMutableRawPointer?,
    _ index: Int
) -> UnsafeMutableRawPointer? {
    guard let handle else {
        return nil
    }
    let box: MPSGraphTensorDataArrayBox = mpsgraph_borrow(handle)
    guard box.values.indices.contains(index) else {
        return nil
    }
    return mpsgraph_retain(box.values[index])
}

@_cdecl("mpsgraph_shaped_type_array_box_len")
public func mpsgraph_shaped_type_array_box_len(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let handle else {
        return 0
    }
    let box: MPSGraphShapedTypeArrayBox = mpsgraph_borrow(handle)
    return box.values.count
}

@_cdecl("mpsgraph_shaped_type_array_box_get")
public func mpsgraph_shaped_type_array_box_get(
    _ handle: UnsafeMutableRawPointer?,
    _ index: Int
) -> UnsafeMutableRawPointer? {
    guard let handle else {
        return nil
    }
    let box: MPSGraphShapedTypeArrayBox = mpsgraph_borrow(handle)
    guard box.values.indices.contains(index) else {
        return nil
    }
    return mpsgraph_retain(box.values[index])
}
