import Foundation
import MetalPerformanceShadersGraph

@_cdecl("mpsgraph_shaped_type_new")
public func mpsgraph_shaped_type_new(
    _ shape: UnsafePointer<Int>?,
    _ shapeLen: Int,
    _ dataTypeRaw: UInt32
) -> UnsafeMutableRawPointer? {
    guard let dataType = mpsgraph_data_type(dataTypeRaw) else {
        return nil
    }
    return mpsgraph_retain(MPSGraphShapedType(shape: mpsgraph_optional_signed_shape(shape, shapeLen), dataType: dataType))
}

@_cdecl("mpsgraph_shaped_type_has_shape")
public func mpsgraph_shaped_type_has_shape(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard let handle else {
        return false
    }
    let shapedType: MPSGraphShapedType = mpsgraph_borrow(handle)
    return shapedType.shape != nil
}

@_cdecl("mpsgraph_shaped_type_shape_len")
public func mpsgraph_shaped_type_shape_len(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let handle else {
        return 0
    }
    let shapedType: MPSGraphShapedType = mpsgraph_borrow(handle)
    return shapedType.shape?.count ?? 0
}

func mpsgraph_copy_optional_shape(
    _ shape: [NSNumber]?,
    _ outShape: UnsafeMutablePointer<Int>?,
    _ outLen: Int
) -> Int {
    guard let shape else {
        return -1
    }
    guard shape.count <= outLen, let outShape else {
        return shape.count
    }
    for (index, value) in shape.enumerated() {
        outShape[index] = value.intValue
    }
    return shape.count
}

@_cdecl("mpsgraph_shaped_type_copy_shape")
public func mpsgraph_shaped_type_copy_shape(
    _ handle: UnsafeMutableRawPointer?,
    _ outShape: UnsafeMutablePointer<Int>?,
    _ outLen: Int
) -> Int {
    guard let handle else {
        return -1
    }
    let shapedType: MPSGraphShapedType = mpsgraph_borrow(handle)
    return mpsgraph_copy_optional_shape(shapedType.shape, outShape, outLen)
}

@_cdecl("mpsgraph_shaped_type_data_type")
public func mpsgraph_shaped_type_data_type(_ handle: UnsafeMutableRawPointer?) -> UInt32 {
    guard let handle else {
        return 0
    }
    let shapedType: MPSGraphShapedType = mpsgraph_borrow(handle)
    return shapedType.dataType.rawValue
}

@_cdecl("mpsgraph_shaped_type_set_shape")
public func mpsgraph_shaped_type_set_shape(
    _ handle: UnsafeMutableRawPointer?,
    _ shape: UnsafePointer<Int>?,
    _ shapeLen: Int
) -> Bool {
    guard let handle else {
        return false
    }
    let shapedType: MPSGraphShapedType = mpsgraph_borrow(handle)
    shapedType.shape = mpsgraph_optional_signed_shape(shape, shapeLen)
    return true
}

@_cdecl("mpsgraph_shaped_type_set_data_type")
public func mpsgraph_shaped_type_set_data_type(
    _ handle: UnsafeMutableRawPointer?,
    _ dataTypeRaw: UInt32
) -> Bool {
    guard let handle, let dataType = mpsgraph_data_type(dataTypeRaw) else {
        return false
    }
    let shapedType: MPSGraphShapedType = mpsgraph_borrow(handle)
    shapedType.dataType = dataType
    return true
}

@_cdecl("mpsgraph_shaped_type_is_equal")
public func mpsgraph_shaped_type_is_equal(
    _ handle: UnsafeMutableRawPointer?,
    _ otherHandle: UnsafeMutableRawPointer?
) -> Bool {
    guard let handle else {
        return false
    }
    let shapedType: MPSGraphShapedType = mpsgraph_borrow(handle)
    let other = otherHandle.map { (ptr: UnsafeMutableRawPointer) -> MPSGraphShapedType in
        mpsgraph_borrow(ptr)
    }
    return shapedType.isEqual(to: other)
}

@_cdecl("mpsgraph_tensor_has_shape")
public func mpsgraph_tensor_has_shape(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard let handle else {
        return false
    }
    let tensor: MPSGraphTensor = mpsgraph_borrow(handle)
    return tensor.shape != nil
}

@_cdecl("mpsgraph_tensor_shape_len")
public func mpsgraph_tensor_shape_len(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let handle else {
        return 0
    }
    let tensor: MPSGraphTensor = mpsgraph_borrow(handle)
    return tensor.shape?.count ?? 0
}

@_cdecl("mpsgraph_tensor_copy_shape")
public func mpsgraph_tensor_copy_shape(
    _ handle: UnsafeMutableRawPointer?,
    _ outShape: UnsafeMutablePointer<Int>?,
    _ outLen: Int
) -> Int {
    guard let handle else {
        return -1
    }
    let tensor: MPSGraphTensor = mpsgraph_borrow(handle)
    return mpsgraph_copy_optional_shape(tensor.shape, outShape, outLen)
}

@_cdecl("mpsgraph_tensor_data_type")
public func mpsgraph_tensor_data_type(_ handle: UnsafeMutableRawPointer?) -> UInt32 {
    guard let handle else {
        return 0
    }
    let tensor: MPSGraphTensor = mpsgraph_borrow(handle)
    return tensor.dataType.rawValue
}

@_cdecl("mpsgraph_tensor_operation")
public func mpsgraph_tensor_operation(_ handle: UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer? {
    guard let handle else {
        return nil
    }
    let tensor: MPSGraphTensor = mpsgraph_borrow(handle)
    return mpsgraph_retain(tensor.operation)
}
