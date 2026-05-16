import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

@_cdecl("mpsgraph_tensor_data_new_with_bytes")
public func mpsgraph_tensor_data_new_with_bytes(
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ bytes: UnsafeRawPointer?,
    _ byteLen: Int,
    _ shape: UnsafePointer<UInt>?,
    _ shapeLen: Int,
    _ dataTypeRaw: UInt32
) -> UnsafeMutableRawPointer? {
    guard let deviceHandle, let dataType = mpsgraph_data_type(dataTypeRaw) else {
        return nil
    }
    guard byteLen == 0 || bytes != nil else {
        return nil
    }

    let tensorData = MPSGraphTensorData(
        device: mpsgraph_graph_device(deviceHandle),
        data: mpsgraph_data(bytes, byteLen),
        shape: mpsgraph_shape(shape, shapeLen),
        dataType: dataType
    )
    return mpsgraph_retain(tensorData)
}

@_cdecl("mpsgraph_tensor_data_new_with_buffer")
public func mpsgraph_tensor_data_new_with_buffer(
    _ bufferHandle: UnsafeMutableRawPointer?,
    _ shape: UnsafePointer<UInt>?,
    _ shapeLen: Int,
    _ dataTypeRaw: UInt32
) -> UnsafeMutableRawPointer? {
    guard let bufferHandle, let dataType = mpsgraph_data_type(dataTypeRaw) else {
        return nil
    }

    let buffer: MTLBuffer = mpsgraph_borrow(bufferHandle)
    let tensorData = MPSGraphTensorData(buffer, shape: mpsgraph_shape(shape, shapeLen), dataType: dataType)
    return mpsgraph_retain(tensorData)
}

@_cdecl("mpsgraph_tensor_data_data_type")
public func mpsgraph_tensor_data_data_type(_ handle: UnsafeMutableRawPointer?) -> UInt32 {
    guard let handle else {
        return 0
    }
    let tensorData: MPSGraphTensorData = mpsgraph_borrow(handle)
    return tensorData.dataType.rawValue
}

@_cdecl("mpsgraph_tensor_data_shape_len")
public func mpsgraph_tensor_data_shape_len(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let handle else {
        return 0
    }
    let tensorData: MPSGraphTensorData = mpsgraph_borrow(handle)
    return tensorData.shape.count
}

@_cdecl("mpsgraph_tensor_data_copy_shape")
public func mpsgraph_tensor_data_copy_shape(
    _ handle: UnsafeMutableRawPointer?,
    _ outShape: UnsafeMutablePointer<UInt>?
) {
    guard let handle, let outShape else {
        return
    }

    let tensorData: MPSGraphTensorData = mpsgraph_borrow(handle)
    for (index, value) in tensorData.shape.enumerated() {
        outShape[index] = UInt(truncating: value)
    }
}

@_cdecl("mpsgraph_tensor_data_read_bytes")
public func mpsgraph_tensor_data_read_bytes(
    _ handle: UnsafeMutableRawPointer?,
    _ dst: UnsafeMutableRawPointer?,
    _ dstLen: Int
) -> Bool {
    guard let handle else {
        return false
    }
    if dstLen == 0 {
        return true
    }
    guard let dst else {
        return false
    }

    let tensorData: MPSGraphTensorData = mpsgraph_borrow(handle)
    tensorData.mpsndarray().readBytes(dst, strideBytes: nil)
    return true
}
