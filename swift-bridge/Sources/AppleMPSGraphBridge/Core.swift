import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

@inline(__always)
public func mpsgraph_retain<T: AnyObject>(_ object: T) -> UnsafeMutableRawPointer {
    Unmanaged.passRetained(object).toOpaque()
}

@inline(__always)
public func mpsgraph_release<T: AnyObject>(_ ptr: UnsafeMutableRawPointer, as _: T.Type) {
    Unmanaged<T>.fromOpaque(ptr).release()
}

@inline(__always)
public func mpsgraph_borrow<T: AnyObject>(_ ptr: UnsafeMutableRawPointer) -> T {
    Unmanaged<T>.fromOpaque(ptr).takeUnretainedValue()
}

@_cdecl("mpsgraph_object_release")
public func mpsgraph_object_release(_ ptr: UnsafeMutableRawPointer?) {
    guard let ptr else { return }
    Unmanaged<AnyObject>.fromOpaque(ptr).release()
}

@inline(__always)
func mpsgraph_optional_name(_ ptr: UnsafePointer<CChar>?) -> String? {
    ptr.map(String.init(cString:))
}

@inline(__always)
func mpsgraph_data(_ bytes: UnsafeRawPointer?, _ byteLen: Int) -> Data {
    if byteLen == 0 {
        return Data()
    }
    guard let bytes else {
        return Data()
    }
    return Data(bytes: bytes, count: byteLen)
}

@inline(__always)
func mpsgraph_optional_shape(_ shape: UnsafePointer<UInt>?, _ shapeLen: Int) -> [NSNumber]? {
    guard let shape else {
        return nil
    }
    return (0..<shapeLen).map { NSNumber(value: Int(shape[$0])) }
}

@inline(__always)
func mpsgraph_shape(_ shape: UnsafePointer<UInt>?, _ shapeLen: Int) -> [NSNumber] {
    guard let shape else {
        return []
    }
    return (0..<shapeLen).map { NSNumber(value: Int(shape[$0])) }
}

@inline(__always)
func mpsgraph_shapes(
    _ flatShapes: UnsafePointer<UInt>?,
    _ shapeLengths: UnsafePointer<UInt>?,
    count: Int
) -> [[NSNumber]]? {
    guard count == 0 || shapeLengths != nil else {
        return nil
    }

    var offset = 0
    var shapes = [[NSNumber]]()
    shapes.reserveCapacity(count)

    for index in 0..<count {
        let shapeLen = Int(shapeLengths![index])
        guard shapeLen == 0 || flatShapes != nil else {
            return nil
        }
        let shape = (0..<shapeLen).map { NSNumber(value: Int(flatShapes![offset + $0])) }
        shapes.append(shape)
        offset += shapeLen
    }

    return shapes
}

@inline(__always)
func mpsgraph_data_type(_ rawValue: UInt32) -> MPSDataType? {
    MPSDataType(rawValue: rawValue)
}

@inline(__always)
func mpsgraph_padding_style(_ rawValue: UInt) -> MPSGraphPaddingStyle? {
    MPSGraphPaddingStyle(rawValue: rawValue)
}

@inline(__always)
func mpsgraph_data_layout(_ rawValue: UInt) -> MPSGraphTensorNamedDataLayout? {
    MPSGraphTensorNamedDataLayout(rawValue: rawValue)
}

@inline(__always)
func mpsgraph_graph_device(_ deviceHandle: UnsafeMutableRawPointer) -> MPSGraphDevice {
    let device: MTLDevice = mpsgraph_borrow(deviceHandle)
    return MPSGraphDevice(mtlDevice: device)
}

@inline(__always)
func mpsgraph_optional_graph_device(_ deviceHandle: UnsafeMutableRawPointer?) -> MPSGraphDevice? {
    guard let deviceHandle else {
        return nil
    }
    return mpsgraph_graph_device(deviceHandle)
}

@inline(__always)
func mpsgraph_optional_signed_shape(_ shape: UnsafePointer<Int>?, _ shapeLen: Int) -> [NSNumber]? {
    guard let shape else {
        return nil
    }
    return (0..<shapeLen).map { NSNumber(value: shape[$0]) }
}

@available(macOS 12.0, *)
@inline(__always)
func mpsgraph_graph_type_array(
    _ handles: UnsafePointer<UnsafeMutableRawPointer?>?,
    count: Int
) -> [MPSGraphType]? {
    guard count == 0 || handles != nil else {
        return nil
    }

    var values = [MPSGraphType]()
    values.reserveCapacity(count)
    for index in 0..<count {
        guard let handle = handles![index] else {
            return nil
        }
        let value: MPSGraphType = mpsgraph_borrow(handle)
        values.append(value)
    }
    return values
}

@inline(__always)
func mpsgraph_tensor_array(
    _ handles: UnsafePointer<UnsafeMutableRawPointer?>?,
    count: Int
) -> [MPSGraphTensor]? {
    guard count == 0 || handles != nil else {
        return nil
    }

    var tensors = [MPSGraphTensor]()
    tensors.reserveCapacity(count)
    for index in 0..<count {
        guard let handle = handles![index] else {
            return nil
        }
        let tensor: MPSGraphTensor = mpsgraph_borrow(handle)
        tensors.append(tensor)
    }
    return tensors
}

@inline(__always)
func mpsgraph_tensor_data_array(
    _ handles: UnsafePointer<UnsafeMutableRawPointer?>?,
    count: Int
) -> [MPSGraphTensorData]? {
    guard count == 0 || handles != nil else {
        return nil
    }

    var tensorData = [MPSGraphTensorData]()
    tensorData.reserveCapacity(count)
    for index in 0..<count {
        guard let handle = handles![index] else {
            return nil
        }
        let value: MPSGraphTensorData = mpsgraph_borrow(handle)
        tensorData.append(value)
    }
    return tensorData
}

@inline(__always)
func mpsgraph_feed_dictionary(
    tensorHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    dataHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    count: Int
) -> [MPSGraphTensor: MPSGraphTensorData]? {
    guard
        let tensors = mpsgraph_tensor_array(tensorHandles, count: count),
        let data = mpsgraph_tensor_data_array(dataHandles, count: count)
    else {
        return nil
    }

    var feeds: [MPSGraphTensor: MPSGraphTensorData] = [:]
    feeds.reserveCapacity(count)
    for index in 0..<count {
        feeds[tensors[index]] = data[index]
    }
    return feeds
}

@inline(__always)
func mpsgraph_feed_type_dictionary(
    tensorHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    flatShapes: UnsafePointer<UInt>?,
    shapeLengths: UnsafePointer<UInt>?,
    dataTypes: UnsafePointer<UInt32>?,
    count: Int
) -> [MPSGraphTensor: MPSGraphShapedType]? {
    guard
        let tensors = mpsgraph_tensor_array(tensorHandles, count: count),
        let shapes = mpsgraph_shapes(flatShapes, shapeLengths, count: count),
        count == 0 || dataTypes != nil
    else {
        return nil
    }

    var feeds: [MPSGraphTensor: MPSGraphShapedType] = [:]
    feeds.reserveCapacity(count)
    for index in 0..<count {
        guard let dataType = mpsgraph_data_type(dataTypes![index]) else {
            return nil
        }
        feeds[tensors[index]] = MPSGraphShapedType(shape: shapes[index], dataType: dataType)
    }
    return feeds
}

@inline(__always)
func mpsgraph_write_results(
    _ results: [MPSGraphTensor: MPSGraphTensorData],
    targets: [MPSGraphTensor],
    outResults: UnsafeMutablePointer<UnsafeMutableRawPointer?>?
) -> Bool {
    guard targets.isEmpty || outResults != nil else {
        return false
    }

    for (index, target) in targets.enumerated() {
        guard let tensorData = results[target] else {
            return false
        }
        outResults![index] = mpsgraph_retain(tensorData)
    }
    return true
}

@inline(__always)
func mpsgraph_write_result_array(
    _ results: [MPSGraphTensorData],
    outResults: UnsafeMutablePointer<UnsafeMutableRawPointer?>?,
    expectedCount: Int
) -> Bool {
    guard results.count == expectedCount else {
        return false
    }
    guard expectedCount == 0 || outResults != nil else {
        return false
    }

    for (index, result) in results.enumerated() {
        outResults![index] = mpsgraph_retain(result)
    }
    return true
}
