import Foundation
import MetalPerformanceShadersGraph

@_cdecl("mpsgraph_random_op_descriptor_new")
public func mpsgraph_random_op_descriptor_new(
    _ distributionRaw: UInt64,
    _ dataTypeRaw: UInt32
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.3, *) else {
        return nil
    }
    guard
        let distribution = MPSGraphRandomDistribution(rawValue: distributionRaw),
        let dataType = mpsgraph_data_type(dataTypeRaw),
        let descriptor = MPSGraphRandomOpDescriptor(distribution: distribution, dataType: dataType)
    else {
        return nil
    }
    return mpsgraph_retain(descriptor)
}

@_cdecl("mpsgraph_random_op_descriptor_distribution")
public func mpsgraph_random_op_descriptor_distribution(_ handle: UnsafeMutableRawPointer?) -> UInt64 {
    guard #available(macOS 12.3, *) else {
        return 0
    }
    guard let handle else {
        return 0
    }
    let descriptor: MPSGraphRandomOpDescriptor = mpsgraph_borrow(handle)
    return descriptor.distribution.rawValue
}

@_cdecl("mpsgraph_random_op_descriptor_set_distribution")
public func mpsgraph_random_op_descriptor_set_distribution(
    _ handle: UnsafeMutableRawPointer?,
    _ rawValue: UInt64
) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle, let distribution = MPSGraphRandomDistribution(rawValue: rawValue) else {
        return false
    }
    let descriptor: MPSGraphRandomOpDescriptor = mpsgraph_borrow(handle)
    descriptor.distribution = distribution
    return true
}

@_cdecl("mpsgraph_random_op_descriptor_data_type")
public func mpsgraph_random_op_descriptor_data_type(_ handle: UnsafeMutableRawPointer?) -> UInt32 {
    guard #available(macOS 12.3, *) else {
        return 0
    }
    guard let handle else {
        return 0
    }
    let descriptor: MPSGraphRandomOpDescriptor = mpsgraph_borrow(handle)
    return descriptor.dataType.rawValue
}

@_cdecl("mpsgraph_random_op_descriptor_set_data_type")
public func mpsgraph_random_op_descriptor_set_data_type(
    _ handle: UnsafeMutableRawPointer?,
    _ rawValue: UInt32
) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle, let dataType = mpsgraph_data_type(rawValue) else {
        return false
    }
    let descriptor: MPSGraphRandomOpDescriptor = mpsgraph_borrow(handle)
    descriptor.dataType = dataType
    return true
}

@_cdecl("mpsgraph_random_op_descriptor_min")
public func mpsgraph_random_op_descriptor_min(_ handle: UnsafeMutableRawPointer?) -> Float {
    guard #available(macOS 12.3, *) else {
        return 0
    }
    guard let handle else {
        return 0
    }
    let descriptor: MPSGraphRandomOpDescriptor = mpsgraph_borrow(handle)
    return descriptor.min
}

@_cdecl("mpsgraph_random_op_descriptor_set_min")
public func mpsgraph_random_op_descriptor_set_min(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Float
) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphRandomOpDescriptor = mpsgraph_borrow(handle)
    descriptor.min = value
    return true
}

@_cdecl("mpsgraph_random_op_descriptor_max")
public func mpsgraph_random_op_descriptor_max(_ handle: UnsafeMutableRawPointer?) -> Float {
    guard #available(macOS 12.3, *) else {
        return 0
    }
    guard let handle else {
        return 0
    }
    let descriptor: MPSGraphRandomOpDescriptor = mpsgraph_borrow(handle)
    return descriptor.max
}

@_cdecl("mpsgraph_random_op_descriptor_set_max")
public func mpsgraph_random_op_descriptor_set_max(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Float
) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphRandomOpDescriptor = mpsgraph_borrow(handle)
    descriptor.max = value
    return true
}

@_cdecl("mpsgraph_random_op_descriptor_min_integer")
public func mpsgraph_random_op_descriptor_min_integer(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard #available(macOS 12.3, *) else {
        return 0
    }
    guard let handle else {
        return 0
    }
    let descriptor: MPSGraphRandomOpDescriptor = mpsgraph_borrow(handle)
    return descriptor.minInteger
}

@_cdecl("mpsgraph_random_op_descriptor_set_min_integer")
public func mpsgraph_random_op_descriptor_set_min_integer(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Int
) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphRandomOpDescriptor = mpsgraph_borrow(handle)
    descriptor.minInteger = value
    return true
}

@_cdecl("mpsgraph_random_op_descriptor_max_integer")
public func mpsgraph_random_op_descriptor_max_integer(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard #available(macOS 12.3, *) else {
        return 0
    }
    guard let handle else {
        return 0
    }
    let descriptor: MPSGraphRandomOpDescriptor = mpsgraph_borrow(handle)
    return descriptor.maxInteger
}

@_cdecl("mpsgraph_random_op_descriptor_set_max_integer")
public func mpsgraph_random_op_descriptor_set_max_integer(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Int
) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphRandomOpDescriptor = mpsgraph_borrow(handle)
    descriptor.maxInteger = value
    return true
}

@_cdecl("mpsgraph_random_op_descriptor_mean")
public func mpsgraph_random_op_descriptor_mean(_ handle: UnsafeMutableRawPointer?) -> Float {
    guard #available(macOS 12.3, *) else {
        return 0
    }
    guard let handle else {
        return 0
    }
    let descriptor: MPSGraphRandomOpDescriptor = mpsgraph_borrow(handle)
    return descriptor.mean
}

@_cdecl("mpsgraph_random_op_descriptor_set_mean")
public func mpsgraph_random_op_descriptor_set_mean(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Float
) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphRandomOpDescriptor = mpsgraph_borrow(handle)
    descriptor.mean = value
    return true
}

@_cdecl("mpsgraph_random_op_descriptor_standard_deviation")
public func mpsgraph_random_op_descriptor_standard_deviation(_ handle: UnsafeMutableRawPointer?) -> Float {
    guard #available(macOS 12.3, *) else {
        return 0
    }
    guard let handle else {
        return 0
    }
    let descriptor: MPSGraphRandomOpDescriptor = mpsgraph_borrow(handle)
    return descriptor.standardDeviation
}

@_cdecl("mpsgraph_random_op_descriptor_set_standard_deviation")
public func mpsgraph_random_op_descriptor_set_standard_deviation(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Float
) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphRandomOpDescriptor = mpsgraph_borrow(handle)
    descriptor.standardDeviation = value
    return true
}

@_cdecl("mpsgraph_random_op_descriptor_sampling_method")
public func mpsgraph_random_op_descriptor_sampling_method(_ handle: UnsafeMutableRawPointer?) -> UInt64 {
    guard #available(macOS 12.3, *) else {
        return 0
    }
    guard let handle else {
        return 0
    }
    let descriptor: MPSGraphRandomOpDescriptor = mpsgraph_borrow(handle)
    return descriptor.samplingMethod.rawValue
}

@_cdecl("mpsgraph_random_op_descriptor_set_sampling_method")
public func mpsgraph_random_op_descriptor_set_sampling_method(
    _ handle: UnsafeMutableRawPointer?,
    _ rawValue: UInt64
) -> Bool {
    guard #available(macOS 12.3, *) else {
        return false
    }
    guard let handle, let samplingMethod = MPSGraphRandomNormalSamplingMethod(rawValue: rawValue) else {
        return false
    }
    let descriptor: MPSGraphRandomOpDescriptor = mpsgraph_borrow(handle)
    descriptor.samplingMethod = samplingMethod
    return true
}

@_cdecl("mpsgraph_graph_random_philox_state_seed")
public func mpsgraph_graph_random_philox_state_seed(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ seed: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    return mpsgraph_retain(graph.randomPhiloxStateTensor(withSeed: seed, name: mpsgraph_optional_name(name)))
}

@_cdecl("mpsgraph_graph_random_philox_state_counter")
public func mpsgraph_graph_random_philox_state_counter(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ counterLow: Int,
    _ counterHigh: Int,
    _ key: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    return mpsgraph_retain(
        graph.randomPhiloxStateTensor(
            withCounterLow: counterLow,
            counterHigh: counterHigh,
            key: key,
            name: mpsgraph_optional_name(name)
        )
    )
}

@_cdecl("mpsgraph_graph_random_tensor")
public func mpsgraph_graph_random_tensor(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ shape: UnsafePointer<UInt>?,
    _ shapeLen: Int,
    _ descriptorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.3, *) else {
        return nil
    }
    guard let graphHandle, let descriptorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let descriptor: MPSGraphRandomOpDescriptor = mpsgraph_borrow(descriptorHandle)
    return mpsgraph_retain(graph.randomTensor(withShape: mpsgraph_shape(shape, shapeLen), descriptor: descriptor, name: mpsgraph_optional_name(name)))
}

@_cdecl("mpsgraph_graph_random_tensor_shape_tensor")
public func mpsgraph_graph_random_tensor_shape_tensor(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ shapeTensorHandle: UnsafeMutableRawPointer?,
    _ descriptorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.3, *) else {
        return nil
    }
    guard let graphHandle, let shapeTensorHandle, let descriptorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let shapeTensor: MPSGraphTensor = mpsgraph_borrow(shapeTensorHandle)
    let descriptor: MPSGraphRandomOpDescriptor = mpsgraph_borrow(descriptorHandle)
    return mpsgraph_retain(graph.randomTensor(withShapeTensor: shapeTensor, descriptor: descriptor, name: mpsgraph_optional_name(name)))
}

@_cdecl("mpsgraph_graph_random_tensor_seed")
public func mpsgraph_graph_random_tensor_seed(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ shape: UnsafePointer<UInt>?,
    _ shapeLen: Int,
    _ descriptorHandle: UnsafeMutableRawPointer?,
    _ seed: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.3, *) else {
        return nil
    }
    guard let graphHandle, let descriptorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let descriptor: MPSGraphRandomOpDescriptor = mpsgraph_borrow(descriptorHandle)
    return mpsgraph_retain(
        graph.randomTensor(
            withShape: mpsgraph_shape(shape, shapeLen),
            descriptor: descriptor,
            seed: seed,
            name: mpsgraph_optional_name(name)
        )
    )
}

@_cdecl("mpsgraph_graph_random_tensor_shape_tensor_seed")
public func mpsgraph_graph_random_tensor_shape_tensor_seed(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ shapeTensorHandle: UnsafeMutableRawPointer?,
    _ descriptorHandle: UnsafeMutableRawPointer?,
    _ seed: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.3, *) else {
        return nil
    }
    guard let graphHandle, let shapeTensorHandle, let descriptorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let shapeTensor: MPSGraphTensor = mpsgraph_borrow(shapeTensorHandle)
    let descriptor: MPSGraphRandomOpDescriptor = mpsgraph_borrow(descriptorHandle)
    return mpsgraph_retain(
        graph.randomTensor(
            withShapeTensor: shapeTensor,
            descriptor: descriptor,
            seed: seed,
            name: mpsgraph_optional_name(name)
        )
    )
}

@_cdecl("mpsgraph_graph_random_tensor_state")
public func mpsgraph_graph_random_tensor_state(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ shape: UnsafePointer<UInt>?,
    _ shapeLen: Int,
    _ descriptorHandle: UnsafeMutableRawPointer?,
    _ stateHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.3, *) else {
        return nil
    }
    guard let graphHandle, let descriptorHandle, let stateHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let descriptor: MPSGraphRandomOpDescriptor = mpsgraph_borrow(descriptorHandle)
    let state: MPSGraphTensor = mpsgraph_borrow(stateHandle)
    let result = graph.randomTensor(
        withShape: mpsgraph_shape(shape, shapeLen),
        descriptor: descriptor,
        stateTensor: state,
        name: mpsgraph_optional_name(name)
    )
    return mpsgraph_tensor_array_box(result)
}

@_cdecl("mpsgraph_graph_random_tensor_shape_tensor_state")
public func mpsgraph_graph_random_tensor_shape_tensor_state(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ shapeTensorHandle: UnsafeMutableRawPointer?,
    _ descriptorHandle: UnsafeMutableRawPointer?,
    _ stateHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.3, *) else {
        return nil
    }
    guard let graphHandle, let shapeTensorHandle, let descriptorHandle, let stateHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let shapeTensor: MPSGraphTensor = mpsgraph_borrow(shapeTensorHandle)
    let descriptor: MPSGraphRandomOpDescriptor = mpsgraph_borrow(descriptorHandle)
    let state: MPSGraphTensor = mpsgraph_borrow(stateHandle)
    let result = graph.randomTensor(
        withShapeTensor: shapeTensor,
        descriptor: descriptor,
        stateTensor: state,
        name: mpsgraph_optional_name(name)
    )
    return mpsgraph_tensor_array_box(result)
}

@_cdecl("mpsgraph_graph_dropout")
public func mpsgraph_graph_dropout(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ rate: Double,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let tensorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let tensor: MPSGraphTensor = mpsgraph_borrow(tensorHandle)
    return mpsgraph_retain(graph.dropout(tensor, rate: rate, name: mpsgraph_optional_name(name)))
}

@_cdecl("mpsgraph_graph_dropout_tensor")
public func mpsgraph_graph_dropout_tensor(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ rateTensorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let tensorHandle, let rateTensorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let tensor: MPSGraphTensor = mpsgraph_borrow(tensorHandle)
    let rateTensor: MPSGraphTensor = mpsgraph_borrow(rateTensorHandle)
    return mpsgraph_retain(graph.dropout(tensor, rate: rateTensor, name: mpsgraph_optional_name(name)))
}
