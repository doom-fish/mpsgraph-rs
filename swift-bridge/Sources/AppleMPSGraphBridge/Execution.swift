import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

@_cdecl("mpsgraph_graph_options")
public func mpsgraph_graph_options(_ handle: UnsafeMutableRawPointer?) -> UInt64 {
    guard let handle else {
        return 0
    }
    let graph: MPSGraph = mpsgraph_borrow(handle)
    return graph.options.rawValue
}

@_cdecl("mpsgraph_graph_set_options")
public func mpsgraph_graph_set_options(_ handle: UnsafeMutableRawPointer?, _ rawValue: UInt64) -> Bool {
    guard let handle, let options = MPSGraphOptions(rawValue: rawValue) else {
        return false
    }
    let graph: MPSGraph = mpsgraph_borrow(handle)
    graph.options = options
    return true
}

@_cdecl("mpsgraph_graph_placeholder_tensors")
public func mpsgraph_graph_placeholder_tensors(_ handle: UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer? {
    guard let handle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(handle)
    return mpsgraph_tensor_array_box(graph.placeholderTensors)
}

@_cdecl("mpsgraph_compilation_descriptor_new")
public func mpsgraph_compilation_descriptor_new() -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.0, *) else {
        return nil
    }

    return mpsgraph_retain(MPSGraphCompilationDescriptor())
}

@_cdecl("mpsgraph_compilation_descriptor_disable_type_inference")
public func mpsgraph_compilation_descriptor_disable_type_inference(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard #available(macOS 12.0, *) else {
            return false
        }

    guard let handle else {
        return false
    }
    let descriptor: MPSGraphCompilationDescriptor = mpsgraph_borrow(handle)
    descriptor.disableTypeInference()
    return true
}

@_cdecl("mpsgraph_compilation_descriptor_optimization_level")
public func mpsgraph_compilation_descriptor_optimization_level(_ handle: UnsafeMutableRawPointer?) -> UInt64 {
    guard #available(macOS 12.3, *) else {
            return 0
        }

    guard let handle else {
        return 0
    }
    let descriptor: MPSGraphCompilationDescriptor = mpsgraph_borrow(handle)
    return descriptor.optimizationLevel.rawValue
}

@_cdecl("mpsgraph_compilation_descriptor_set_optimization_level")
public func mpsgraph_compilation_descriptor_set_optimization_level(
    _ handle: UnsafeMutableRawPointer?,
    _ rawValue: UInt64
) -> Bool {
    guard #available(macOS 12.3, *) else {
            return false
        }

    guard let handle, let value = MPSGraphOptimization(rawValue: rawValue) else {
        return false
    }
    let descriptor: MPSGraphCompilationDescriptor = mpsgraph_borrow(handle)
    descriptor.optimizationLevel = value
    return true
}

@_cdecl("mpsgraph_compilation_descriptor_wait_for_completion")
public func mpsgraph_compilation_descriptor_wait_for_completion(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard #available(macOS 13.0, *) else {
            return false
        }

    guard let handle else {
        return false
    }
    let descriptor: MPSGraphCompilationDescriptor = mpsgraph_borrow(handle)
    return descriptor.waitForCompilationCompletion
}

@_cdecl("mpsgraph_compilation_descriptor_set_wait_for_completion")
public func mpsgraph_compilation_descriptor_set_wait_for_completion(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Bool
) -> Bool {
    guard #available(macOS 13.0, *) else {
            return false
        }

    guard let handle else {
        return false
    }
    let descriptor: MPSGraphCompilationDescriptor = mpsgraph_borrow(handle)
    descriptor.waitForCompilationCompletion = value
    return true
}

@_cdecl("mpsgraph_compilation_descriptor_optimization_profile")
public func mpsgraph_compilation_descriptor_optimization_profile(_ handle: UnsafeMutableRawPointer?) -> UInt64 {
    guard #available(macOS 12.3, *) else {
            return 0
        }

    guard let handle else {
        return 0
    }
    let descriptor: MPSGraphCompilationDescriptor = mpsgraph_borrow(handle)
    return descriptor.optimizationProfile.rawValue
}

@_cdecl("mpsgraph_compilation_descriptor_set_optimization_profile")
public func mpsgraph_compilation_descriptor_set_optimization_profile(
    _ handle: UnsafeMutableRawPointer?,
    _ rawValue: UInt64
) -> Bool {
    guard #available(macOS 12.3, *) else {
            return false
        }

    guard let handle, let value = MPSGraphOptimizationProfile(rawValue: rawValue) else {
        return false
    }
    let descriptor: MPSGraphCompilationDescriptor = mpsgraph_borrow(handle)
    descriptor.optimizationProfile = value
    return true
}

@_cdecl("mpsgraph_compilation_descriptor_reduced_precision_fast_math")
public func mpsgraph_compilation_descriptor_reduced_precision_fast_math(_ handle: UnsafeMutableRawPointer?) -> UInt {
    guard #available(macOS 26.0, *) else {
            return 0
        }

    guard let handle else {
        return 0
    }
    let descriptor: MPSGraphCompilationDescriptor = mpsgraph_borrow(handle)
    return descriptor.reducedPrecisionFastMath.rawValue
}

@_cdecl("mpsgraph_compilation_descriptor_set_reduced_precision_fast_math")
public func mpsgraph_compilation_descriptor_set_reduced_precision_fast_math(
    _ handle: UnsafeMutableRawPointer?,
    _ rawValue: UInt
) -> Bool {
    guard #available(macOS 26.0, *) else {
            return false
        }

    guard let handle else {
        return false
    }
    let descriptor: MPSGraphCompilationDescriptor = mpsgraph_borrow(handle)
    descriptor.reducedPrecisionFastMath = MPSGraphReducedPrecisionFastMath(rawValue: rawValue)
    return true
}

@_cdecl("mpsgraph_execution_descriptor_new")
public func mpsgraph_execution_descriptor_new() -> UnsafeMutableRawPointer? {
    mpsgraph_retain(MPSGraphExecutionDescriptor())
}

@_cdecl("mpsgraph_execution_descriptor_wait_until_completed")
public func mpsgraph_execution_descriptor_wait_until_completed(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphExecutionDescriptor = mpsgraph_borrow(handle)
    return descriptor.waitUntilCompleted
}

@_cdecl("mpsgraph_execution_descriptor_set_wait_until_completed")
public func mpsgraph_execution_descriptor_set_wait_until_completed(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Bool
) -> Bool {
    guard let handle else {
        return false
    }
    let descriptor: MPSGraphExecutionDescriptor = mpsgraph_borrow(handle)
    descriptor.waitUntilCompleted = value
    return true
}

@_cdecl("mpsgraph_execution_descriptor_compilation_descriptor")
public func mpsgraph_execution_descriptor_compilation_descriptor(
    _ handle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.3, *) else {
            return nil
        }

    guard let handle else {
        return nil
    }
    let descriptor: MPSGraphExecutionDescriptor = mpsgraph_borrow(handle)
    guard let compilationDescriptor = descriptor.compilationDescriptor else {
        return nil
    }
    return mpsgraph_retain(compilationDescriptor)
}

@_cdecl("mpsgraph_execution_descriptor_set_compilation_descriptor")
public func mpsgraph_execution_descriptor_set_compilation_descriptor(
    _ handle: UnsafeMutableRawPointer?,
    _ compilationDescriptorHandle: UnsafeMutableRawPointer?
) -> Bool {
    guard #available(macOS 12.3, *) else {
            return false
        }

    guard let handle else {
        return false
    }
    let descriptor: MPSGraphExecutionDescriptor = mpsgraph_borrow(handle)
    descriptor.compilationDescriptor = compilationDescriptorHandle.map { ptr in
        let compilationDescriptor: MPSGraphCompilationDescriptor = mpsgraph_borrow(ptr)
        return compilationDescriptor
    }
    return true
}

@_cdecl("mpsgraph_executable_execution_descriptor_new")
public func mpsgraph_executable_execution_descriptor_new() -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.0, *) else {
        return nil
    }

    return mpsgraph_retain(MPSGraphExecutableExecutionDescriptor())
}

@_cdecl("mpsgraph_executable_execution_descriptor_wait_until_completed")
public func mpsgraph_executable_execution_descriptor_wait_until_completed(
    _ handle: UnsafeMutableRawPointer?
) -> Bool {
    guard #available(macOS 12.0, *) else {
            return false
        }

    guard let handle else {
        return false
    }
    let descriptor: MPSGraphExecutableExecutionDescriptor = mpsgraph_borrow(handle)
    return descriptor.waitUntilCompleted
}

@_cdecl("mpsgraph_executable_execution_descriptor_set_wait_until_completed")
public func mpsgraph_executable_execution_descriptor_set_wait_until_completed(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Bool
) -> Bool {
    guard #available(macOS 12.0, *) else {
            return false
        }

    guard let handle else {
        return false
    }
    let descriptor: MPSGraphExecutableExecutionDescriptor = mpsgraph_borrow(handle)
    descriptor.waitUntilCompleted = value
    return true
}

@_cdecl("mpsgraph_executable_serialization_descriptor_new")
public func mpsgraph_executable_serialization_descriptor_new() -> UnsafeMutableRawPointer? {
    guard #available(macOS 14.0, *) else {
        return nil
    }

    return mpsgraph_retain(MPSGraphExecutableSerializationDescriptor())
}

@_cdecl("mpsgraph_executable_serialization_descriptor_append")
public func mpsgraph_executable_serialization_descriptor_append(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard #available(macOS 14.0, *) else {
            return false
        }

    guard let handle else {
        return false
    }
    let descriptor: MPSGraphExecutableSerializationDescriptor = mpsgraph_borrow(handle)
    return descriptor.append
}

@_cdecl("mpsgraph_executable_serialization_descriptor_set_append")
public func mpsgraph_executable_serialization_descriptor_set_append(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Bool
) -> Bool {
    guard #available(macOS 14.0, *) else {
            return false
        }

    guard let handle else {
        return false
    }
    let descriptor: MPSGraphExecutableSerializationDescriptor = mpsgraph_borrow(handle)
    descriptor.append = value
    return true
}

@_cdecl("mpsgraph_executable_serialization_descriptor_deployment_platform")
public func mpsgraph_executable_serialization_descriptor_deployment_platform(
    _ handle: UnsafeMutableRawPointer?
) -> UInt64 {
    guard #available(macOS 14.0, *) else {
            return 0
        }

    guard let handle else {
        return 0
    }
    let descriptor: MPSGraphExecutableSerializationDescriptor = mpsgraph_borrow(handle)
    return descriptor.deploymentPlatform.rawValue
}

@_cdecl("mpsgraph_executable_serialization_descriptor_set_deployment_platform")
public func mpsgraph_executable_serialization_descriptor_set_deployment_platform(
    _ handle: UnsafeMutableRawPointer?,
    _ rawValue: UInt64
) -> Bool {
    guard #available(macOS 14.0, *) else {
            return false
        }

    guard let handle, let value = MPSGraphDeploymentPlatform(rawValue: rawValue) else {
        return false
    }
    let descriptor: MPSGraphExecutableSerializationDescriptor = mpsgraph_borrow(handle)
    descriptor.deploymentPlatform = value
    return true
}

@_cdecl("mpsgraph_executable_serialization_descriptor_minimum_deployment_target_len")
public func mpsgraph_executable_serialization_descriptor_minimum_deployment_target_len(
    _ handle: UnsafeMutableRawPointer?
) -> Int {
    guard #available(macOS 14.0, *) else {
            return 0
        }

    guard let handle else {
        return 0
    }
    let descriptor: MPSGraphExecutableSerializationDescriptor = mpsgraph_borrow(handle)
    return descriptor.minimumDeploymentTarget.utf8.count
}

@_cdecl("mpsgraph_executable_serialization_descriptor_copy_minimum_deployment_target")
public func mpsgraph_executable_serialization_descriptor_copy_minimum_deployment_target(
    _ handle: UnsafeMutableRawPointer?,
    _ outBytes: UnsafeMutablePointer<UInt8>?,
    _ outLen: Int
) -> Bool {
    guard #available(macOS 14.0, *) else {
            return false
        }

    guard let handle else {
        return false
    }
    guard let outBytes else {
        return outLen == 0
    }
    let descriptor: MPSGraphExecutableSerializationDescriptor = mpsgraph_borrow(handle)
    let bytes = Array(descriptor.minimumDeploymentTarget.utf8)
    guard bytes.count == outLen else {
        return false
    }
    for (index, value) in bytes.enumerated() {
        outBytes[index] = value
    }
    return true
}

@_cdecl("mpsgraph_executable_serialization_descriptor_set_minimum_deployment_target")
public func mpsgraph_executable_serialization_descriptor_set_minimum_deployment_target(
    _ handle: UnsafeMutableRawPointer?,
    _ value: UnsafePointer<CChar>?
) -> Bool {
    guard #available(macOS 14.0, *) else {
            return false
        }

    guard let handle, let value else {
        return false
    }
    let descriptor: MPSGraphExecutableSerializationDescriptor = mpsgraph_borrow(handle)
    descriptor.minimumDeploymentTarget = String(cString: value)
    return true
}

@_cdecl("mpsgraph_graph_compile_with_descriptor")
public func mpsgraph_graph_compile_with_descriptor(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ feedTensorHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ feedCount: Int,
    _ flatShapes: UnsafePointer<UInt>?,
    _ shapeLengths: UnsafePointer<UInt>?,
    _ dataTypes: UnsafePointer<UInt32>?,
    _ targetTensorHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ targetCount: Int,
    _ compilationDescriptorHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.0, *) else {
            return nil
        }

    guard let graphHandle else {
        return nil
    }
    guard
        let feeds = mpsgraph_feed_type_dictionary(
            tensorHandles: feedTensorHandles,
            flatShapes: flatShapes,
            shapeLengths: shapeLengths,
            dataTypes: dataTypes,
            count: feedCount
        ),
        let targets = mpsgraph_tensor_array(targetTensorHandles, count: targetCount)
    else {
        return nil
    }

    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let device = mpsgraph_optional_graph_device(deviceHandle)
    let compilationDescriptor = compilationDescriptorHandle.map { ptr in
        let descriptor: MPSGraphCompilationDescriptor = mpsgraph_borrow(ptr)
        return descriptor
    }
    let executable = graph.compile(
        with: device,
        feeds: feeds,
        targetTensors: targets,
        targetOperations: nil,
        compilationDescriptor: compilationDescriptor
    )
    return mpsgraph_retain(executable)
}

@_cdecl("mpsgraph_executable_options")
public func mpsgraph_executable_options(_ handle: UnsafeMutableRawPointer?) -> UInt64 {
    guard #available(macOS 12.0, *) else {
            return 0
        }

    guard let handle else {
        return 0
    }
    let executable: MPSGraphExecutable = mpsgraph_borrow(handle)
    return executable.options.rawValue
}

@_cdecl("mpsgraph_executable_set_options")
public func mpsgraph_executable_set_options(_ handle: UnsafeMutableRawPointer?, _ rawValue: UInt64) -> Bool {
    guard #available(macOS 12.0, *) else {
            return false
        }

    guard let handle, let options = MPSGraphOptions(rawValue: rawValue) else {
        return false
    }
    let executable: MPSGraphExecutable = mpsgraph_borrow(handle)
    executable.options = options
    return true
}

@_cdecl("mpsgraph_executable_feed_tensors")
public func mpsgraph_executable_feed_tensors(_ handle: UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.0, *) else {
            return nil
        }

    guard let handle else {
        return nil
    }
    let executable: MPSGraphExecutable = mpsgraph_borrow(handle)
    return mpsgraph_tensor_array_box(executable.feedTensors)
}

@_cdecl("mpsgraph_executable_target_tensors")
public func mpsgraph_executable_target_tensors(_ handle: UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.0, *) else {
            return nil
        }

    guard let handle else {
        return nil
    }
    let executable: MPSGraphExecutable = mpsgraph_borrow(handle)
    return mpsgraph_tensor_array_box(executable.targetTensors)
}

@_cdecl("mpsgraph_executable_specialize")
public func mpsgraph_executable_specialize(
    _ handle: UnsafeMutableRawPointer?,
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ inputTypeHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ inputTypeCount: Int,
    _ compilationDescriptorHandle: UnsafeMutableRawPointer?
) -> Bool {
    guard #available(macOS 12.0, *) else {
            return false
        }

    guard let handle, let inputTypes = mpsgraph_graph_type_array(inputTypeHandles, count: inputTypeCount) else {
        return false
    }
    let executable: MPSGraphExecutable = mpsgraph_borrow(handle)
    let device = mpsgraph_optional_graph_device(deviceHandle)
    let compilationDescriptor = compilationDescriptorHandle.map { ptr in
        let descriptor: MPSGraphCompilationDescriptor = mpsgraph_borrow(ptr)
        return descriptor
    }
    executable.specialize(with: device, inputTypes: inputTypes, compilationDescriptor: compilationDescriptor)
    return true
}

@_cdecl("mpsgraph_executable_get_output_types")
public func mpsgraph_executable_get_output_types(
    _ handle: UnsafeMutableRawPointer?,
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ inputTypeHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ inputTypeCount: Int,
    _ compilationDescriptorHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 13.2, *) else {
            return nil
        }

    guard let handle, let inputTypes = mpsgraph_graph_type_array(inputTypeHandles, count: inputTypeCount) else {
        return nil
    }
    let executable: MPSGraphExecutable = mpsgraph_borrow(handle)
    let device = mpsgraph_optional_graph_device(deviceHandle)
    let compilationDescriptor = compilationDescriptorHandle.map { ptr in
        let descriptor: MPSGraphCompilationDescriptor = mpsgraph_borrow(ptr)
        return descriptor
    }
    let outputTypes = executable.getOutputTypes(with: device, inputTypes: inputTypes, compilationDescriptor: compilationDescriptor)
    return mpsgraph_shaped_type_array_box(outputTypes)
}

@_cdecl("mpsgraph_executable_run_with_descriptor")
public func mpsgraph_executable_run_with_descriptor(
    _ executableHandle: UnsafeMutableRawPointer?,
    _ commandQueueHandle: UnsafeMutableRawPointer?,
    _ inputHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ inputCount: Int,
    _ resultHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ resultCount: Int,
    _ executionDescriptorHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.0, *) else {
            return nil
        }

    guard let executableHandle, let commandQueueHandle, let inputs = mpsgraph_tensor_data_array(inputHandles, count: inputCount) else {
        return nil
    }
    let executable: MPSGraphExecutable = mpsgraph_borrow(executableHandle)
    let commandQueue: MTLCommandQueue = mpsgraph_borrow(commandQueueHandle)
    let results = mpsgraph_tensor_data_array(resultHandles, count: resultCount)
    let executionDescriptor = executionDescriptorHandle.map { ptr in
        let descriptor: MPSGraphExecutableExecutionDescriptor = mpsgraph_borrow(ptr)
        return descriptor
    }
    let output = executable.run(with: commandQueue, inputs: inputs, results: results, executionDescriptor: executionDescriptor)
    return mpsgraph_tensor_data_array_box(output)
}

@_cdecl("mpsgraph_executable_run_async_with_descriptor")
public func mpsgraph_executable_run_async_with_descriptor(
    _ executableHandle: UnsafeMutableRawPointer?,
    _ commandQueueHandle: UnsafeMutableRawPointer?,
    _ inputHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ inputCount: Int,
    _ resultHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ resultCount: Int,
    _ executionDescriptorHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.0, *) else {
            return nil
        }

    guard let executableHandle, let commandQueueHandle, let inputs = mpsgraph_tensor_data_array(inputHandles, count: inputCount) else {
        return nil
    }
    let executable: MPSGraphExecutable = mpsgraph_borrow(executableHandle)
    let commandQueue: MTLCommandQueue = mpsgraph_borrow(commandQueueHandle)
    let results = mpsgraph_tensor_data_array(resultHandles, count: resultCount)
    let executionDescriptor = executionDescriptorHandle.map { ptr in
        let descriptor: MPSGraphExecutableExecutionDescriptor = mpsgraph_borrow(ptr)
        return descriptor
    }
    let output = executable.runAsync(with: commandQueue, inputs: inputs, results: results, executionDescriptor: executionDescriptor)
    return mpsgraph_tensor_data_array_box(output)
}

@_cdecl("mpsgraph_executable_serialize_package")
public func mpsgraph_executable_serialize_package(
    _ handle: UnsafeMutableRawPointer?,
    _ path: UnsafePointer<CChar>?,
    _ descriptorHandle: UnsafeMutableRawPointer?
) -> Bool {
    guard #available(macOS 14.0, *) else {
            return false
        }

    guard let handle, let path else {
        return false
    }
    let executable: MPSGraphExecutable = mpsgraph_borrow(handle)
    let descriptor = descriptorHandle.map { ptr in
        let serializationDescriptor: MPSGraphExecutableSerializationDescriptor = mpsgraph_borrow(ptr)
        return serializationDescriptor
    }
    executable.serialize(package: URL(fileURLWithPath: String(cString: path)), descriptor: descriptor)
    return true
}

@_cdecl("mpsgraph_executable_new_with_package")
public func mpsgraph_executable_new_with_package(
    _ path: UnsafePointer<CChar>?,
    _ compilationDescriptorHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 14.0, *) else {
            return nil
        }

    guard let path else {
        return nil
    }
    let compilationDescriptor = compilationDescriptorHandle.map { ptr in
        let descriptor: MPSGraphCompilationDescriptor = mpsgraph_borrow(ptr)
        return descriptor
    }
    let executable = MPSGraphExecutable(
        package: URL(fileURLWithPath: String(cString: path)),
        descriptor: compilationDescriptor
    )
    return mpsgraph_retain(executable)
}
