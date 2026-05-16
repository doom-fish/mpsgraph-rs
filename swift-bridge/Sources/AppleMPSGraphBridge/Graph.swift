import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

@_cdecl("mpsgraph_graph_new")
public func mpsgraph_graph_new() -> UnsafeMutableRawPointer? {
    mpsgraph_retain(MPSGraph())
}

@_cdecl("mpsgraph_graph_placeholder")
public func mpsgraph_graph_placeholder(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ shape: UnsafePointer<UInt>?,
    _ shapeLen: Int,
    _ dataTypeRaw: UInt32,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let dataType = mpsgraph_data_type(dataTypeRaw) else {
        return nil
    }

    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let tensor = graph.placeholder(
        shape: mpsgraph_optional_shape(shape, shapeLen),
        dataType: dataType,
        name: mpsgraph_optional_name(name)
    )
    return mpsgraph_retain(tensor)
}

@_cdecl("mpsgraph_graph_constant_data")
public func mpsgraph_graph_constant_data(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ bytes: UnsafeRawPointer?,
    _ byteLen: Int,
    _ shape: UnsafePointer<UInt>?,
    _ shapeLen: Int,
    _ dataTypeRaw: UInt32
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let dataType = mpsgraph_data_type(dataTypeRaw) else {
        return nil
    }
    guard byteLen == 0 || bytes != nil else {
        return nil
    }

    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let tensor = graph.constant(mpsgraph_data(bytes, byteLen), shape: mpsgraph_shape(shape, shapeLen), dataType: dataType)
    return mpsgraph_retain(tensor)
}

@_cdecl("mpsgraph_graph_constant_scalar")
public func mpsgraph_graph_constant_scalar(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ scalar: Double,
    _ dataTypeRaw: UInt32
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let dataType = mpsgraph_data_type(dataTypeRaw) else {
        return nil
    }

    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    return mpsgraph_retain(graph.constant(scalar, dataType: dataType))
}

@_cdecl("mpsgraph_graph_constant_scalar_shaped")
public func mpsgraph_graph_constant_scalar_shaped(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ scalar: Double,
    _ shape: UnsafePointer<UInt>?,
    _ shapeLen: Int,
    _ dataTypeRaw: UInt32
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let dataType = mpsgraph_data_type(dataTypeRaw) else {
        return nil
    }

    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    return mpsgraph_retain(graph.constant(scalar, shape: mpsgraph_shape(shape, shapeLen), dataType: dataType))
}

func mpsgraph_binary_tensor_op(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ primaryHandle: UnsafeMutableRawPointer?,
    _ secondaryHandle: UnsafeMutableRawPointer?,
    _ body: (MPSGraph, MPSGraphTensor, MPSGraphTensor) -> MPSGraphTensor
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let primaryHandle, let secondaryHandle else {
        return nil
    }

    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let primary: MPSGraphTensor = mpsgraph_borrow(primaryHandle)
    let secondary: MPSGraphTensor = mpsgraph_borrow(secondaryHandle)
    return mpsgraph_retain(body(graph, primary, secondary))
}

func mpsgraph_unary_tensor_op(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ body: (MPSGraph, MPSGraphTensor) -> MPSGraphTensor
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let tensorHandle else {
        return nil
    }

    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let tensor: MPSGraphTensor = mpsgraph_borrow(tensorHandle)
    return mpsgraph_retain(body(graph, tensor))
}

@_cdecl("mpsgraph_graph_addition")
public func mpsgraph_graph_addition(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ primaryHandle: UnsafeMutableRawPointer?,
    _ secondaryHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    mpsgraph_binary_tensor_op(graphHandle, primaryHandle, secondaryHandle) {
        $0.addition($1, $2, name: mpsgraph_optional_name(name))
    }
}

@_cdecl("mpsgraph_graph_subtraction")
public func mpsgraph_graph_subtraction(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ primaryHandle: UnsafeMutableRawPointer?,
    _ secondaryHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    mpsgraph_binary_tensor_op(graphHandle, primaryHandle, secondaryHandle) {
        $0.subtraction($1, $2, name: mpsgraph_optional_name(name))
    }
}

@_cdecl("mpsgraph_graph_multiplication")
public func mpsgraph_graph_multiplication(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ primaryHandle: UnsafeMutableRawPointer?,
    _ secondaryHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    mpsgraph_binary_tensor_op(graphHandle, primaryHandle, secondaryHandle) {
        $0.multiplication($1, $2, name: mpsgraph_optional_name(name))
    }
}

@_cdecl("mpsgraph_graph_division")
public func mpsgraph_graph_division(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ primaryHandle: UnsafeMutableRawPointer?,
    _ secondaryHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    mpsgraph_binary_tensor_op(graphHandle, primaryHandle, secondaryHandle) {
        $0.division($1, $2, name: mpsgraph_optional_name(name))
    }
}

@_cdecl("mpsgraph_graph_matrix_multiplication")
public func mpsgraph_graph_matrix_multiplication(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ primaryHandle: UnsafeMutableRawPointer?,
    _ secondaryHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    mpsgraph_binary_tensor_op(graphHandle, primaryHandle, secondaryHandle) {
        $0.matrixMultiplication(primary: $1, secondary: $2, name: mpsgraph_optional_name(name))
    }
}

@_cdecl("mpsgraph_graph_relu")
public func mpsgraph_graph_relu(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    mpsgraph_unary_tensor_op(graphHandle, tensorHandle) {
        $0.reLU(with: $1, name: mpsgraph_optional_name(name))
    }
}

@_cdecl("mpsgraph_graph_sigmoid")
public func mpsgraph_graph_sigmoid(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    mpsgraph_unary_tensor_op(graphHandle, tensorHandle) {
        $0.sigmoid(with: $1, name: mpsgraph_optional_name(name))
    }
}

@_cdecl("mpsgraph_graph_softmax")
public func mpsgraph_graph_softmax(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ axis: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    mpsgraph_unary_tensor_op(graphHandle, tensorHandle) {
        $0.softMax(with: $1, axis: axis, name: mpsgraph_optional_name(name))
    }
}

@_cdecl("mpsgraph_graph_reshape")
public func mpsgraph_graph_reshape(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ shape: UnsafePointer<UInt>?,
    _ shapeLen: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    mpsgraph_unary_tensor_op(graphHandle, tensorHandle) {
        $0.reshape($1, shape: mpsgraph_shape(shape, shapeLen), name: mpsgraph_optional_name(name))
    }
}

@_cdecl("mpsgraph_graph_transpose")
public func mpsgraph_graph_transpose(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ permutation: UnsafePointer<UInt>?,
    _ permutationLen: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 13.0, *) else {
        return nil
    }

    return mpsgraph_unary_tensor_op(graphHandle, tensorHandle) {
        $0.transpose($1, permutation: mpsgraph_shape(permutation, permutationLen), name: mpsgraph_optional_name(name))
    }
}

@_cdecl("mpsgraph_graph_slice")
public func mpsgraph_graph_slice(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ dimension: Int,
    _ start: Int,
    _ length: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    mpsgraph_unary_tensor_op(graphHandle, tensorHandle) {
        $0.sliceTensor($1, dimension: dimension, start: start, length: length, name: mpsgraph_optional_name(name))
    }
}

@_cdecl("mpsgraph_graph_broadcast")
public func mpsgraph_graph_broadcast(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ shape: UnsafePointer<UInt>?,
    _ shapeLen: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.0, *) else {
        return nil
    }

    return mpsgraph_unary_tensor_op(graphHandle, tensorHandle) {
        $0.broadcast($1, shape: mpsgraph_shape(shape, shapeLen), name: mpsgraph_optional_name(name))
    }
}

func mpsgraph_axes_tensor_op(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ axes: UnsafePointer<UInt>?,
    _ axesLen: Int,
    _ body: (MPSGraph, MPSGraphTensor, [NSNumber]) -> MPSGraphTensor
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let tensorHandle else {
        return nil
    }

    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let tensor: MPSGraphTensor = mpsgraph_borrow(tensorHandle)
    return mpsgraph_retain(body(graph, tensor, mpsgraph_shape(axes, axesLen)))
}

@_cdecl("mpsgraph_graph_reduction_sum")
public func mpsgraph_graph_reduction_sum(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ axes: UnsafePointer<UInt>?,
    _ axesLen: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    mpsgraph_axes_tensor_op(graphHandle, tensorHandle, axes, axesLen) {
        $0.reductionSum(with: $1, axes: $2, name: mpsgraph_optional_name(name))
    }
}

@_cdecl("mpsgraph_graph_reduction_maximum")
public func mpsgraph_graph_reduction_maximum(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ axes: UnsafePointer<UInt>?,
    _ axesLen: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    mpsgraph_axes_tensor_op(graphHandle, tensorHandle, axes, axesLen) {
        $0.reductionMaximum(with: $1, axes: $2, name: mpsgraph_optional_name(name))
    }
}

@_cdecl("mpsgraph_graph_reduction_minimum")
public func mpsgraph_graph_reduction_minimum(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ axes: UnsafePointer<UInt>?,
    _ axesLen: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    mpsgraph_axes_tensor_op(graphHandle, tensorHandle, axes, axesLen) {
        $0.reductionMinimum(with: $1, axes: $2, name: mpsgraph_optional_name(name))
    }
}

@_cdecl("mpsgraph_graph_mean")
public func mpsgraph_graph_mean(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ axes: UnsafePointer<UInt>?,
    _ axesLen: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    mpsgraph_axes_tensor_op(graphHandle, tensorHandle, axes, axesLen) {
        $0.mean(of: $1, axes: $2, name: mpsgraph_optional_name(name))
    }
}

@_cdecl("mpsgraph_convolution2d_descriptor_new")
public func mpsgraph_convolution2d_descriptor_new(
    _ strideInX: Int,
    _ strideInY: Int,
    _ dilationRateInX: Int,
    _ dilationRateInY: Int,
    _ groups: Int,
    _ paddingLeft: Int,
    _ paddingRight: Int,
    _ paddingTop: Int,
    _ paddingBottom: Int,
    _ paddingStyleRaw: UInt,
    _ dataLayoutRaw: UInt,
    _ weightsLayoutRaw: UInt
) -> UnsafeMutableRawPointer? {
    guard
        let paddingStyle = mpsgraph_padding_style(paddingStyleRaw),
        let dataLayout = mpsgraph_data_layout(dataLayoutRaw),
        let weightsLayout = mpsgraph_data_layout(weightsLayoutRaw)
    else {
        return nil
    }

    let descriptor = MPSGraphConvolution2DOpDescriptor()
    descriptor.strideInX = strideInX
    descriptor.strideInY = strideInY
    descriptor.dilationRateInX = dilationRateInX
    descriptor.dilationRateInY = dilationRateInY
    descriptor.groups = groups
    descriptor.paddingLeft = paddingLeft
    descriptor.paddingRight = paddingRight
    descriptor.paddingTop = paddingTop
    descriptor.paddingBottom = paddingBottom
    descriptor.paddingStyle = paddingStyle
    descriptor.dataLayout = dataLayout
    descriptor.weightsLayout = weightsLayout
    return mpsgraph_retain(descriptor)
}

@_cdecl("mpsgraph_pooling2d_descriptor_new")
public func mpsgraph_pooling2d_descriptor_new(
    _ kernelWidth: Int,
    _ kernelHeight: Int,
    _ strideInX: Int,
    _ strideInY: Int,
    _ dilationRateInX: Int,
    _ dilationRateInY: Int,
    _ paddingLeft: Int,
    _ paddingRight: Int,
    _ paddingTop: Int,
    _ paddingBottom: Int,
    _ paddingStyleRaw: UInt,
    _ dataLayoutRaw: UInt
) -> UnsafeMutableRawPointer? {
    guard
        let paddingStyle = mpsgraph_padding_style(paddingStyleRaw),
        let dataLayout = mpsgraph_data_layout(dataLayoutRaw)
    else {
        return nil
    }

    let descriptor = MPSGraphPooling2DOpDescriptor()
    descriptor.kernelWidth = kernelWidth
    descriptor.kernelHeight = kernelHeight
    descriptor.strideInX = strideInX
    descriptor.strideInY = strideInY
    descriptor.dilationRateInX = dilationRateInX
    descriptor.dilationRateInY = dilationRateInY
    descriptor.paddingLeft = paddingLeft
    descriptor.paddingRight = paddingRight
    descriptor.paddingTop = paddingTop
    descriptor.paddingBottom = paddingBottom
    descriptor.paddingStyle = paddingStyle
    descriptor.dataLayout = dataLayout
    return mpsgraph_retain(descriptor)
}

@_cdecl("mpsgraph_graph_convolution2d")
public func mpsgraph_graph_convolution2d(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ sourceHandle: UnsafeMutableRawPointer?,
    _ weightsHandle: UnsafeMutableRawPointer?,
    _ descriptorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let sourceHandle, let weightsHandle, let descriptorHandle else {
        return nil
    }

    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let source: MPSGraphTensor = mpsgraph_borrow(sourceHandle)
    let weights: MPSGraphTensor = mpsgraph_borrow(weightsHandle)
    let descriptor: MPSGraphConvolution2DOpDescriptor = mpsgraph_borrow(descriptorHandle)
    return mpsgraph_retain(
        graph.convolution2D(source, weights: weights, descriptor: descriptor, name: mpsgraph_optional_name(name))
    )
}

@_cdecl("mpsgraph_graph_max_pooling2d")
public func mpsgraph_graph_max_pooling2d(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ sourceHandle: UnsafeMutableRawPointer?,
    _ descriptorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let sourceHandle, let descriptorHandle else {
        return nil
    }

    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let source: MPSGraphTensor = mpsgraph_borrow(sourceHandle)
    let descriptor: MPSGraphPooling2DOpDescriptor = mpsgraph_borrow(descriptorHandle)
    return mpsgraph_retain(
        graph.maxPooling2D(withSourceTensor: source, descriptor: descriptor, name: mpsgraph_optional_name(name))
    )
}

@_cdecl("mpsgraph_graph_normalize")
public func mpsgraph_graph_normalize(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ meanHandle: UnsafeMutableRawPointer?,
    _ varianceHandle: UnsafeMutableRawPointer?,
    _ gammaHandle: UnsafeMutableRawPointer?,
    _ betaHandle: UnsafeMutableRawPointer?,
    _ epsilon: Float,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let tensorHandle, let meanHandle, let varianceHandle else {
        return nil
    }

    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let tensor: MPSGraphTensor = mpsgraph_borrow(tensorHandle)
    let mean: MPSGraphTensor = mpsgraph_borrow(meanHandle)
    let variance: MPSGraphTensor = mpsgraph_borrow(varianceHandle)
    let gamma: MPSGraphTensor? = gammaHandle.map { mpsgraph_borrow($0) }
    let beta: MPSGraphTensor? = betaHandle.map { mpsgraph_borrow($0) }
    return mpsgraph_retain(
        graph.normalize(
            tensor,
            mean: mean,
            variance: variance,
            gamma: gamma,
            beta: beta,
            epsilon: epsilon,
            name: mpsgraph_optional_name(name)
        )
    )
}

@_cdecl("mpsgraph_graph_run")
public func mpsgraph_graph_run(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ feedTensorHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ feedDataHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ feedCount: Int,
    _ targetTensorHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ targetCount: Int,
    _ outResults: UnsafeMutablePointer<UnsafeMutableRawPointer?>?
) -> Bool {
    guard
        let graphHandle,
        let feeds = mpsgraph_feed_dictionary(tensorHandles: feedTensorHandles, dataHandles: feedDataHandles, count: feedCount),
        let targets = mpsgraph_tensor_array(targetTensorHandles, count: targetCount)
    else {
        return false
    }

    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let results = graph.run(feeds: feeds, targetTensors: targets, targetOperations: nil)
    return mpsgraph_write_results(results, targets: targets, outResults: outResults)
}

@_cdecl("mpsgraph_graph_run_with_command_queue")
public func mpsgraph_graph_run_with_command_queue(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ commandQueueHandle: UnsafeMutableRawPointer?,
    _ feedTensorHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ feedDataHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ feedCount: Int,
    _ targetTensorHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ targetCount: Int,
    _ outResults: UnsafeMutablePointer<UnsafeMutableRawPointer?>?
) -> Bool {
    guard
        let graphHandle,
        let commandQueueHandle,
        let feeds = mpsgraph_feed_dictionary(tensorHandles: feedTensorHandles, dataHandles: feedDataHandles, count: feedCount),
        let targets = mpsgraph_tensor_array(targetTensorHandles, count: targetCount)
    else {
        return false
    }

    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let commandQueue: MTLCommandQueue = mpsgraph_borrow(commandQueueHandle)
    let results = graph.run(with: commandQueue, feeds: feeds, targetTensors: targets, targetOperations: nil)
    return mpsgraph_write_results(results, targets: targets, outResults: outResults)
}

@_cdecl("mpsgraph_graph_compile")
public func mpsgraph_graph_compile(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ feedTensorHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ feedCount: Int,
    _ flatShapes: UnsafePointer<UInt>?,
    _ shapeLengths: UnsafePointer<UInt>?,
    _ dataTypes: UnsafePointer<UInt32>?,
    _ targetTensorHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ targetCount: Int
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.0, *) else {
        return nil
    }
    guard
        let graphHandle,
        let deviceHandle,
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
    let executable = graph.compile(
        with: mpsgraph_graph_device(deviceHandle),
        feeds: feeds,
        targetTensors: targets,
        targetOperations: nil,
        compilationDescriptor: nil
    )
    return mpsgraph_retain(executable)
}

@_cdecl("mpsgraph_executable_run")
public func mpsgraph_executable_run(
    _ executableHandle: UnsafeMutableRawPointer?,
    _ commandQueueHandle: UnsafeMutableRawPointer?,
    _ inputHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ inputCount: Int,
    _ outputCount: Int,
    _ outResults: UnsafeMutablePointer<UnsafeMutableRawPointer?>?
) -> Bool {
    guard #available(macOS 12.0, *) else {
        return false
    }
    guard
        let executableHandle,
        let commandQueueHandle,
        let inputs = mpsgraph_tensor_data_array(inputHandles, count: inputCount)
    else {
        return false
    }

    let executable: MPSGraphExecutable = mpsgraph_borrow(executableHandle)
    let commandQueue: MTLCommandQueue = mpsgraph_borrow(commandQueueHandle)
    let executionDescriptor: MPSGraphExecutableExecutionDescriptor? = nil
    let results = executable.run(
        with: commandQueue,
        inputs: inputs,
        results: nil,
        executionDescriptor: executionDescriptor
    )
    return mpsgraph_write_result_array(results, outResults: outResults, expectedCount: outputCount)
}
