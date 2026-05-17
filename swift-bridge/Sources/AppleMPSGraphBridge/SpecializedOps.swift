import Foundation
import Metal
import MetalPerformanceShadersGraph

@inline(__always)
private func mpsgraph_shared_event(_ handle: UnsafeMutableRawPointer?) -> MTLSharedEvent? {
    guard let handle else {
        return nil
    }
    let event: MTLSharedEvent = mpsgraph_borrow(handle)
    return event
}

@available(macOS 13.0, *)
@inline(__always)
private func mpsgraph_execution_stage(_ rawValue: UInt64) -> MPSGraphExecutionStage? {
    MPSGraphExecutionStage(rawValue: rawValue)
}

@_cdecl("mpsgraph_graph_convolution3d")
public func mpsgraph_graph_convolution3d(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ sourceTensorHandle: UnsafeMutableRawPointer?,
    _ weightsTensorHandle: UnsafeMutableRawPointer?,
    _ descriptorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 13.2, *) else {
        return nil
    }
    guard let graphHandle, let sourceTensorHandle, let weightsTensorHandle, let descriptorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let sourceTensor: MPSGraphTensor = mpsgraph_borrow(sourceTensorHandle)
    let weightsTensor: MPSGraphTensor = mpsgraph_borrow(weightsTensorHandle)
    let descriptor: MPSGraphConvolution3DOpDescriptor = mpsgraph_borrow(descriptorHandle)
    return mpsgraph_retain(
        graph.convolution3D(sourceTensor, weights: weightsTensor, descriptor: descriptor, name: mpsgraph_optional_name(name))
    )
}

@_cdecl("mpsgraph_graph_convolution_transpose2d")
public func mpsgraph_graph_convolution_transpose2d(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ sourceTensorHandle: UnsafeMutableRawPointer?,
    _ weightsTensorHandle: UnsafeMutableRawPointer?,
    _ outputShape: UnsafePointer<UInt>?,
    _ outputShapeLen: Int,
    _ descriptorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let sourceTensorHandle, let weightsTensorHandle, let descriptorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let sourceTensor: MPSGraphTensor = mpsgraph_borrow(sourceTensorHandle)
    let weightsTensor: MPSGraphTensor = mpsgraph_borrow(weightsTensorHandle)
    let descriptor: MPSGraphConvolution2DOpDescriptor = mpsgraph_borrow(descriptorHandle)
    return mpsgraph_retain(
        graph.convolutionTranspose2D(
            sourceTensor,
            weights: weightsTensor,
            outputShape: mpsgraph_shape(outputShape, outputShapeLen),
            descriptor: descriptor,
            name: mpsgraph_optional_name(name)
        )
    )
}

@_cdecl("mpsgraph_graph_cumulative_sum")
public func mpsgraph_graph_cumulative_sum(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ axis: Int,
    _ exclusive: Bool,
    _ reverse: Bool,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 13.0, *) else {
        return nil
    }
    guard let graphHandle, let tensorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let tensor: MPSGraphTensor = mpsgraph_borrow(tensorHandle)
    return mpsgraph_retain(
        graph.cumulativeSum(tensor, axis: axis, exclusive: exclusive, reverse: reverse, name: mpsgraph_optional_name(name))
    )
}

@_cdecl("mpsgraph_graph_depthwise_convolution2d")
public func mpsgraph_graph_depthwise_convolution2d(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ sourceTensorHandle: UnsafeMutableRawPointer?,
    _ weightsTensorHandle: UnsafeMutableRawPointer?,
    _ descriptorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let sourceTensorHandle, let weightsTensorHandle, let descriptorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let sourceTensor: MPSGraphTensor = mpsgraph_borrow(sourceTensorHandle)
    let weightsTensor: MPSGraphTensor = mpsgraph_borrow(weightsTensorHandle)
    let descriptor: MPSGraphDepthwiseConvolution2DOpDescriptor = mpsgraph_borrow(descriptorHandle)
    return mpsgraph_retain(
        graph.depthwiseConvolution2D(sourceTensor, weights: weightsTensor, descriptor: descriptor, name: mpsgraph_optional_name(name))
    )
}

@_cdecl("mpsgraph_graph_depthwise_convolution3d")
public func mpsgraph_graph_depthwise_convolution3d(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ sourceTensorHandle: UnsafeMutableRawPointer?,
    _ weightsTensorHandle: UnsafeMutableRawPointer?,
    _ descriptorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.0, *) else {
        return nil
    }
    guard let graphHandle, let sourceTensorHandle, let weightsTensorHandle, let descriptorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let sourceTensor: MPSGraphTensor = mpsgraph_borrow(sourceTensorHandle)
    let weightsTensor: MPSGraphTensor = mpsgraph_borrow(weightsTensorHandle)
    let descriptor: MPSGraphDepthwiseConvolution3DOpDescriptor = mpsgraph_borrow(descriptorHandle)
    return mpsgraph_retain(
        graph.depthwiseConvolution3D(sourceTensor, weights: weightsTensor, descriptor: descriptor, name: mpsgraph_optional_name(name))
    )
}

@_cdecl("mpsgraph_graph_fast_fourier_transform")
public func mpsgraph_graph_fast_fourier_transform(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ axes: UnsafePointer<UInt>?,
    _ axesLen: Int,
    _ descriptorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 14.0, *) else {
        return nil
    }
    guard let graphHandle, let tensorHandle, let descriptorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let tensor: MPSGraphTensor = mpsgraph_borrow(tensorHandle)
    let descriptor: MPSGraphFFTDescriptor = mpsgraph_borrow(descriptorHandle)
    return mpsgraph_retain(
        graph.fastFourierTransform(tensor, axes: mpsgraph_shape(axes, axesLen), descriptor: descriptor, name: mpsgraph_optional_name(name))
    )
}

@_cdecl("mpsgraph_graph_im_to_col")
public func mpsgraph_graph_im_to_col(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ sourceTensorHandle: UnsafeMutableRawPointer?,
    _ descriptorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 14.0, *) else {
        return nil
    }
    guard let graphHandle, let sourceTensorHandle, let descriptorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let sourceTensor: MPSGraphTensor = mpsgraph_borrow(sourceTensorHandle)
    let descriptor: MPSGraphImToColOpDescriptor = mpsgraph_borrow(descriptorHandle)
    return mpsgraph_retain(graph.imToCol(sourceTensor, descriptor: descriptor, name: mpsgraph_optional_name(name)))
}

@_cdecl("mpsgraph_graph_band_part")
public func mpsgraph_graph_band_part(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ numLower: Int,
    _ numUpper: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let tensorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let tensor: MPSGraphTensor = mpsgraph_borrow(tensorHandle)
    return mpsgraph_retain(graph.bandPart(tensor, numLower: numLower, numUpper: numUpper, name: mpsgraph_optional_name(name)))
}

@_cdecl("mpsgraph_graph_softmax_cross_entropy")
public func mpsgraph_graph_softmax_cross_entropy(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ sourceTensorHandle: UnsafeMutableRawPointer?,
    _ labelsTensorHandle: UnsafeMutableRawPointer?,
    _ axis: Int,
    _ reductionTypeRaw: UInt64,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let sourceTensorHandle, let labelsTensorHandle,
          let reductionType = mpsgraph_loss_reduction_type(reductionTypeRaw)
    else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let sourceTensor: MPSGraphTensor = mpsgraph_borrow(sourceTensorHandle)
    let labelsTensor: MPSGraphTensor = mpsgraph_borrow(labelsTensorHandle)
    return mpsgraph_retain(
        graph.softMaxCrossEntropy(sourceTensor, labels: labelsTensor, axis: axis, reuctionType: reductionType, name: mpsgraph_optional_name(name))
    )
}

@_cdecl("mpsgraph_graph_matrix_inverse")
public func mpsgraph_graph_matrix_inverse(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 13.0, *) else {
        return nil
    }
    guard let graphHandle, let tensorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let tensor: MPSGraphTensor = mpsgraph_borrow(tensorHandle)
    return mpsgraph_retain(graph.inverse(input: tensor, name: mpsgraph_optional_name(name)))
}

@_cdecl("mpsgraph_graph_variable_data")
public func mpsgraph_graph_variable_data(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ bytes: UnsafeRawPointer?,
    _ byteLen: Int,
    _ shape: UnsafePointer<UInt>?,
    _ shapeLen: Int,
    _ dataTypeRaw: UInt32,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let dataType = mpsgraph_data_type(dataTypeRaw) else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    return mpsgraph_retain(
        graph.variable(with: mpsgraph_data(bytes, byteLen), shape: mpsgraph_shape(shape, shapeLen), dataType: dataType, name: mpsgraph_optional_name(name))
    )
}

@_cdecl("mpsgraph_graph_read_variable")
public func mpsgraph_graph_read_variable(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ variableTensorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let variableTensorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let variableTensor: MPSGraphTensor = mpsgraph_borrow(variableTensorHandle)
    return mpsgraph_retain(graph.read(variableTensor, name: mpsgraph_optional_name(name)))
}

@_cdecl("mpsgraph_graph_assign_variable")
public func mpsgraph_graph_assign_variable(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ variableTensorHandle: UnsafeMutableRawPointer?,
    _ valueTensorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let variableTensorHandle, let valueTensorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let variableTensor: MPSGraphTensor = mpsgraph_borrow(variableTensorHandle)
    let valueTensor: MPSGraphTensor = mpsgraph_borrow(valueTensorHandle)
    return mpsgraph_retain(graph.assign(variableTensor, tensor: valueTensor, name: mpsgraph_optional_name(name)))
}

@_cdecl("mpsgraph_graph_non_maximum_suppression")
public func mpsgraph_graph_non_maximum_suppression(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ boxesTensorHandle: UnsafeMutableRawPointer?,
    _ scoresTensorHandle: UnsafeMutableRawPointer?,
    _ iouThreshold: Float,
    _ scoreThreshold: Float,
    _ perClassSuppression: Bool,
    _ coordinateModeRaw: UInt,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 14.0, *) else {
        return nil
    }
    guard let graphHandle, let boxesTensorHandle, let scoresTensorHandle,
          let coordinateMode = mpsgraph_nms_coordinate_mode(coordinateModeRaw)
    else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let boxesTensor: MPSGraphTensor = mpsgraph_borrow(boxesTensorHandle)
    let scoresTensor: MPSGraphTensor = mpsgraph_borrow(scoresTensorHandle)
    return mpsgraph_retain(
        graph.nonMaximumSuppression(
            withBoxesTensor: boxesTensor,
            scoresTensor: scoresTensor,
            iouThreshold: iouThreshold,
            scoreThreshold: scoreThreshold,
            perClassSuppression: perClassSuppression,
            coordinateMode: coordinateMode,
            name: mpsgraph_optional_name(name)
        )
    )
}

@_cdecl("mpsgraph_graph_non_zero_indices")
public func mpsgraph_graph_non_zero_indices(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 14.0, *) else {
        return nil
    }
    guard let graphHandle, let tensorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let tensor: MPSGraphTensor = mpsgraph_borrow(tensorHandle)
    return mpsgraph_retain(graph.nonZeroIndices(tensor, name: mpsgraph_optional_name(name)))
}

@_cdecl("mpsgraph_graph_one_hot")
public func mpsgraph_graph_one_hot(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ indicesTensorHandle: UnsafeMutableRawPointer?,
    _ depth: Int,
    _ dataTypeRaw: UInt32,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let indicesTensorHandle, let dataType = mpsgraph_data_type(dataTypeRaw) else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let indicesTensor: MPSGraphTensor = mpsgraph_borrow(indicesTensorHandle)
    return mpsgraph_retain(
        graph.oneHot(withIndicesTensor: indicesTensor, depth: depth, dataType: dataType, name: mpsgraph_optional_name(name))
    )
}

@_cdecl("mpsgraph_graph_stochastic_gradient_descent")
public func mpsgraph_graph_stochastic_gradient_descent(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ learningRateTensorHandle: UnsafeMutableRawPointer?,
    _ valuesTensorHandle: UnsafeMutableRawPointer?,
    _ gradientTensorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let learningRateTensorHandle, let valuesTensorHandle, let gradientTensorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let learningRateTensor: MPSGraphTensor = mpsgraph_borrow(learningRateTensorHandle)
    let valuesTensor: MPSGraphTensor = mpsgraph_borrow(valuesTensorHandle)
    let gradientTensor: MPSGraphTensor = mpsgraph_borrow(gradientTensorHandle)
    return mpsgraph_retain(
        graph.stochasticGradientDescent(
            learningRate: learningRateTensor,
            values: valuesTensor,
            gradient: gradientTensor,
            name: mpsgraph_optional_name(name)
        )
    )
}

@_cdecl("mpsgraph_graph_max_pooling4d")
public func mpsgraph_graph_max_pooling4d(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ sourceTensorHandle: UnsafeMutableRawPointer?,
    _ descriptorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.0, *) else {
        return nil
    }
    guard let graphHandle, let sourceTensorHandle, let descriptorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let sourceTensor: MPSGraphTensor = mpsgraph_borrow(sourceTensorHandle)
    let descriptor: MPSGraphPooling4DOpDescriptor = mpsgraph_borrow(descriptorHandle)
    return mpsgraph_retain(graph.maxPooling4D(sourceTensor, descriptor: descriptor, name: mpsgraph_optional_name(name)))
}

@_cdecl("mpsgraph_graph_max_pooling4d_return_indices")
public func mpsgraph_graph_max_pooling4d_return_indices(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ sourceTensorHandle: UnsafeMutableRawPointer?,
    _ descriptorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.2, *) else {
        return nil
    }
    guard let graphHandle, let sourceTensorHandle, let descriptorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let sourceTensor: MPSGraphTensor = mpsgraph_borrow(sourceTensorHandle)
    let descriptor: MPSGraphPooling4DOpDescriptor = mpsgraph_borrow(descriptorHandle)
    return mpsgraph_tensor_array_box(
        graph.maxPooling4DReturnIndices(sourceTensor, descriptor: descriptor, name: mpsgraph_optional_name(name))
    )
}

@_cdecl("mpsgraph_graph_quantize")
public func mpsgraph_graph_quantize(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ scale: Double,
    _ zeroPoint: Double,
    _ dataTypeRaw: UInt32,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 13.1, *) else {
        return nil
    }
    guard let graphHandle, let tensorHandle, let dataType = mpsgraph_data_type(dataTypeRaw) else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let tensor: MPSGraphTensor = mpsgraph_borrow(tensorHandle)
    return mpsgraph_retain(graph.quantize(tensor, scale: scale, zeroPoint: zeroPoint, dataType: dataType, name: mpsgraph_optional_name(name)))
}

@_cdecl("mpsgraph_graph_dequantize")
public func mpsgraph_graph_dequantize(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ scale: Double,
    _ zeroPoint: Double,
    _ dataTypeRaw: UInt32,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 13.1, *) else {
        return nil
    }
    guard let graphHandle, let tensorHandle, let dataType = mpsgraph_data_type(dataTypeRaw) else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let tensor: MPSGraphTensor = mpsgraph_borrow(tensorHandle)
    return mpsgraph_retain(graph.dequantize(tensor, scale: scale, zeroPoint: zeroPoint, dataType: dataType, name: mpsgraph_optional_name(name)))
}

@_cdecl("mpsgraph_graph_resize")
public func mpsgraph_graph_resize(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ imagesTensorHandle: UnsafeMutableRawPointer?,
    _ size: UnsafePointer<UInt>?,
    _ sizeLen: Int,
    _ modeRaw: UInt,
    _ centerResult: Bool,
    _ alignCorners: Bool,
    _ layoutRaw: UInt,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 13.0, *) else {
        return nil
    }
    guard let graphHandle, let imagesTensorHandle,
          let mode = mpsgraph_resize_mode(modeRaw),
          let layout = mpsgraph_data_layout(layoutRaw)
    else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let imagesTensor: MPSGraphTensor = mpsgraph_borrow(imagesTensorHandle)
    return mpsgraph_retain(
        graph.resize(
            imagesTensor,
            size: mpsgraph_shape(size, sizeLen),
            mode: mode,
            centerResult: centerResult,
            alignCorners: alignCorners,
            layout: layout,
            name: mpsgraph_optional_name(name)
        )
    )
}

@_cdecl("mpsgraph_graph_resize_nearest")
public func mpsgraph_graph_resize_nearest(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ imagesTensorHandle: UnsafeMutableRawPointer?,
    _ sizeTensorHandle: UnsafeMutableRawPointer?,
    _ nearestRoundingModeRaw: UInt,
    _ centerResult: Bool,
    _ alignCorners: Bool,
    _ layoutRaw: UInt,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 13.0, *) else {
        return nil
    }
    guard let graphHandle, let imagesTensorHandle, let sizeTensorHandle,
          let nearestRoundingMode = mpsgraph_resize_nearest_rounding_mode(nearestRoundingModeRaw),
          let layout = mpsgraph_data_layout(layoutRaw)
    else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let imagesTensor: MPSGraphTensor = mpsgraph_borrow(imagesTensorHandle)
    let sizeTensor: MPSGraphTensor = mpsgraph_borrow(sizeTensorHandle)
    return mpsgraph_retain(
        graph.resizeNearest(
            imagesTensor,
            sizeTensor: sizeTensor,
            nearestRoundingMode: nearestRoundingMode,
            centerResult: centerResult,
            alignCorners: alignCorners,
            layout: layout,
            name: mpsgraph_optional_name(name)
        )
    )
}

@_cdecl("mpsgraph_graph_sample_grid")
public func mpsgraph_graph_sample_grid(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ sourceTensorHandle: UnsafeMutableRawPointer?,
    _ coordinateTensorHandle: UnsafeMutableRawPointer?,
    _ layoutRaw: UInt,
    _ normalizeCoordinates: Bool,
    _ relativeCoordinates: Bool,
    _ alignCorners: Bool,
    _ paddingModeRaw: Int,
    _ samplingModeRaw: UInt,
    _ constantValue: Double,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 13.1, *) else {
        return nil
    }
    guard let graphHandle, let sourceTensorHandle, let coordinateTensorHandle,
          let layout = mpsgraph_data_layout(layoutRaw),
          let paddingMode = mpsgraph_padding_mode(paddingModeRaw),
          let samplingMode = mpsgraph_resize_mode(samplingModeRaw)
    else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let sourceTensor: MPSGraphTensor = mpsgraph_borrow(sourceTensorHandle)
    let coordinateTensor: MPSGraphTensor = mpsgraph_borrow(coordinateTensorHandle)
    return mpsgraph_retain(
        graph.sampleGrid(
            withSourceTensor: sourceTensor,
            coordinateTensor: coordinateTensor,
            layout: layout,
            normalizeCoordinates: normalizeCoordinates,
            relativeCoordinates: relativeCoordinates,
            alignCorners: alignCorners,
            paddingMode: paddingMode,
            samplingMode: samplingMode,
            constantValue: constantValue,
            name: mpsgraph_optional_name(name)
        )
    )
}

@_cdecl("mpsgraph_graph_scatter_nd")
public func mpsgraph_graph_scatter_nd(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ updatesTensorHandle: UnsafeMutableRawPointer?,
    _ indicesTensorHandle: UnsafeMutableRawPointer?,
    _ shape: UnsafePointer<UInt>?,
    _ shapeLen: Int,
    _ batchDimensions: Int,
    _ modeRaw: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.0, *) else {
        return nil
    }
    guard let graphHandle, let updatesTensorHandle, let indicesTensorHandle,
          let mode = mpsgraph_scatter_mode(modeRaw)
    else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let updatesTensor: MPSGraphTensor = mpsgraph_borrow(updatesTensorHandle)
    let indicesTensor: MPSGraphTensor = mpsgraph_borrow(indicesTensorHandle)
    return mpsgraph_retain(
        graph.scatterND(
            withUpdatesTensor: updatesTensor,
            indicesTensor: indicesTensor,
            shape: mpsgraph_shape(shape, shapeLen),
            batchDimensions: batchDimensions,
            mode: mode,
            name: mpsgraph_optional_name(name)
        )
    )
}

@_cdecl("mpsgraph_graph_scatter")
public func mpsgraph_graph_scatter(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ updatesTensorHandle: UnsafeMutableRawPointer?,
    _ indicesTensorHandle: UnsafeMutableRawPointer?,
    _ shape: UnsafePointer<UInt>?,
    _ shapeLen: Int,
    _ axis: Int,
    _ modeRaw: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.0, *) else {
        return nil
    }
    guard let graphHandle, let updatesTensorHandle, let indicesTensorHandle,
          let mode = mpsgraph_scatter_mode(modeRaw)
    else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let updatesTensor: MPSGraphTensor = mpsgraph_borrow(updatesTensorHandle)
    let indicesTensor: MPSGraphTensor = mpsgraph_borrow(indicesTensorHandle)
    return mpsgraph_retain(
        graph.scatter(
            updatesTensor,
            indices: indicesTensor,
            shape: mpsgraph_shape(shape, shapeLen),
            axis: axis,
            mode: mode,
            name: mpsgraph_optional_name(name)
        )
    )
}

@_cdecl("mpsgraph_graph_scatter_along_axis")
public func mpsgraph_graph_scatter_along_axis(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ axis: Int,
    _ updatesTensorHandle: UnsafeMutableRawPointer?,
    _ indicesTensorHandle: UnsafeMutableRawPointer?,
    _ shape: UnsafePointer<UInt>?,
    _ shapeLen: Int,
    _ modeRaw: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.3, *) else {
        return nil
    }
    guard let graphHandle, let updatesTensorHandle, let indicesTensorHandle,
          let mode = mpsgraph_scatter_mode(modeRaw)
    else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let updatesTensor: MPSGraphTensor = mpsgraph_borrow(updatesTensorHandle)
    let indicesTensor: MPSGraphTensor = mpsgraph_borrow(indicesTensorHandle)
    return mpsgraph_retain(
        graph.scatterAlongAxis(
            axis,
            updates: updatesTensor,
            indices: indicesTensor,
            shape: mpsgraph_shape(shape, shapeLen),
            mode: mode,
            name: mpsgraph_optional_name(name)
        )
    )
}

@_cdecl("mpsgraph_graph_sort")
public func mpsgraph_graph_sort(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ axis: Int,
    _ descending: Bool,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 13.0, *) else {
        return nil
    }
    guard let graphHandle, let tensorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let tensor: MPSGraphTensor = mpsgraph_borrow(tensorHandle)
    return mpsgraph_retain(graph.sort(tensor, axis: axis, descending: descending, name: mpsgraph_optional_name(name)))
}

@_cdecl("mpsgraph_graph_arg_sort")
public func mpsgraph_graph_arg_sort(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ axis: Int,
    _ descending: Bool,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 13.0, *) else {
        return nil
    }
    guard let graphHandle, let tensorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let tensor: MPSGraphTensor = mpsgraph_borrow(tensorHandle)
    return mpsgraph_retain(graph.argSort(tensor, axis: axis, descending: descending, name: mpsgraph_optional_name(name)))
}

@_cdecl("mpsgraph_graph_sparse_tensor_with_descriptor")
public func mpsgraph_graph_sparse_tensor_with_descriptor(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ descriptorHandle: UnsafeMutableRawPointer?,
    _ tensorHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ tensorCount: Int,
    _ shape: UnsafePointer<UInt>?,
    _ shapeLen: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.0, *) else {
        return nil
    }
    guard let graphHandle, let descriptorHandle,
          let tensors = mpsgraph_tensor_array(tensorHandles, count: tensorCount)
    else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let descriptor: MPSGraphCreateSparseOpDescriptor = mpsgraph_borrow(descriptorHandle)
    return mpsgraph_retain(
        graph.sparseTensor(
            sparseTensorWithDescriptor: descriptor,
            tensors: tensors,
            shape: mpsgraph_shape(shape, shapeLen),
            name: mpsgraph_optional_name(name)
        )
    )
}

@_cdecl("mpsgraph_graph_stencil")
public func mpsgraph_graph_stencil(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ sourceTensorHandle: UnsafeMutableRawPointer?,
    _ weightsTensorHandle: UnsafeMutableRawPointer?,
    _ descriptorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.0, *) else {
        return nil
    }
    guard let graphHandle, let sourceTensorHandle, let weightsTensorHandle, let descriptorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let sourceTensor: MPSGraphTensor = mpsgraph_borrow(sourceTensorHandle)
    let weightsTensor: MPSGraphTensor = mpsgraph_borrow(weightsTensorHandle)
    let descriptor: MPSGraphStencilOpDescriptor = mpsgraph_borrow(descriptorHandle)
    return mpsgraph_retain(
        graph.stencil(withSourceTensor: sourceTensor, weightsTensor: weightsTensor, descriptor: descriptor, name: mpsgraph_optional_name(name))
    )
}

@_cdecl("mpsgraph_graph_topk_gradient")
public func mpsgraph_graph_topk_gradient(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ gradientTensorHandle: UnsafeMutableRawPointer?,
    _ sourceTensorHandle: UnsafeMutableRawPointer?,
    _ k: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 14.0, *) else {
        return nil
    }
    guard let graphHandle, let gradientTensorHandle, let sourceTensorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let gradientTensor: MPSGraphTensor = mpsgraph_borrow(gradientTensorHandle)
    let sourceTensor: MPSGraphTensor = mpsgraph_borrow(sourceTensorHandle)
    return mpsgraph_retain(graph.topKGradient(gradientTensor, input: sourceTensor, k: k, name: mpsgraph_optional_name(name)))
}

@_cdecl("mpsgraph_execution_descriptor_wait_for_event")
public func mpsgraph_execution_descriptor_wait_for_event(
    _ handle: UnsafeMutableRawPointer?,
    _ eventHandle: UnsafeMutableRawPointer?,
    _ value: UInt64
) -> Bool {
    guard #available(macOS 13.0, *) else {
        return false
    }
    guard let handle, let event = mpsgraph_shared_event(eventHandle) else {
        return false
    }
    let descriptor: MPSGraphExecutionDescriptor = mpsgraph_borrow(handle)
    descriptor.wait(for: event, value: value)
    return true
}

@_cdecl("mpsgraph_execution_descriptor_signal_event")
public func mpsgraph_execution_descriptor_signal_event(
    _ handle: UnsafeMutableRawPointer?,
    _ eventHandle: UnsafeMutableRawPointer?,
    _ executionStageRaw: UInt64,
    _ value: UInt64
) -> Bool {
    guard #available(macOS 13.0, *) else {
        return false
    }
    guard let handle,
          let event = mpsgraph_shared_event(eventHandle),
          let executionStage = mpsgraph_execution_stage(executionStageRaw)
    else {
        return false
    }
    let descriptor: MPSGraphExecutionDescriptor = mpsgraph_borrow(handle)
    descriptor.signal(event, atExecutionEvent: executionStage, value: value)
    return true
}

@_cdecl("mpsgraph_executable_execution_descriptor_wait_for_event")
public func mpsgraph_executable_execution_descriptor_wait_for_event(
    _ handle: UnsafeMutableRawPointer?,
    _ eventHandle: UnsafeMutableRawPointer?,
    _ value: UInt64
) -> Bool {
    guard #available(macOS 13.0, *) else {
        return false
    }
    guard let handle, let event = mpsgraph_shared_event(eventHandle) else {
        return false
    }
    let descriptor: MPSGraphExecutableExecutionDescriptor = mpsgraph_borrow(handle)
    descriptor.wait(for: event, value: value)
    return true
}

@_cdecl("mpsgraph_executable_execution_descriptor_signal_event")
public func mpsgraph_executable_execution_descriptor_signal_event(
    _ handle: UnsafeMutableRawPointer?,
    _ eventHandle: UnsafeMutableRawPointer?,
    _ executionStageRaw: UInt64,
    _ value: UInt64
) -> Bool {
    guard #available(macOS 13.0, *) else {
        return false
    }
    guard let handle,
          let event = mpsgraph_shared_event(eventHandle),
          let executionStage = mpsgraph_execution_stage(executionStageRaw)
    else {
        return false
    }
    let descriptor: MPSGraphExecutableExecutionDescriptor = mpsgraph_borrow(handle)
    descriptor.signal(event, atExecutionEvent: executionStage, value: value)
    return true
}
