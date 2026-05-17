import Foundation
import Metal
import MetalPerformanceShadersGraph

@_cdecl("mpsgraph_object_retain")
public func mpsgraph_object_retain(_ handle: UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer? {
    guard let handle else {
        return nil
    }
    let object = Unmanaged<AnyObject>.fromOpaque(handle).takeUnretainedValue()
    return Unmanaged.passRetained(object).toOpaque()
}

@inline(__always)
func mpsgraph_padding_mode(_ rawValue: Int) -> MPSGraphPaddingMode? {
    MPSGraphPaddingMode(rawValue: rawValue)
}

@inline(__always)
func mpsgraph_reduction_mode(_ rawValue: UInt) -> MPSGraphReductionMode? {
    MPSGraphReductionMode(rawValue: rawValue)
}

@available(macOS 12.2, *)
@inline(__always)
func mpsgraph_pooling_return_indices_mode(_ rawValue: UInt) -> MPSGraphPoolingReturnIndicesMode? {
    MPSGraphPoolingReturnIndicesMode(rawValue: rawValue)
}

@available(macOS 14.0, *)
@inline(__always)
func mpsgraph_fft_scaling_mode(_ rawValue: UInt) -> MPSGraphFFTScalingMode? {
    MPSGraphFFTScalingMode(rawValue: rawValue)
}

@inline(__always)
func mpsgraph_resize_mode(_ rawValue: UInt) -> MPSGraphResizeMode? {
    MPSGraphResizeMode(rawValue: rawValue)
}

@inline(__always)
func mpsgraph_resize_nearest_rounding_mode(_ rawValue: UInt) -> MPSGraphResizeNearestRoundingMode? {
    if rawValue >= 4 {
        guard #available(macOS 13.2, *) else {
            return nil
        }
    }
    return MPSGraphResizeNearestRoundingMode(rawValue: rawValue)
}

@inline(__always)
func mpsgraph_scatter_mode(_ rawValue: Int) -> MPSGraphScatterMode? {
    MPSGraphScatterMode(rawValue: rawValue)
}

@inline(__always)
func mpsgraph_sparse_storage_type(_ rawValue: UInt64) -> MPSGraphSparseStorageType? {
    MPSGraphSparseStorageType(rawValue: rawValue)
}

@available(macOS 14.0, *)
@inline(__always)
func mpsgraph_nms_coordinate_mode(_ rawValue: UInt) -> MPSGraphNonMaximumSuppressionCoordinateMode? {
    MPSGraphNonMaximumSuppressionCoordinateMode(rawValue: rawValue)
}

@inline(__always)
func mpsgraph_loss_reduction_type(_ rawValue: UInt64) -> MPSGraphLossReductionType? {
    switch rawValue {
    case 0:
        if #available(macOS 14.0, *) {
            return MPSGraphLossReductionType.none
        }
        return .axis
    case 1:
        return .sum
    case 2:
        guard #available(macOS 12.0, *) else {
            return nil
        }
        return .mean
    default:
        return nil
    }
}

@_cdecl("mpsgraph_operation_as_variable")
public func mpsgraph_operation_as_variable(_ handle: UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer? {
    guard let handle else {
        return nil
    }
    let operation: MPSGraphOperation = mpsgraph_borrow(handle)
    guard let variable = operation as? MPSGraphVariableOp else {
        return nil
    }
    return mpsgraph_retain(variable)
}

@_cdecl("mpsgraph_variable_op_shape_len")
public func mpsgraph_variable_op_shape_len(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let handle else {
        return 0
    }
    let variable: MPSGraphVariableOp = mpsgraph_borrow(handle)
    return variable.shape.count
}

@_cdecl("mpsgraph_variable_op_copy_shape")
public func mpsgraph_variable_op_copy_shape(
    _ handle: UnsafeMutableRawPointer?,
    _ outShape: UnsafeMutablePointer<Int>?
) {
    guard let handle, let outShape else {
        return
    }
    let variable: MPSGraphVariableOp = mpsgraph_borrow(handle)
    for (index, value) in variable.shape.enumerated() {
        outShape[index] = value.intValue
    }
}

@_cdecl("mpsgraph_variable_op_data_type")
public func mpsgraph_variable_op_data_type(_ handle: UnsafeMutableRawPointer?) -> UInt32 {
    guard let handle else {
        return 0
    }
    let variable: MPSGraphVariableOp = mpsgraph_borrow(handle)
    return variable.dataType.rawValue
}

@_cdecl("mpsgraph_convolution3d_descriptor_new")
public func mpsgraph_convolution3d_descriptor_new(
    _ strideInX: Int,
    _ strideInY: Int,
    _ strideInZ: Int,
    _ dilationRateInX: Int,
    _ dilationRateInY: Int,
    _ dilationRateInZ: Int,
    _ groups: Int,
    _ paddingLeft: Int,
    _ paddingRight: Int,
    _ paddingTop: Int,
    _ paddingBottom: Int,
    _ paddingFront: Int,
    _ paddingBack: Int,
    _ paddingStyleRaw: UInt,
    _ dataLayoutRaw: UInt,
    _ weightsLayoutRaw: UInt
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 13.2, *) else {
        return nil
    }
    guard
        let paddingStyle = mpsgraph_padding_style(paddingStyleRaw),
        let dataLayout = mpsgraph_data_layout(dataLayoutRaw),
        let weightsLayout = mpsgraph_data_layout(weightsLayoutRaw)
    else {
        return nil
    }

    let descriptor = MPSGraphConvolution3DOpDescriptor()
    descriptor.strideInX = strideInX
    descriptor.strideInY = strideInY
    descriptor.strideInZ = strideInZ
    descriptor.dilationRateInX = dilationRateInX
    descriptor.dilationRateInY = dilationRateInY
    descriptor.dilationRateInZ = dilationRateInZ
    descriptor.groups = groups
    descriptor.paddingLeft = paddingLeft
    descriptor.paddingRight = paddingRight
    descriptor.paddingTop = paddingTop
    descriptor.paddingBottom = paddingBottom
    descriptor.paddingFront = paddingFront
    descriptor.paddingBack = paddingBack
    descriptor.paddingStyle = paddingStyle
    descriptor.dataLayout = dataLayout
    descriptor.weightsLayout = weightsLayout
    return mpsgraph_retain(descriptor)
}

@_cdecl("mpsgraph_depthwise_convolution2d_descriptor_new")
public func mpsgraph_depthwise_convolution2d_descriptor_new(
    _ strideInX: Int,
    _ strideInY: Int,
    _ dilationRateInX: Int,
    _ dilationRateInY: Int,
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

    let descriptor = MPSGraphDepthwiseConvolution2DOpDescriptor()
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
    descriptor.weightsLayout = weightsLayout
    return mpsgraph_retain(descriptor)
}

@_cdecl("mpsgraph_depthwise_convolution3d_descriptor_new")
public func mpsgraph_depthwise_convolution3d_descriptor_new(
    _ strides: UnsafePointer<UInt>?,
    _ stridesLen: Int,
    _ dilationRates: UnsafePointer<UInt>?,
    _ dilationRatesLen: Int,
    _ paddingValues: UnsafePointer<UInt>?,
    _ paddingValuesLen: Int,
    _ paddingStyleRaw: UInt,
    _ channelDimensionIndex: Int
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.0, *) else {
        return nil
    }
    guard stridesLen == 3, dilationRatesLen == 3, paddingValuesLen == 6,
          let paddingStyle = mpsgraph_padding_style(paddingStyleRaw)
    else {
        return nil
    }

    let descriptor = MPSGraphDepthwiseConvolution3DOpDescriptor()
    descriptor.strides = mpsgraph_shape(strides, stridesLen)
    descriptor.dilationRates = mpsgraph_shape(dilationRates, dilationRatesLen)
    descriptor.paddingValues = mpsgraph_shape(paddingValues, paddingValuesLen)
    descriptor.paddingStyle = paddingStyle
    descriptor.channelDimensionIndex = channelDimensionIndex
    return mpsgraph_retain(descriptor)
}

@_cdecl("mpsgraph_fft_descriptor_new")
public func mpsgraph_fft_descriptor_new(
    _ inverse: Bool,
    _ scalingModeRaw: UInt,
    _ roundToOddHermitean: Bool
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 14.0, *) else {
        return nil
    }
    guard let scalingMode = mpsgraph_fft_scaling_mode(scalingModeRaw) else {
        return nil
    }

    let descriptor = MPSGraphFFTDescriptor()
    descriptor.inverse = inverse
    descriptor.scalingMode = scalingMode
    descriptor.roundToOddHermitean = roundToOddHermitean
    return mpsgraph_retain(descriptor)
}

@_cdecl("mpsgraph_im_to_col_descriptor_new")
public func mpsgraph_im_to_col_descriptor_new(
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
    _ dataLayoutRaw: UInt
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 14.0, *) else {
        return nil
    }
    guard let dataLayout = mpsgraph_data_layout(dataLayoutRaw) else {
        return nil
    }

    let descriptor = MPSGraphImToColOpDescriptor()
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
    descriptor.dataLayout = dataLayout
    return mpsgraph_retain(descriptor)
}

@_cdecl("mpsgraph_pooling4d_descriptor_new")
public func mpsgraph_pooling4d_descriptor_new(
    _ kernelSizes: UnsafePointer<UInt>?,
    _ kernelSizesLen: Int,
    _ strides: UnsafePointer<UInt>?,
    _ stridesLen: Int,
    _ dilationRates: UnsafePointer<UInt>?,
    _ dilationRatesLen: Int,
    _ paddingValues: UnsafePointer<UInt>?,
    _ paddingValuesLen: Int,
    _ paddingStyleRaw: UInt,
    _ ceilMode: Bool,
    _ includeZeroPadToAverage: Bool,
    _ returnIndicesModeRaw: UInt,
    _ returnIndicesDataTypeRaw: UInt32
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.0, *) else {
        return nil
    }
    guard kernelSizesLen == 4, stridesLen == 4, dilationRatesLen == 4, paddingValuesLen == 8,
          let paddingStyle = mpsgraph_padding_style(paddingStyleRaw)
    else {
        return nil
    }

    let descriptor = MPSGraphPooling4DOpDescriptor()
    descriptor.kernelSizes = mpsgraph_shape(kernelSizes, kernelSizesLen)
    descriptor.strides = mpsgraph_shape(strides, stridesLen)
    descriptor.dilationRates = mpsgraph_shape(dilationRates, dilationRatesLen)
    descriptor.paddingValues = mpsgraph_shape(paddingValues, paddingValuesLen)
    descriptor.paddingStyle = paddingStyle
    descriptor.ceilMode = ceilMode
    descriptor.includeZeroPadToAverage = includeZeroPadToAverage
    if #available(macOS 12.2, *) {
        guard let returnIndicesMode = mpsgraph_pooling_return_indices_mode(returnIndicesModeRaw),
              let returnIndicesDataType = mpsgraph_data_type(returnIndicesDataTypeRaw)
        else {
            return nil
        }
        descriptor.returnIndicesMode = returnIndicesMode
        descriptor.returnIndicesDataType = returnIndicesDataType
    } else if returnIndicesModeRaw != 0 {
        return nil
    }
    return mpsgraph_retain(descriptor)
}

@_cdecl("mpsgraph_sparse_descriptor_new")
public func mpsgraph_sparse_descriptor_new(
    _ storageTypeRaw: UInt64,
    _ dataTypeRaw: UInt32
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.0, *) else {
        return nil
    }
    guard let storageType = mpsgraph_sparse_storage_type(storageTypeRaw),
          let dataType = mpsgraph_data_type(dataTypeRaw)
    else {
        return nil
    }

    let descriptor = MPSGraphCreateSparseOpDescriptor()
    descriptor.sparseStorageType = storageType
    descriptor.dataType = dataType
    return mpsgraph_retain(descriptor)
}

@_cdecl("mpsgraph_stencil_descriptor_new")
public func mpsgraph_stencil_descriptor_new(
    _ reductionModeRaw: UInt,
    _ offsets: UnsafePointer<Int>?,
    _ offsetsLen: Int,
    _ strides: UnsafePointer<UInt>?,
    _ stridesLen: Int,
    _ dilationRates: UnsafePointer<UInt>?,
    _ dilationRatesLen: Int,
    _ explicitPadding: UnsafePointer<UInt>?,
    _ explicitPaddingLen: Int,
    _ boundaryModeRaw: Int,
    _ paddingStyleRaw: UInt,
    _ paddingConstant: Float
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.0, *) else {
        return nil
    }
    guard offsetsLen == 4, stridesLen == 4, dilationRatesLen == 4, explicitPaddingLen == 8,
          let reductionMode = mpsgraph_reduction_mode(reductionModeRaw),
          let boundaryMode = mpsgraph_padding_mode(boundaryModeRaw),
          let paddingStyle = mpsgraph_padding_style(paddingStyleRaw)
    else {
        return nil
    }

    let descriptor = MPSGraphStencilOpDescriptor()
    descriptor.reductionMode = reductionMode
    descriptor.offsets = mpsgraph_optional_signed_shape(offsets, offsetsLen) ?? []
    descriptor.strides = mpsgraph_shape(strides, stridesLen)
    descriptor.dilationRates = mpsgraph_shape(dilationRates, dilationRatesLen)
    descriptor.explicitPadding = mpsgraph_shape(explicitPadding, explicitPaddingLen)
    descriptor.boundaryMode = boundaryMode
    descriptor.paddingStyle = paddingStyle
    descriptor.paddingConstant = paddingConstant
    return mpsgraph_retain(descriptor)
}
