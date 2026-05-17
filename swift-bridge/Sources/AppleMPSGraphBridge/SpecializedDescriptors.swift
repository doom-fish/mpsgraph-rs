import Foundation
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

@inline(__always)
func mpsgraph_pooling_return_indices_mode(_ rawValue: UInt) -> MPSGraphPoolingReturnIndicesMode? {
    MPSGraphPoolingReturnIndicesMode(rawValue: rawValue)
}

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
    MPSGraphResizeNearestRoundingMode(rawValue: rawValue)
}

@inline(__always)
func mpsgraph_scatter_mode(_ rawValue: Int) -> MPSGraphScatterMode? {
    MPSGraphScatterMode(rawValue: rawValue)
}

@inline(__always)
func mpsgraph_sparse_storage_type(_ rawValue: UInt64) -> MPSGraphSparseStorageType? {
    MPSGraphSparseStorageType(rawValue: rawValue)
}

@inline(__always)
func mpsgraph_nms_coordinate_mode(_ rawValue: UInt) -> MPSGraphNonMaximumSuppressionCoordinateMode? {
    MPSGraphNonMaximumSuppressionCoordinateMode(rawValue: rawValue)
}

@inline(__always)
func mpsgraph_loss_reduction_type(_ rawValue: UInt64) -> MPSGraphLossReductionType? {
    MPSGraphLossReductionType(rawValue: rawValue)
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
        let weightsLayout = mpsgraph_data_layout(weightsLayoutRaw),
        let descriptor = MPSGraphConvolution3DOpDescriptor.descriptor(
            withStrideInX: strideInX,
            strideInY: strideInY,
            strideInZ: strideInZ,
            dilationRateInX: dilationRateInX,
            dilationRateInY: dilationRateInY,
            dilationRateInZ: dilationRateInZ,
            groups: groups,
            paddingLeft: paddingLeft,
            paddingRight: paddingRight,
            paddingTop: paddingTop,
            paddingBottom: paddingBottom,
            paddingFront: paddingFront,
            paddingBack: paddingBack,
            paddingStyle: paddingStyle,
            dataLayout: dataLayout,
            weightsLayout: weightsLayout
        )
    else {
        return nil
    }
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
        let weightsLayout = mpsgraph_data_layout(weightsLayoutRaw),
        let descriptor = MPSGraphDepthwiseConvolution2DOpDescriptor.descriptor(
            withStrideInX: strideInX,
            strideInY: strideInY,
            dilationRateInX: dilationRateInX,
            dilationRateInY: dilationRateInY,
            paddingLeft: paddingLeft,
            paddingRight: paddingRight,
            paddingTop: paddingTop,
            paddingBottom: paddingBottom,
            paddingStyle: paddingStyle,
            dataLayout: dataLayout,
            weightsLayout: weightsLayout
        )
    else {
        return nil
    }
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
          let paddingStyle = mpsgraph_padding_style(paddingStyleRaw),
          let descriptor = MPSGraphDepthwiseConvolution3DOpDescriptor.descriptor(
            withStrides: mpsgraph_shape(strides, stridesLen),
            dilationRates: mpsgraph_shape(dilationRates, dilationRatesLen),
            paddingValues: mpsgraph_shape(paddingValues, paddingValuesLen),
            paddingStyle: paddingStyle
          )
    else {
        return nil
    }
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
    guard let descriptor = MPSGraphFFTDescriptor.descriptor(),
          let scalingMode = mpsgraph_fft_scaling_mode(scalingModeRaw)
    else {
        return nil
    }
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
    guard let dataLayout = mpsgraph_data_layout(dataLayoutRaw),
          let descriptor = MPSGraphImToColOpDescriptor.descriptor(
            withKernelWidth: kernelWidth,
            kernelHeight: kernelHeight,
            strideInX: strideInX,
            strideInY: strideInY,
            dilationRateInX: dilationRateInX,
            dilationRateInY: dilationRateInY,
            paddingLeft: paddingLeft,
            paddingRight: paddingRight,
            paddingTop: paddingTop,
            paddingBottom: paddingBottom,
            dataLayout: dataLayout
          )
    else {
        return nil
    }
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
          let paddingStyle = mpsgraph_padding_style(paddingStyleRaw),
          let descriptor = MPSGraphPooling4DOpDescriptor.descriptor(
            withKernelSizes: mpsgraph_shape(kernelSizes, kernelSizesLen),
            strides: mpsgraph_shape(strides, stridesLen),
            dilationRates: mpsgraph_shape(dilationRates, dilationRatesLen),
            paddingValues: mpsgraph_shape(paddingValues, paddingValuesLen),
            paddingStyle: paddingStyle
          )
    else {
        return nil
    }
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
          let dataType = mpsgraph_data_type(dataTypeRaw),
          let descriptor = MPSGraphCreateSparseOpDescriptor.descriptor(withStorageType: storageType, dataType: dataType)
    else {
        return nil
    }
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
          let paddingStyle = mpsgraph_padding_style(paddingStyleRaw),
          let descriptor = MPSGraphStencilOpDescriptor.descriptor(
            withReductionMode: reductionMode,
            offsets: mpsgraph_optional_signed_shape(offsets, offsetsLen) ?? [],
            strides: mpsgraph_shape(strides, stridesLen),
            dilationRates: mpsgraph_shape(dilationRates, dilationRatesLen),
            explicitPadding: mpsgraph_shape(explicitPadding, explicitPaddingLen),
            boundaryMode: boundaryMode,
            paddingStyle: paddingStyle,
            paddingConstant: paddingConstant
          )
    else {
        return nil
    }
    return mpsgraph_retain(descriptor)
}
