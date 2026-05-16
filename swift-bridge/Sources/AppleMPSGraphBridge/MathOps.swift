import Foundation
import MetalPerformanceShadersGraph

@_cdecl("mpsgraph_graph_arithmetic_unary")
public func mpsgraph_graph_arithmetic_unary(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ op: UInt32,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let tensorHandle else {
        return nil
    }

    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let tensor: MPSGraphTensor = mpsgraph_borrow(tensorHandle)
    let name = mpsgraph_optional_name(name)
    let result: MPSGraphTensor

    switch op {
    case 0: result = graph.identity(with: tensor, name: name)
    case 1: result = graph.exponent(with: tensor, name: name)
    case 2: result = graph.exponentBase2(with: tensor, name: name)
    case 3: result = graph.exponentBase10(with: tensor, name: name)
    case 4: result = graph.logarithm(with: tensor, name: name)
    case 5: result = graph.logarithmBase2(with: tensor, name: name)
    case 6: result = graph.logarithmBase10(with: tensor, name: name)
    case 7: result = graph.square(with: tensor, name: name)
    case 8: result = graph.squareRoot(with: tensor, name: name)
    case 9: result = graph.reciprocal(with: tensor, name: name)
    case 10: result = graph.absolute(with: tensor, name: name)
    case 11: result = graph.negative(with: tensor, name: name)
    case 12: result = graph.sign(with: tensor, name: name)
    case 13: result = graph.signbit(with: tensor, name: name)
    case 14: result = graph.ceil(with: tensor, name: name)
    case 15: result = graph.floor(with: tensor, name: name)
    case 16: result = graph.round(with: tensor, name: name)
    case 17: result = graph.rint(with: tensor, name: name)
    case 18: result = graph.sin(with: tensor, name: name)
    case 19: result = graph.cos(with: tensor, name: name)
    case 20: result = graph.tan(with: tensor, name: name)
    case 21: result = graph.sinh(with: tensor, name: name)
    case 22: result = graph.cosh(with: tensor, name: name)
    case 23: result = graph.tanh(with: tensor, name: name)
    case 24: result = graph.asin(with: tensor, name: name)
    case 25: result = graph.acos(with: tensor, name: name)
    case 26: result = graph.atan(with: tensor, name: name)
    case 27: result = graph.asinh(with: tensor, name: name)
    case 28: result = graph.acosh(with: tensor, name: name)
    case 29: result = graph.atanh(with: tensor, name: name)
    case 30: result = graph.isNaN(with: tensor, name: name)
    case 31: result = graph.isInfinite(with: tensor, name: name)
    default: return nil
    }

    return mpsgraph_retain(result)
}

@_cdecl("mpsgraph_graph_arithmetic_binary")
public func mpsgraph_graph_arithmetic_binary(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ op: UInt32,
    _ primaryHandle: UnsafeMutableRawPointer?,
    _ secondaryHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let primaryHandle, let secondaryHandle else {
        return nil
    }

    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let primary: MPSGraphTensor = mpsgraph_borrow(primaryHandle)
    let secondary: MPSGraphTensor = mpsgraph_borrow(secondaryHandle)
    let name = mpsgraph_optional_name(name)
    let result: MPSGraphTensor

    switch op {
    case 0: result = graph.addition(primary, secondary, name: name)
    case 1: result = graph.subtraction(primary, secondary, name: name)
    case 2: result = graph.multiplication(primary, secondary, name: name)
    case 3: result = graph.division(primary, secondary, name: name)
    case 4: result = graph.divisionNoNaN(primary, secondary, name: name)
    case 5: result = graph.power(primary, secondary, name: name)
    case 6: result = graph.minimum(primary, secondary, name: name)
    case 7: result = graph.maximum(primary, secondary, name: name)
    case 8: result = graph.equal(primary, secondary, name: name)
    case 9: result = graph.notEqual(primary, secondary, name: name)
    case 10: result = graph.greaterThan(primary, secondary, name: name)
    case 11: result = graph.greaterThanOrEqualTo(primary, secondary, name: name)
    case 12: result = graph.lessThan(primary, secondary, name: name)
    case 13: result = graph.lessThanOrEqualTo(primary, secondary, name: name)
    case 14: result = graph.logicalAND(primary, secondary, name: name)
    case 15: result = graph.logicalOR(primary, secondary, name: name)
    case 16: result = graph.atan2(withPrimaryTensor: primary, secondaryTensor: secondary, name: name)
    case 17: result = graph.floorModulo(primary, secondary, name: name)
    default: return nil
    }

    return mpsgraph_retain(result)
}

@_cdecl("mpsgraph_graph_select")
public func mpsgraph_graph_select(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ predicateHandle: UnsafeMutableRawPointer?,
    _ trueHandle: UnsafeMutableRawPointer?,
    _ falseHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let predicateHandle, let trueHandle, let falseHandle else {
        return nil
    }

    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let predicate: MPSGraphTensor = mpsgraph_borrow(predicateHandle)
    let trueTensor: MPSGraphTensor = mpsgraph_borrow(trueHandle)
    let falseTensor: MPSGraphTensor = mpsgraph_borrow(falseHandle)
    return mpsgraph_retain(
        graph.select(predicate: predicate, trueTensor: trueTensor, falseTensor: falseTensor, name: mpsgraph_optional_name(name))
    )
}

@_cdecl("mpsgraph_graph_relu_gradient")
public func mpsgraph_graph_relu_gradient(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ gradientHandle: UnsafeMutableRawPointer?,
    _ sourceHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let gradientHandle, let sourceHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let gradient: MPSGraphTensor = mpsgraph_borrow(gradientHandle)
    let source: MPSGraphTensor = mpsgraph_borrow(sourceHandle)
    return mpsgraph_retain(graph.reLUGradient(withIncomingGradient: gradient, sourceTensor: source, name: mpsgraph_optional_name(name)))
}

@_cdecl("mpsgraph_graph_sigmoid_gradient")
public func mpsgraph_graph_sigmoid_gradient(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ gradientHandle: UnsafeMutableRawPointer?,
    _ sourceHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let gradientHandle, let sourceHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let gradient: MPSGraphTensor = mpsgraph_borrow(gradientHandle)
    let source: MPSGraphTensor = mpsgraph_borrow(sourceHandle)
    return mpsgraph_retain(graph.sigmoidGradient(withIncomingGradient: gradient, sourceTensor: source, name: mpsgraph_optional_name(name)))
}

@_cdecl("mpsgraph_graph_softmax_gradient")
public func mpsgraph_graph_softmax_gradient(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ gradientHandle: UnsafeMutableRawPointer?,
    _ sourceHandle: UnsafeMutableRawPointer?,
    _ axis: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let gradientHandle, let sourceHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let gradient: MPSGraphTensor = mpsgraph_borrow(gradientHandle)
    let source: MPSGraphTensor = mpsgraph_borrow(sourceHandle)
    return mpsgraph_retain(graph.softMaxGradient(withIncomingGradient: gradient, sourceTensor: source, axis: axis, name: mpsgraph_optional_name(name)))
}

@_cdecl("mpsgraph_graph_leaky_relu_scalar")
public func mpsgraph_graph_leaky_relu_scalar(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ alpha: Double,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.0, *) else {
        return nil
    }
    guard let graphHandle, let tensorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let tensor: MPSGraphTensor = mpsgraph_borrow(tensorHandle)
    return mpsgraph_retain(graph.leakyReLU(with: tensor, alpha: alpha, name: mpsgraph_optional_name(name)))
}

@_cdecl("mpsgraph_graph_leaky_relu_tensor")
public func mpsgraph_graph_leaky_relu_tensor(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ alphaTensorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.0, *) else {
        return nil
    }
    guard let graphHandle, let tensorHandle, let alphaTensorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let tensor: MPSGraphTensor = mpsgraph_borrow(tensorHandle)
    let alphaTensor: MPSGraphTensor = mpsgraph_borrow(alphaTensorHandle)
    return mpsgraph_retain(graph.leakyReLU(with: tensor, alphaTensor: alphaTensor, name: mpsgraph_optional_name(name)))
}

@_cdecl("mpsgraph_graph_leaky_relu_gradient")
public func mpsgraph_graph_leaky_relu_gradient(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ gradientHandle: UnsafeMutableRawPointer?,
    _ sourceHandle: UnsafeMutableRawPointer?,
    _ alphaTensorHandle: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 12.0, *) else {
        return nil
    }
    guard let graphHandle, let gradientHandle, let sourceHandle, let alphaTensorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let gradient: MPSGraphTensor = mpsgraph_borrow(gradientHandle)
    let source: MPSGraphTensor = mpsgraph_borrow(sourceHandle)
    let alphaTensor: MPSGraphTensor = mpsgraph_borrow(alphaTensorHandle)
    return mpsgraph_retain(graph.leakyReLUGradient(withIncomingGradient: gradient, sourceTensor: source, alphaTensor: alphaTensor, name: mpsgraph_optional_name(name)))
}

@_cdecl("mpsgraph_graph_reduction_axis")
public func mpsgraph_graph_reduction_axis(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ op: UInt32,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ axis: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let tensorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let tensor: MPSGraphTensor = mpsgraph_borrow(tensorHandle)
    let name = mpsgraph_optional_name(name)
    let result: MPSGraphTensor

    switch op {
    case 0: result = graph.reductionSum(with: tensor, axis: axis, name: name)
    case 1: result = graph.reductionMaximum(with: tensor, axis: axis, name: name)
    case 2: result = graph.reductionMinimum(with: tensor, axis: axis, name: name)
    case 3: result = graph.reductionProduct(with: tensor, axis: axis, name: name)
    default: return nil
    }

    return mpsgraph_retain(result)
}

@_cdecl("mpsgraph_graph_reduction_axes")
public func mpsgraph_graph_reduction_axes(
    _ graphHandle: UnsafeMutableRawPointer?,
    _ op: UInt32,
    _ tensorHandle: UnsafeMutableRawPointer?,
    _ axes: UnsafePointer<UInt>?,
    _ axesLen: Int,
    _ name: UnsafePointer<CChar>?
) -> UnsafeMutableRawPointer? {
    guard let graphHandle, let tensorHandle else {
        return nil
    }
    let graph: MPSGraph = mpsgraph_borrow(graphHandle)
    let tensor: MPSGraphTensor = mpsgraph_borrow(tensorHandle)
    let axes = mpsgraph_shape(axes, axesLen)
    let name = mpsgraph_optional_name(name)
    let result: MPSGraphTensor

    switch op {
    case 0: result = graph.reductionSum(with: tensor, axes: axes, name: name)
    case 1: result = graph.reductionMaximum(with: tensor, axes: axes, name: name)
    case 2: result = graph.reductionMinimum(with: tensor, axes: axes, name: name)
    case 3: result = graph.reductionProduct(with: tensor, axes: axes, name: name)
    default: return nil
    }

    return mpsgraph_retain(result)
}
