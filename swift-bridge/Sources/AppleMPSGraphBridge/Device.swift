import Foundation
import Metal
import MetalPerformanceShadersGraph

@_cdecl("mpsgraph_device_new_with_metal_device")
public func mpsgraph_device_new_with_metal_device(
    _ metalDeviceHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let metalDeviceHandle else {
        return nil
    }
    let metalDevice: MTLDevice = mpsgraph_borrow(metalDeviceHandle)
    return mpsgraph_retain(MPSGraphDevice(mtlDevice: metalDevice))
}

@_cdecl("mpsgraph_device_type")
public func mpsgraph_device_type(_ handle: UnsafeMutableRawPointer?) -> UInt32 {
    guard let handle else {
        return UInt32.max
    }
    let device: MPSGraphDevice = mpsgraph_borrow(handle)
    return device.type.rawValue
}
