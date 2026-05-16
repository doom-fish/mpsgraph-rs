// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "AppleMPSGraphBridge",
    platforms: [
        .macOS(.v11),
    ],
    products: [
        .library(name: "AppleMPSGraphBridge", type: .static, targets: ["AppleMPSGraphBridge"]),
    ],
    targets: [
        .target(
            name: "AppleMPSGraphBridge",
            dependencies: [],
            path: "Sources/AppleMPSGraphBridge",
            publicHeadersPath: "include"
        ),
    ]
)
