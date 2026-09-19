// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Calc",
    products: [.library(name: "Calc", targets: ["Calc"])],
    targets: [
        .target(name: "Calc"),
        .testTarget(name: "CalcTests", dependencies: ["Calc"]),
    ]
)
