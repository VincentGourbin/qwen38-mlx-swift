// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Qwen38MLXSwift",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "Qwen38Core", targets: ["Qwen38Core"]),
        .library(name: "Qwen38Server", targets: ["Qwen38Server"]),
        .executable(name: "qwen38", targets: ["Qwen38CLI"]),
        .executable(name: "qwen38-bench-ui", targets: ["Qwen38BenchUI"]),
    ],
    dependencies: [
        // Keep MLX versions deliberate: even patch releases have changed APIs.
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.6"),
        // Local pinned checkout of upstream PR #545 (based on post-#351 MTP support).
        // Switch back to a tagged revision only once the Qwen MTP changes are released.
        .package(path: "Vendor/mlx-swift-lm"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.8.2"),
        .package(url: "https://github.com/VincentGourbin/swift-mlx-profiler", from: "1.5.0"),
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.0.0"),
    ],
    targets: [
        .target(
            name: "Qwen38Core",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "MLXProfiler", package: "swift-mlx-profiler"),
            ]
        ),
        .target(
            name: "Qwen38Server",
            dependencies: [
                "Qwen38Core",
                .product(name: "Hummingbird", package: "hummingbird"),
            ]
        ),
        .executableTarget(
            name: "Qwen38CLI",
            dependencies: [
                "Qwen38Core",
                "Qwen38Server",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXProfiler", package: "swift-mlx-profiler"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .executableTarget(
            name: "Qwen38BenchUI",
            dependencies: ["Qwen38Core", "Qwen38Server"]
        ),
        .testTarget(
            name: "Qwen38Tests",
            dependencies: ["Qwen38Core", "Qwen38Server"]
        ),
    ]
)
