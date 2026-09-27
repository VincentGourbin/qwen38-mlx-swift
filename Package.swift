// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Qwen38MLXSwift",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "Qwen38Core", targets: ["Qwen38Core"]),
        .library(name: "Qwen38Server", targets: ["Qwen38Server"]),
        .library(name: "Qwen38Agent", targets: ["Qwen38Agent"]),
        .executable(name: "qwen38", targets: ["Qwen38CLI"]),
        .executable(name: "qwen38-bench-ui", targets: ["Qwen38BenchUI"]),
    ],
    dependencies: [
        // Keep MLX versions deliberate: even patch releases have changed APIs.
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.31.6"),
        // Local pinned checkout of upstream PR #545 (based on post-#351 MTP support).
        // Switch back to a tagged revision only once the Qwen MTP changes are released.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", branch: "main"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.3"),
        // Rendu des gabarits de chat : 2.5 aligne `tojson` sur `json.dumps`
        // de Python (séparateurs, `/` non échappé, `ensure_ascii=False`),
        // ce que transformers utilise — un `tojson` différent change les
        // jetons du prompt outillé que le modèle voit.
        .package(url: "https://github.com/huggingface/swift-jinja", from: "2.5.1"),
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
                .product(name: "Jinja", package: "swift-jinja"),
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
        // Panneau Agent (GUI) : garde de chemin, troncature des sorties
        // d'outils et machine à états de la boucle, en Foundation pur — sans
        // AppKit/SwiftUI ni MLX — pour rester testable sans checkpoint ni
        // réseau (Tests/Qwen38Tests). Le client HTTP qui parle au serveur
        // vit dans Qwen38BenchUI, pas ici : ce module ne fait aucun appel
        // réseau lui-même.
        .target(
            name: "Qwen38Agent"
        ),
        .executableTarget(
            name: "Qwen38BenchUI",
            dependencies: ["Qwen38Core", "Qwen38Server", "Qwen38Agent"]
        ),
        .testTarget(
            name: "Qwen38Tests",
            dependencies: [
                "Qwen38Core", "Qwen38Server", "Qwen38Agent",
                .product(name: "Jinja", package: "swift-jinja"),
            ]
        ),
    ]
)
