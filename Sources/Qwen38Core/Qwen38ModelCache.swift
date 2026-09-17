import Foundation

/// Local model discovery.
///
/// The root is resolved once, in this order:
///   1. `customModelsDirectory`, set programmatically (CLI `--model-path`, GUI, tests);
///   2. `$QWEN38_MODELS_DIR` — use it to point at an external volume, e.g.
///      `export QWEN38_MODELS_DIR=/Volumes/MySSD/models`;
///   3. `~/models` when that directory exists;
///   4. `~/Library/Caches/models` otherwise (the download destination).
public enum Qwen38ModelCache {
    /// `$QWEN38_MODELS_DIR` if set, else `~/models`.
    public static let defaultModelDirectory: URL = {
        let environment = ProcessInfo.processInfo.environment["QWEN38_MODELS_DIR"] ?? ""
        if !environment.isEmpty {
            return URL(fileURLWithPath: (environment as NSString).expandingTildeInPath, isDirectory: true)
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("models", isDirectory: true)
    }()

    /// Override used by the CLI, GUI, and tests. Set this once during process
    /// startup, before loading a model.
    nonisolated(unsafe) public static var customModelsDirectory: URL?

    public static var modelsDirectory: URL {
        if let customModelsDirectory {
            return customModelsDirectory
        }
        if FileManager.default.fileExists(atPath: defaultModelDirectory.path) {
            return defaultModelDirectory
        }
        return FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("models", isDirectory: true)
    }

    public static func path(for modelID: String, under root: URL = modelsDirectory) -> URL {
        modelID.split(separator: "/").reduce(root) { partial, component in
            partial.appendingPathComponent(String(component), isDirectory: true)
        }
    }

    public static func isModelDirectory(_ directory: URL) -> Bool {
        let fm = FileManager.default
        guard fm.fileExists(atPath: directory.appendingPathComponent("config.json").path) else {
            return false
        }
        return (try? fm.contentsOfDirectory(atPath: directory.path))?.contains {
            $0.hasSuffix(".safetensors") || $0 == "model.safetensors.index.json"
        } == true
    }

    public static func diskSize(of directory: URL) -> Int64 {
        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey]
        ) else { return 0 }

        return enumerator.reduce(into: Int64(0)) { total, item in
            guard let url = item as? URL,
                  let values = try? url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey]),
                  values.isRegularFile == true,
                  let size = values.fileSize else { return }
            total += Int64(size)
        }
    }
}
