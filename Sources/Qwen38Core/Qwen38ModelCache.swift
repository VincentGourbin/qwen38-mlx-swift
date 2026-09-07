import Foundation

/// Local model discovery for the external Lexar volume and the user cache.
public enum Qwen38ModelCache {
    public static let defaultModelDirectory = URL(fileURLWithPath: "/Volumes/Lexar/models", isDirectory: true)

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
