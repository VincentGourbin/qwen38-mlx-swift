import Foundation

public struct Qwen38DownloadFile: Sendable, Equatable {
    public let path: String
    public let size: Int64?

    public init(path: String, size: Int64?) {
        self.path = path
        self.size = size
    }
}

public struct Qwen38DownloadProgress: Sendable, Equatable {
    public let modelID: String
    public let file: Qwen38DownloadFile
    public let fileIndex: Int
    public let fileCount: Int
    public let bytesReceived: Int64
    public let completedBytes: Int64
    public let totalBytes: Int64?
    public let bytesPerSecond: Double
    public let skipped: Bool

    public var fractionCompleted: Double? {
        guard let totalBytes, totalBytes > 0 else { return nil }
        return Double(completedBytes + bytesReceived) / Double(totalBytes)
    }

    public init(
        modelID: String,
        file: Qwen38DownloadFile,
        fileIndex: Int,
        fileCount: Int,
        bytesReceived: Int64,
        completedBytes: Int64,
        totalBytes: Int64?,
        bytesPerSecond: Double,
        skipped: Bool
    ) {
        self.modelID = modelID
        self.file = file
        self.fileIndex = fileIndex
        self.fileCount = fileCount
        self.bytesReceived = bytesReceived
        self.completedBytes = completedBytes
        self.totalBytes = totalBytes
        self.bytesPerSecond = bytesPerSecond
        self.skipped = skipped
    }
}

public enum Qwen38DownloadError: LocalizedError, Equatable {
    case invalidModelID(String)
    case invalidRemotePath(String)
    case manifestRequestFailed(Int)
    case downloadRequestFailed(String, Int)
    case missingContent(String)
    case invalidContentLength(String)

    public var errorDescription: String? {
        switch self {
        case .invalidModelID(let id): return "Identifiant Hugging Face invalide : \(id)"
        case .invalidRemotePath(let path): return "Chemin distant invalide : \(path)"
        case .manifestRequestFailed(let status):
            return "Le manifeste Hugging Face a répondu HTTP \(status)."
        case .downloadRequestFailed(let path, let status):
            return "Le téléchargement de \(path) a répondu HTTP \(status)."
        case .missingContent(let path): return "Fichier téléchargé absent : \(path)"
        case .invalidContentLength(let path):
            return "Taille Content-Length incohérente pour \(path)."
        }
    }
}

/// Direct Hugging Face downloader. It deliberately does not use `Hub`: large
/// LFS redirects have stalled in the swift-huggingface downloader in previous
/// projects. Completed files are reused, so a cancelled run resumes at the
/// next file without keeping a second full model copy.
public actor Qwen38DownloadManager {
    public typealias ProgressHandler = @Sendable (Qwen38DownloadProgress) -> Void

    private let apiBaseURL: URL
    private let contentBaseURL: URL

    public init(
        apiBaseURL: URL = URL(string: "https://huggingface.co/api")!,
        contentBaseURL: URL = URL(string: "https://huggingface.co")!
    ) {
        self.apiBaseURL = apiBaseURL
        self.contentBaseURL = contentBaseURL
    }

    public func listFiles(
        modelID: String,
        token: String? = nil
    ) async throws -> [Qwen38DownloadFile] {
        guard Self.isValidModelID(modelID) else {
            throw Qwen38DownloadError.invalidModelID(modelID)
        }

        var url = apiBaseURL
            .appendingPathComponent("models", isDirectory: true)
        for component in modelID.split(separator: "/") {
            url.appendPathComponent(String(component), isDirectory: true)
        }
        url = url
            .appendingPathComponent("tree", isDirectory: true)
            .appendingPathComponent("main", isDirectory: true)
        var components = URLComponents(url: url, resolvingAgainstBaseURL: false)!
        components.queryItems = [
            URLQueryItem(name: "recursive", value: "true"),
            URLQueryItem(name: "expand", value: "1"),
            // HF currently rejects limits above 100 for the index tree API.
            URLQueryItem(name: "limit", value: "100"),
        ]
        url = components.url!

        var request = URLRequest(url: url)
        request.timeoutInterval = 30
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        addAuthorization(token, to: &request)
        let session = Self.makeSession(resourceTimeout: 60)
        defer { session.invalidateAndCancel() }
        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw Qwen38DownloadError.manifestRequestFailed(-1)
        }
        guard (200..<300).contains(http.statusCode) else {
            throw Qwen38DownloadError.manifestRequestFailed(http.statusCode)
        }

        let entries = try JSONDecoder().decode([HFEntry].self, from: data)
        return entries
            .filter { $0.type == "file" }
            .compactMap { entry in
                guard Self.isSafeRemotePath(entry.path) else { return nil }
                return Qwen38DownloadFile(path: entry.path, size: entry.size)
            }
            .filter(Self.isModelFile)
            .sorted { $0.path < $1.path }
    }

    @discardableResult
    public func download(
        modelID: String,
        to destination: URL = Qwen38ModelCache.modelsDirectory,
        token: String? = nil,
        progress: ProgressHandler? = nil
    ) async throws -> URL {
        let files = try await listFiles(modelID: modelID, token: token)
        guard !files.isEmpty else { throw Qwen38DownloadError.missingContent(modelID) }

        let modelDirectory = Qwen38ModelCache.path(for: modelID, under: destination)
        try FileManager.default.createDirectory(at: modelDirectory, withIntermediateDirectories: true)

        let totalBytes = files.reduce(into: Int64(0)) { total, file in
            if let size = file.size { total += size }
        }
        let hasTotal = totalBytes > 0 ? totalBytes : nil
        let speedWindow = Qwen38SpeedWindow()
        var completedBytes: Int64 = 0

        for (offset, file) in files.enumerated() {
            let target = modelDirectory.appendingPathComponent(file.path)
            try FileManager.default.createDirectory(
                at: target.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )

            if isComplete(target: target, expectedSize: file.size) {
                let size = file.size ?? fileSize(target)
                completedBytes += size
                speedWindow.record(totalBytes: completedBytes)
                progress?(Qwen38DownloadProgress(
                    modelID: modelID, file: file, fileIndex: offset,
                    fileCount: files.count, bytesReceived: size,
                    completedBytes: completedBytes - size, totalBytes: hasTotal,
                    bytesPerSecond: speedWindow.rate, skipped: true
                ))
                continue
            }

            var remoteURL = contentBaseURL
            for component in modelID.split(separator: "/") {
                remoteURL.appendPathComponent(String(component), isDirectory: true)
            }
            remoteURL = remoteURL
                .appendingPathComponent("resolve", isDirectory: true)
                .appendingPathComponent("main", isDirectory: true)
                .appendingPathComponent(file.path)
            let staging = target
                .deletingLastPathComponent()
                .appendingPathComponent(".\(target.lastPathComponent).part-\(UUID().uuidString)")

            var request = URLRequest(url: remoteURL)
            request.timeoutInterval = 30
            request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
            addAuthorization(token, to: &request)
            // Native URLSession download() does not expose byte callbacks.  A
            // sample at the beginning and end still gives a useful completed
            // file rate (and avoids reporting 0 KB/s for every large shard).
            speedWindow.record(totalBytes: completedBytes)

            let downloaded = try await Self.downloadFile(
                request: request,
                stagingURL: staging
            )
            defer { try? FileManager.default.removeItem(at: downloaded) }

            if let expected = file.size, fileSize(downloaded) != expected {
                throw Qwen38DownloadError.invalidContentLength(file.path)
            }
            if FileManager.default.fileExists(atPath: target.path) {
                try FileManager.default.removeItem(at: target)
            }
            try FileManager.default.moveItem(at: downloaded, to: target)
            let size = file.size ?? fileSize(target)
            completedBytes += size
            speedWindow.record(totalBytes: completedBytes)
            progress?(Qwen38DownloadProgress(
                modelID: modelID, file: file, fileIndex: offset,
                fileCount: files.count, bytesReceived: size,
                completedBytes: completedBytes - size, totalBytes: hasTotal,
                bytesPerSecond: speedWindow.rate, skipped: false
            ))
        }

        return modelDirectory
    }

    private static func downloadFile(
        request: URLRequest,
        stagingURL: URL
    ) async throws -> URL {
        let session = makeSession(resourceTimeout: 24 * 60 * 60)
        defer { session.invalidateAndCancel() }
        let (location, response) = try await session.download(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw Qwen38DownloadError.downloadRequestFailed(
                request.url?.lastPathComponent ?? "", -1
            )
        }
        guard (200..<300).contains(http.statusCode) else {
            throw Qwen38DownloadError.downloadRequestFailed(
                request.url?.lastPathComponent ?? "", http.statusCode
            )
        }
        try FileManager.default.moveItem(at: location, to: stagingURL)
        return stagingURL
    }

    private static func makeSession(resourceTimeout: TimeInterval) -> URLSession {
        let configuration = URLSessionConfiguration.default
        configuration.waitsForConnectivity = false
        configuration.timeoutIntervalForRequest = 30
        configuration.timeoutIntervalForResource = resourceTimeout
        return URLSession(configuration: configuration)
    }

    private func addAuthorization(_ token: String?, to request: inout URLRequest) {
        if let token, !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
    }

    private func isComplete(target: URL, expectedSize: Int64?) -> Bool {
        guard FileManager.default.fileExists(atPath: target.path) else { return false }
        guard let expectedSize else { return true }
        return fileSize(target) == expectedSize
    }

    private func fileSize(_ url: URL) -> Int64 {
        let values = try? url.resourceValues(forKeys: [.fileSizeKey])
        return Int64(values?.fileSize ?? 0)
    }

    private struct HFEntry: Decodable {
        let type: String
        let path: String
        let size: Int64?
    }

    private static func isValidModelID(_ modelID: String) -> Bool {
        let components = modelID.split(separator: "/")
        return components.count == 2 && components.allSatisfy {
            !$0.isEmpty && !$0.contains("..")
        }
    }

    private static func isSafeRemotePath(_ path: String) -> Bool {
        !path.isEmpty && !path.hasPrefix("/") &&
            !path.split(separator: "/").contains { $0 == ".." }
    }

    private static func isModelFile(_ file: Qwen38DownloadFile) -> Bool {
        let path = file.path.lowercased()
        return path == "config.json" || path == "generation_config.json" ||
            path == "tokenizer.json" || path == "tokenizer_config.json" ||
            path == "chat_template.jinja" || path == "preprocessor_config.json" ||
            path.hasSuffix(".safetensors") || path.hasSuffix(".json")
    }
}

private final class Qwen38SpeedWindow: @unchecked Sendable {
    private struct Sample {
        let time: Date
        let bytes: Int64
    }

    private let lock = NSLock()
    private var samples: [Sample] = []
    private var lastTotalBytes: Int64 = 0

    var rate: Double {
        lock.lock()
        defer { lock.unlock() }
        guard let first = samples.first, let last = samples.last else { return 0 }
        let elapsed = last.time.timeIntervalSince(first.time)
        return elapsed > 0 ? Double(last.bytes - first.bytes) / elapsed : 0
    }

    func record(totalBytes: Int64) {
        lock.lock()
        defer { lock.unlock() }
        let now = Date()
        guard totalBytes >= lastTotalBytes else { return }
        lastTotalBytes = totalBytes
        samples.append(Sample(time: now, bytes: totalBytes))
        let cutoff = now.addingTimeInterval(-3)
        samples.removeAll { $0.time < cutoff }
    }
}
