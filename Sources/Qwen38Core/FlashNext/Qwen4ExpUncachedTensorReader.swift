import Foundation
#if canImport(Darwin)
import Darwin
#endif
import MLX

public enum Qwen4ExpUncachedTensorReaderError: LocalizedError, Equatable {
    case cannotOpen(URL, errno: Int32)
    case missingTensor(String)
    case shortRead(URL, expected: Int, got: Int)

    public var errorDescription: String? {
        switch self {
        case .cannotOpen(let url, let code):
            return "Impossible d'ouvrir \(url.path) sans cache (errno \(code))."
        case .missingTensor(let key):
            return "Tenseur absent du shard : \(key)."
        case .shortRead(let url, let expected, let got):
            return "Lecture tronquée sur \(url.path) : attendu \(expected) octets, lu \(got)."
        }
    }
}

/// Reads individual safetensors tensors with `pread`, after marking the file
/// descriptor `F_NOCACHE` so the read bytes never enter the kernel's unified
/// buffer cache.
///
/// `loadArraysAndMetadata` (mlx-swift's normal entry point) offers no such
/// control. Loading a Flash-Next checkpoint's 84-113 GB of resident tensors
/// through it fills the file cache to 42-49 GB before the kernel starts
/// evicting it under memory pressure, down to only a 9-16 GB floor — memory
/// stolen from the resident process and from every other open application
/// (see docs/knowledge/log.md, "H6 : deux tentatives", 2026-09-08 soir, and
/// PLAN.md task P2-mem-a). This reader is the fix: every resident tensor
/// read goes through `pread` into a heap buffer that never touches the page
/// cache, at the cost of losing the (unwanted, here) benefit of the OS
/// caching a re-read of the same bytes.
///
/// Header parsing is shared with the n-gram mmap reader via
/// `Qwen4ExpSafetensorsHeader` — this type does not parse safetensors JSON
/// itself.
public final class Qwen4ExpUncachedTensorReader {
    public let url: URL
    public let header: Qwen4ExpSafetensorsHeader
    private let fileDescriptor: Int32

    public init(url: URL) throws {
        self.url = url
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            throw Qwen4ExpUncachedTensorReaderError.cannotOpen(url, errno: errno)
        }
        // F_NOCACHE is best-effort: on a filesystem that refuses it, reads
        // still succeed and are still correct — only the memory benefit is
        // lost. Do not fail the whole load over that.
        _ = fcntl(fd, F_NOCACHE, 1)
        self.fileDescriptor = fd
        do {
            self.header = try Qwen4ExpSafetensorsHeader.read(fileDescriptor: fd, url: url)
        } catch {
            close(fd)
            throw error
        }
    }

    deinit {
        close(fileDescriptor)
    }

    /// Read one tensor and return it as an already-evaluated `MLXArray`,
    /// bit-exact with what `loadArraysAndMetadata(url:stream:)` would
    /// produce for the same key.
    public func array(for key: String) throws -> MLXArray {
        guard let descriptor = header.tensors[key] else {
            throw Qwen4ExpUncachedTensorReaderError.missingTensor(key)
        }
        let dtype = mlxDType(for: descriptor.dtype)
        let byteCount = descriptor.byteCount
        guard byteCount > 0 else {
            let array = MLXArray.zeros(descriptor.shape, dtype: dtype)
            eval(array)
            return array
        }

        var data = Data(count: byteCount)
        let absoluteStart = header.dataSectionStart + descriptor.dataOffsets.start
        let bytesRead = try data.withUnsafeMutableBytes { raw -> Int in
            guard let base = raw.baseAddress else { return 0 }
            return try Self.preadFull(
                fileDescriptor: fileDescriptor, buffer: base, count: byteCount,
                offset: off_t(absoluteStart))
        }
        guard bytesRead == byteCount else {
            throw Qwen4ExpUncachedTensorReaderError.shortRead(
                url, expected: byteCount, got: bytesRead)
        }

        let array = MLXArray(data, descriptor.shape, dtype: dtype)
        eval(array)
        return array
    }

    private func mlxDType(for dtype: Qwen4ExpSafetensorsDType) -> DType {
        switch dtype {
        case .uint8: return .uint8
        case .int8: return .int8
        case .int16: return .int16
        case .uint16: return .uint16
        case .int32: return .int32
        case .uint32: return .uint32
        case .int64: return .int64
        case .uint64: return .uint64
        case .float16: return .float16
        case .bfloat16: return .bfloat16
        case .float32: return .float32
        case .float64: return .float64
        case .bool: return .bool
        }
    }

    private static func preadFull(
        fileDescriptor: Int32, buffer: UnsafeMutableRawPointer, count: Int, offset: off_t
    ) throws -> Int {
        var totalRead = 0
        while totalRead < count {
            let n = pread(fileDescriptor, buffer + totalRead, count - totalRead, offset + off_t(totalRead))
            if n < 0 {
                if errno == EINTR { continue }
                break
            }
            if n == 0 { break }
            totalRead += n
        }
        return totalRead
    }
}
