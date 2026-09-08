import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// The dtype tags used in a safetensors header (the `"dtype"` field of each
/// tensor entry). Only the ones actually emitted by `mlx.save`/the Flash-Next
/// converters are exercised today, but the full safetensors set is listed
/// here so a new checkpoint dtype fails with a clear error instead of a
/// silent skip.
public enum Qwen4ExpSafetensorsDType: String, Sendable {
    case uint8 = "U8"
    case int8 = "I8"
    case int16 = "I16"
    case uint16 = "U16"
    case int32 = "I32"
    case uint32 = "U32"
    case int64 = "I64"
    case uint64 = "U64"
    case float16 = "F16"
    case bfloat16 = "BF16"
    case float32 = "F32"
    case float64 = "F64"
    case bool = "BOOL"

    /// Size in bytes of a single element, matching safetensors' own layout
    /// (no padding between elements).
    public var byteWidth: Int {
        switch self {
        case .uint8, .int8, .bool: return 1
        case .int16, .uint16, .float16, .bfloat16: return 2
        case .int32, .uint32, .float32: return 4
        case .int64, .uint64, .float64: return 8
        }
    }
}

/// One tensor entry from a safetensors header.
public struct Qwen4ExpSafetensorsTensorDescriptor: Sendable {
    public let dtype: Qwen4ExpSafetensorsDType
    public let shape: [Int]
    /// Byte range, relative to the start of the data section (i.e. relative
    /// to `Qwen4ExpSafetensorsHeader.dataSectionStart`), exactly as encoded
    /// in the safetensors `data_offsets` field.
    public let dataOffsets: (start: UInt64, end: UInt64)

    public var byteCount: Int { Int(dataOffsets.end - dataOffsets.start) }
    public var elementCount: Int { shape.reduce(1, *) }
}

public enum Qwen4ExpSafetensorsHeaderError: LocalizedError, Equatable {
    case truncated(URL)
    case invalidJSON(URL)

    public var errorDescription: String? {
        switch self {
        case .truncated(let url):
            return "En-tête safetensors tronqué : \(url.path)"
        case .invalidJSON(let url):
            return "En-tête safetensors invalide : \(url.path)"
        }
    }
}

/// A parsed safetensors header: the byte offset where the data section
/// starts, plus every tensor's dtype/shape/byte-range.
///
/// A safetensors file is `<8-byte little-endian header length N>
/// <N bytes of JSON> <raw tensor data>`. `data_offsets` in the JSON is
/// relative to the end of the header, so `dataSectionStart` (`8 + N`) is the
/// value every caller needs to turn those into absolute file offsets.
///
/// This is the single parser for that format in the codebase: both the
/// n-gram mmap reader (`Qwen4ExpLazyNGramStorage`, which keeps reading the
/// bytes itself through `mmap`) and the `F_NOCACHE` resident-tensor reader
/// (`Qwen4ExpUncachedTensorReader`) call into this type instead of each
/// re-parsing the header (see PLAN.md §6.3, "n'écrire qu'un seul parseur").
public struct Qwen4ExpSafetensorsHeader: Sendable {
    public let dataSectionStart: UInt64
    public let tensors: [String: Qwen4ExpSafetensorsTensorDescriptor]

    /// Absolute byte range (from the start of the file) for one tensor.
    public func absoluteRange(for key: String) -> (start: UInt64, end: UInt64)? {
        guard let descriptor = tensors[key] else { return nil }
        return (
            dataSectionStart + descriptor.dataOffsets.start,
            dataSectionStart + descriptor.dataOffsets.end
        )
    }

    /// Parse using a regular buffered `FileHandle` read. Suitable for a
    /// caller that will map or read the rest of the file through its own
    /// path (e.g. the n-gram table's POSIX `mmap`).
    public static func read(url: URL) throws -> Qwen4ExpSafetensorsHeader {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        return try parse(url: url) { offset, count in
            try handle.seek(toOffset: offset)
            guard let data = try handle.read(upToCount: count), data.count == count else {
                throw Qwen4ExpSafetensorsHeaderError.truncated(url)
            }
            return data
        }
    }

    /// Parse using `pread` at an already-open file descriptor. Lets a caller
    /// that opened the shard with `fcntl(F_NOCACHE, 1)` read the (tiny)
    /// header without a second, regular (cached) `open` of the same file.
    public static func read(fileDescriptor: Int32, url: URL) throws -> Qwen4ExpSafetensorsHeader {
        try parse(url: url) { offset, count in
            var buffer = Data(count: count)
            let bytesRead = buffer.withUnsafeMutableBytes { raw -> Int in
                guard let base = raw.baseAddress else { return 0 }
                return pread(fileDescriptor, base, count, off_t(offset))
            }
            guard bytesRead == count else {
                throw Qwen4ExpSafetensorsHeaderError.truncated(url)
            }
            return buffer
        }
    }

    private static func parse(
        url: URL, readAt: (UInt64, Int) throws -> Data
    ) throws -> Qwen4ExpSafetensorsHeader {
        let lengthData = try readAt(0, 8)
        let headerLength = lengthData.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> UInt64 in
            var value = UInt64(0)
            for index in 0 ..< 8 {
                value |= UInt64(raw[index]) << UInt64(index * 8)
            }
            return value
        }
        guard headerLength <= UInt64(Int.max) else {
            throw Qwen4ExpSafetensorsHeaderError.invalidJSON(url)
        }
        let headerData = try readAt(8, Int(headerLength))
        guard let object = try? JSONSerialization.jsonObject(with: headerData)
                as? [String: Any] else {
            throw Qwen4ExpSafetensorsHeaderError.invalidJSON(url)
        }

        var tensors = [String: Qwen4ExpSafetensorsTensorDescriptor]()
        for (key, value) in object {
            // `__metadata__` carries free-form string metadata, not a tensor.
            guard key != "__metadata__" else { continue }
            guard let tensor = value as? [String: Any],
                  let dtypeString = tensor["dtype"] as? String,
                  let dtype = Qwen4ExpSafetensorsDType(rawValue: dtypeString),
                  let shapeValues = tensor["shape"] as? [Any],
                  let offsetValues = tensor["data_offsets"] as? [Any],
                  offsetValues.count == 2
            else { continue }
            let shape = shapeValues.compactMap { ($0 as? NSNumber)?.intValue }
            guard shape.count == shapeValues.count,
                  let start = (offsetValues[0] as? NSNumber)?.uint64Value,
                  let end = (offsetValues[1] as? NSNumber)?.uint64Value
            else { continue }
            tensors[key] = Qwen4ExpSafetensorsTensorDescriptor(
                dtype: dtype, shape: shape, dataOffsets: (start, end))
        }
        return Qwen4ExpSafetensorsHeader(
            dataSectionStart: 8 + headerLength, tensors: tensors)
    }
}
