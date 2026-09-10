import Foundation
#if canImport(Darwin)
import Darwin
#endif
import MLX
import MLXNN

public struct Qwen4ExpNGramCacheStats: Sendable, Equatable {
    public var hits: Int
    public var misses: Int
    public var entries: Int
    /// P4.5: cumulative wall time spent in the uncached row read (mmap
    /// touch on a miss — cold Lexar pages vs already-faulted-in pages) that
    /// backs every miss counted above. `ContinuousClock`, always on.
    public var missDuration: TimeInterval

    public init(
        hits: Int = 0, misses: Int = 0, entries: Int = 0, missDuration: TimeInterval = 0
    ) {
        self.hits = hits
        self.misses = misses
        self.entries = entries
        self.missDuration = missDuration
    }

    public var lookups: Int { hits + misses }

    public var hitRate: Double? {
        guard lookups > 0 else { return nil }
        return Double(hits) / Double(lookups)
    }

    /// Average wall time per miss (P4.5's "latence d'un miss").
    public var meanMissDuration: TimeInterval? {
        guard misses > 0 else { return nil }
        return missDuration / Double(misses)
    }
}

/// P6.2: instrumentation for the PLE lookup path — how many `MLXArray(...)`
/// host→device tensor constructions and `MLX.dequantized` calls a lookup
/// batch emits, and how much host wall time went into the row reads that
/// feed them. Measured once per resident model (the storage object is
/// created once in `.resident` mode, see `Qwen38FlashNextEngine`), reset
/// with `resetLookupStats()` before a probe run.
public struct Qwen4ExpPLELookupStats: Sendable, Equatable {
    public var lookupCalls: Int
    public var arraysConstructed: Int
    public var dequantizeCalls: Int
    public var hostReadSeconds: TimeInterval

    public init(
        lookupCalls: Int = 0, arraysConstructed: Int = 0, dequantizeCalls: Int = 0,
        hostReadSeconds: TimeInterval = 0
    ) {
        self.lookupCalls = lookupCalls
        self.arraysConstructed = arraysConstructed
        self.dequantizeCalls = dequantizeCalls
        self.hostReadSeconds = hostReadSeconds
    }
}

/// A row-wise reader for Flash-Next's very large quantized n-gram table.
///
/// A normal MLX gather over a lazy safetensors array can materialize the
/// complete source tensor on the device.  Flash-Next's table is split into
/// 128 shards, so that would make a single PLE lookup consume roughly 32 GB.
/// This reader keeps only the small safetensors headers and reads the rows
/// requested by the current token batch before uploading them to MLX.
public final class Qwen4ExpLazyNGramStorage: @unchecked Sendable {
    private struct TensorLocation {
        let url: URL
        let dataStart: UInt64
        let rowBytes: Int
        let rowCount: Int
    }

    private struct ShardLocation {
        let weight: TensorLocation
        let scales: TensorLocation
        let biases: TensorLocation?
    }

    private let shards: [Int: ShardLocation]
    private let groupSize: Int
    private let bits: Int
    private let mode: QuantizationMode
    private let packedWidth: Int
    private let scaleWidth: Int
    /// The n-gram tensors stay virtual: POSIX `mmap` maps the checkpoint files
    /// without copying their payload into RAM. Only pages touched by a lookup
    /// are faulted in by the operating system.
    private let mappedFiles: [URL: MappedFile]

    private final class MappedFile: @unchecked Sendable {
        let baseAddress: UnsafeMutableRawPointer
        let byteCount: Int

        init?(url: URL) {
            let fileDescriptor = open(url.path, O_RDONLY)
            guard fileDescriptor >= 0 else { return nil }
            var status = stat()
            guard fstat(fileDescriptor, &status) == 0,
                  status.st_size > 0,
                  let size = Int(exactly: status.st_size) else {
                close(fileDescriptor)
                return nil
            }
            let mapped = mmap(nil, size, PROT_READ, MAP_PRIVATE, fileDescriptor, 0)
            close(fileDescriptor)
            guard mapped != MAP_FAILED else { return nil }
            self.baseAddress = mapped!
            self.byteCount = size
        }

        deinit {
            munmap(baseAddress, byteCount)
        }
    }

    public struct CacheStats: Sendable, Equatable {
        public let hits: Int
        public let misses: Int
        public let entries: Int
        public let missDuration: TimeInterval

        fileprivate init(hits: Int, misses: Int, entries: Int, missDuration: TimeInterval) {
            self.hits = hits
            self.misses = misses
            self.entries = entries
            self.missDuration = missDuration
        }

        public var publicStats: Qwen4ExpNGramCacheStats {
            Qwen4ExpNGramCacheStats(
                hits: hits, misses: misses, entries: entries, missDuration: missDuration)
        }
    }

    private enum CachedRowKind: UInt8 {
        case packed
        case half
    }

    private struct CachedRowKey: Hashable {
        let url: URL
        let dataStart: UInt64
        let row: Int
        let kind: CachedRowKind
    }

    private enum CachedRowValue {
        case packed([UInt32])
        case half([UInt16])
    }

    private final class RowCache: @unchecked Sendable {
        private let capacity: Int
        private let lock = NSLock()
        private var values: [CachedRowKey: CachedRowValue] = [:]
        private var order: [CachedRowKey] = []
        private var hits = 0
        private var misses = 0
        /// P4.5: cumulative wall time of the uncached reads that back every
        /// miss above (recorded by the caller via `recordMissDuration`,
        /// since the actual mmap-backed read happens outside this lock).
        private var missDuration: TimeInterval = 0

        init(capacity: Int) {
            precondition(capacity > 0)
            self.capacity = capacity
        }

        func packed(location: TensorLocation, row: Int) -> [UInt32]? {
            value(location: location, row: row, kind: .packed) { value in
                guard case .packed(let packed) = value else { return nil }
                return packed
            }
        }

        func half(location: TensorLocation, row: Int) -> [UInt16]? {
            value(location: location, row: row, kind: .half) { value in
                guard case .half(let half) = value else { return nil }
                return half
            }
        }

        func storePacked(_ packed: [UInt32], location: TensorLocation, row: Int) {
            store(.packed(packed), location: location, row: row, kind: .packed)
        }

        func storeHalf(_ half: [UInt16], location: TensorLocation, row: Int) {
            store(.half(half), location: location, row: row, kind: .half)
        }

        func stats() -> CacheStats {
            lock.lock()
            defer { lock.unlock() }
            return CacheStats(
                hits: hits, misses: misses, entries: values.count, missDuration: missDuration)
        }

        /// P4.5: called once per batch of uncached rows actually fetched
        /// (`readPackedRows`/`readUInt16Rows`), with the wall time of that
        /// fetch — cold Lexar pages the first time a shard's mmap region is
        /// touched, already-faulted-in pages afterwards.
        func recordMissDuration(_ duration: TimeInterval) {
            lock.lock()
            defer { lock.unlock() }
            missDuration += duration
        }

        private func value<T>(
            location: TensorLocation, row: Int, kind: CachedRowKind,
            project: (CachedRowValue) -> T?
        ) -> T? {
            let key = CachedRowKey(
                url: location.url, dataStart: location.dataStart, row: row, kind: kind)
            lock.lock()
            defer { lock.unlock() }
            guard let value = values[key], let projected = project(value) else {
                misses += 1
                return nil
            }
            hits += 1
            order.removeAll { $0 == key }
            order.append(key)
            return projected
        }

        private func store(
            _ value: CachedRowValue, location: TensorLocation, row: Int, kind: CachedRowKind
        ) {
            let key = CachedRowKey(
                url: location.url, dataStart: location.dataStart, row: row, kind: kind)
            lock.lock()
            defer { lock.unlock() }
            values[key] = value
            order.removeAll { $0 == key }
            order.append(key)
            while order.count > capacity {
                values.removeValue(forKey: order.removeFirst())
            }
        }
    }

    private let rowCache = RowCache(capacity: 4096)

    /// P6.2 instrumentation box: counts `MLXArray(...)` constructions and
    /// `MLX.dequantized` calls emitted per lookup, plus the host wall time
    /// spent in the row reads that feed them. NSLock-protected like
    /// `RowCache` above (lookups can run concurrently with `asyncEval`).
    private final class LookupStatsBox: @unchecked Sendable {
        private let lock = NSLock()
        private var lookupCalls = 0
        private var arraysConstructed = 0
        private var dequantizeCalls = 0
        private var hostReadSeconds: TimeInterval = 0

        func record(arrays: Int, dequantize: Int, readSeconds: TimeInterval) {
            lock.lock()
            defer { lock.unlock() }
            lookupCalls += 1
            arraysConstructed += arrays
            dequantizeCalls += dequantize
            hostReadSeconds += readSeconds
        }

        func snapshot() -> Qwen4ExpPLELookupStats {
            lock.lock()
            defer { lock.unlock() }
            return Qwen4ExpPLELookupStats(
                lookupCalls: lookupCalls, arraysConstructed: arraysConstructed,
                dequantizeCalls: dequantizeCalls, hostReadSeconds: hostReadSeconds)
        }

        func reset() {
            lock.lock()
            defer { lock.unlock() }
            lookupCalls = 0
            arraysConstructed = 0
            dequantizeCalls = 0
            hostReadSeconds = 0
        }
    }

    private let lookupStatsBox = LookupStatsBox()

    public init(
        directory: URL,
        rawKeysByShardFile: [String: [String]],
        layerIndex: Int,
        shardCount: Int,
        dimensions: Int,
        quantization: Qwen4ExpQuantizationSpec
    ) throws {
        self.groupSize = quantization.groupSize
        self.bits = quantization.bits
        self.mode = quantization.mode
        precondition(dimensions % quantization.groupSize == 0)
        self.packedWidth = dimensions * quantization.bits / 32
        self.scaleWidth = dimensions / quantization.groupSize

        var locations = [Int: [String: TensorLocation]]()
        for (shardFile, rawKeys) in rawKeysByShardFile {
            let url = directory.appendingPathComponent(shardFile)
            let header = try Self.readHeader(url: url)
            for rawKey in rawKeys where rawKey.contains(".ngram_embedding.shard_") {
                guard let marker = rawKey.range(of: ".ngram_embedding.shard_") else {
                    continue
                }
                let suffix = rawKey[marker.upperBound...]
                guard let separator = suffix.firstIndex(of: "."),
                      let shard = Int(suffix[..<separator]) else {
                    continue
                }
                let kind = String(suffix[suffix.index(after: separator)...])
                guard let descriptor = header[rawKey] else {
                    throw Qwen4ExpCheckpointSliceLoaderError.missingTensor(rawKey)
                }
                let location = TensorLocation(
                    url: url,
                    dataStart: descriptor.dataStart,
                    rowBytes: descriptor.rowBytes,
                    rowCount: descriptor.rowCount)
                locations[shard, default: [:]][kind] = location
            }
        }

        guard locations.count == shardCount else {
            throw Qwen4ExpCheckpointSliceLoaderError.invalidIndex(
                directory.appendingPathComponent("config.json"))
        }
        let mappedURLs = Set(locations.values.flatMap { entries in
            entries.values.map(\.url)
        })
        var mappings = [URL: MappedFile]()
        for url in mappedURLs {
            // Use POSIX mmap instead of Data(mappedIfSafe): Foundation may
            // copy an ExFAT file when it decides that mapping is unsafe. The
            // explicit MAP_PRIVATE path remains virtual on the USB volume.
            if let mapping = MappedFile(url: url) { mappings[url] = mapping }
        }
        self.mappedFiles = mappings
        var result = [Int: ShardLocation]()
        for shard in 0 ..< shardCount {
            guard let entries = locations[shard],
                  let weight = entries["weight"],
                  let scales = entries["scales"],
                  weight.rowBytes == packedWidth * MemoryLayout<UInt32>.size,
                  scales.rowBytes == scaleWidth * MemoryLayout<UInt16>.size else {
                throw Qwen4ExpCheckpointSliceLoaderError.invalidIndex(
                    directory.appendingPathComponent("config.json"))
            }
            result[shard] = ShardLocation(
                weight: weight, scales: scales, biases: entries["biases"])
        }
        self.shards = result
        _ = layerIndex
    }

    public func lookup(shard: Int, rows: [Int32]) -> MLXArray {
        guard let location = shards[shard] else {
            preconditionFailure("Shard n-gram inexistante: \(shard)")
        }
        let rowIndices = rows.map(Int.init)
        precondition(rowIndices.allSatisfy { $0 >= 0 && $0 < location.weight.rowCount })
        let readStart = ContinuousClock.now
        let packed = readPackedRows(location.weight, rows: rowIndices)
        let scales = readUInt16Rows(location.scales, rows: rowIndices)
        let biases = location.biases.map { readUInt16Rows($0, rows: rowIndices) }
        let readSeconds = (ContinuousClock.now - readStart).seconds
        let packedArray = MLXArray(packed).reshaped([rows.count, packedWidth])
        let scalesArray = MLXArray(scales).reshaped([rows.count, scaleWidth])
            // Safetensors stores BF16 as its raw UInt16 bit pattern. `asType`
            // performs a numeric conversion; `view` preserves those bits.
            .view(dtype: .bfloat16)
        let biasesArray = biases.map {
            MLXArray($0).reshaped([rows.count, scaleWidth]).view(dtype: .bfloat16)
        }
        let result = MLX.dequantized(
            packedArray, scales: scalesArray, biases: biasesArray,
            groupSize: groupSize, bits: bits, mode: mode)
        lookupStatsBox.record(
            arrays: biasesArray == nil ? 2 : 3, dequantize: 1, readSeconds: readSeconds)
        return result
    }

    /// P6.2: the batched counterpart of `lookup(shard:rows:)` — one request
    /// list spanning every shard touched by a token batch, one packed array,
    /// one scales array, one biases array (if any) and a single
    /// `MLX.dequantized` call, instead of one of each per distinct shard.
    /// Row bytes are still read per-shard (`readPackedRows`/`readUInt16Rows`
    /// batch contiguous runs within a shard already, P5.4), but the result
    /// is assembled into flat host buffers **in request order** before a
    /// single upload, so the caller gets rows back already ordered by
    /// position — no on-device scatter/add needed afterwards.
    public func lookupBatch(_ requests: [(shard: Int, row: Int32)]) -> MLXArray {
        precondition(!requests.isEmpty)
        var positionsByShard: [Int: [Int32]] = [:]
        for (index, request) in requests.enumerated() {
            positionsByShard[request.shard, default: []].append(Int32(index))
        }
        var packedFlat = [UInt32](repeating: 0, count: requests.count * packedWidth)
        var scalesFlat = [UInt16](repeating: 0, count: requests.count * scaleWidth)
        var biasesFlat: [UInt16]?
        let readStart = ContinuousClock.now
        for (shard, positions) in positionsByShard {
            guard let location = shards[shard] else {
                preconditionFailure("Shard n-gram inexistante: \(shard)")
            }
            let rowIndices = positions.map { Int(requests[Int($0)].row) }
            precondition(rowIndices.allSatisfy { $0 >= 0 && $0 < location.weight.rowCount })
            let packedRows = readPackedRows(location.weight, rows: rowIndices)
            let scaleRows = readUInt16Rows(location.scales, rows: rowIndices)
            let biasRows = location.biases.map { readUInt16Rows($0, rows: rowIndices) }
            for (localIndex, globalPosition) in positions.enumerated() {
                let globalIndex = Int(globalPosition)
                packedFlat.replaceSubrange(
                    (globalIndex * packedWidth) ..< ((globalIndex + 1) * packedWidth),
                    with: packedRows[(localIndex * packedWidth) ..< ((localIndex + 1) * packedWidth)])
                scalesFlat.replaceSubrange(
                    (globalIndex * scaleWidth) ..< ((globalIndex + 1) * scaleWidth),
                    with: scaleRows[(localIndex * scaleWidth) ..< ((localIndex + 1) * scaleWidth)])
                if let biasRows {
                    if biasesFlat == nil {
                        biasesFlat = [UInt16](repeating: 0, count: requests.count * scaleWidth)
                    }
                    biasesFlat!.replaceSubrange(
                        (globalIndex * scaleWidth) ..< ((globalIndex + 1) * scaleWidth),
                        with: biasRows[(localIndex * scaleWidth) ..< ((localIndex + 1) * scaleWidth)])
                }
            }
        }
        let readSeconds = (ContinuousClock.now - readStart).seconds
        let packedArray = MLXArray(packedFlat).reshaped([requests.count, packedWidth])
        let scalesArray = MLXArray(scalesFlat).reshaped([requests.count, scaleWidth])
            .view(dtype: .bfloat16)
        let biasesArray = biasesFlat.map {
            MLXArray($0).reshaped([requests.count, scaleWidth]).view(dtype: .bfloat16)
        }
        let result = MLX.dequantized(
            packedArray, scales: scalesArray, biases: biasesArray,
            groupSize: groupSize, bits: bits, mode: mode)
        lookupStatsBox.record(
            arrays: biasesArray == nil ? 2 : 3, dequantize: 1, readSeconds: readSeconds)
        return result
    }

    public func cacheStats() -> CacheStats {
        rowCache.stats()
    }

    public func lookupStats() -> Qwen4ExpPLELookupStats {
        lookupStatsBox.snapshot()
    }

    public func resetLookupStats() {
        lookupStatsBox.reset()
    }

    private func readPackedRows(_ location: TensorLocation, rows: [Int]) -> [UInt32] {
        guard !rows.isEmpty else { return [] }
        var cached = [[UInt32]?](repeating: nil, count: rows.count)
        var missing = [Int]()
        var missingSet = Set<Int>()
        for (index, row) in rows.enumerated() {
            if let value = rowCache.packed(location: location, row: row) {
                cached[index] = value
            } else if missingSet.insert(row).inserted {
                missing.append(row)
            }
        }
        if !missing.isEmpty {
            let missStart = ContinuousClock.now
            let fetched = readPackedRowsUncached(location, rows: missing)
            rowCache.recordMissDuration((ContinuousClock.now - missStart).seconds)
            for row in missing {
                guard let value = fetched[row] else {
                    preconditionFailure("Lecture n-gram incomplète: \(location.url.path):\(row)")
                }
                rowCache.storePacked(value, location: location, row: row)
            }
        }
        return rows.enumerated().flatMap { index, row in
            if let value = cached[index] { return value }
            guard let value = rowCache.packed(location: location, row: row) else {
                preconditionFailure("Cache n-gram incohérent pour \(location.url.path):\(row)")
            }
            return value
        }
    }

    private func readPackedRowsUncached(_ location: TensorLocation, rows: [Int]) -> [Int: [UInt32]] {
        if let mapping = mappedFiles[location.url] {
            let raw = UnsafeRawBufferPointer(
                start: UnsafeRawPointer(mapping.baseAddress), count: mapping.byteCount)
            let start = Int(location.dataStart)
            let byteWidth = location.rowBytes
            let end = start + (rows.max() ?? 0) * byteWidth + byteWidth
            guard start >= 0, end <= raw.count else {
                preconditionFailure("Mapping safetensors invalide: \(location.url.path)")
            }
            return Self.readContiguousRuns(UInt32.self, mapping: raw, start: start, byteWidth: byteWidth, rows: rows)
        }
        return Self.readPackedRowsFromFile(location, rows: rows)
    }

    private static func readPackedRowsFromFile(_ location: TensorLocation, rows: [Int]) -> [Int: [UInt32]] {
        guard let handle = try? FileHandle(forReadingFrom: location.url) else {
            preconditionFailure("Impossible d'ouvrir \(location.url.path)")
        }
        defer { try? handle.close() }
        return Self.readContiguousRunsFromFile(
            UInt32.self, handle: handle, dataStart: location.dataStart, byteWidth: location.rowBytes,
            rows: rows, url: location.url)
    }

    /// P5.4: bulk row reads for the n-gram table's mmap-backed shards.
    ///
    /// The original implementation read every requested row's bytes one
    /// `UInt32`/`UInt16` at a time via manual little-endian shift-assembly
    /// (avoiding unaligned typed loads). Measured contribution on the real
    /// 3-bit checkpoint: the PLE layer (couche 1) alone took 2.75 s of a
    /// 9.4 s pure-forward 2 543-token prefill — 29 % of total forward time
    /// across 48 layers, against ~135 ms for a typical non-PLE/QSA layer
    /// (docs/knowledge/log.md "P5.4", `results/p54/chat.trace.json`).
    /// Sorting the requested rows and copying each *contiguous* run in one
    /// `copyMemory` call reproduces the exact same bytes — ARM64 Apple
    /// Silicon is little-endian, matching the manual reconstruction bit for
    /// bit — with a single bulk copy per run instead of one loop iteration
    /// per element.
    private static func readContiguousRuns<Element: FixedWidthInteger & UnsignedInteger>(
        _ type: Element.Type, mapping raw: UnsafeRawBufferPointer, start: Int, byteWidth: Int,
        rows: [Int]
    ) -> [Int: [Element]] {
        var result: [Int: [Element]] = [:]
        guard !rows.isEmpty else { return result }
        result.reserveCapacity(rows.count)
        let elementsPerRow = byteWidth / MemoryLayout<Element>.size
        let sortedRows = rows.sorted()
        var i = 0
        while i < sortedRows.count {
            var j = i
            while j + 1 < sortedRows.count, sortedRows[j + 1] == sortedRows[j] + 1 {
                j += 1
            }
            let firstRow = sortedRows[i]
            let rangeRowCount = j - i + 1
            let rangeByteCount = rangeRowCount * byteWidth
            let sourceStart = start + firstRow * byteWidth
            var rangeValues = [Element](repeating: 0, count: rangeRowCount * elementsPerRow)
            rangeValues.withUnsafeMutableBytes { destination in
                destination.copyMemory(
                    from: UnsafeRawBufferPointer(rebasing: raw[sourceStart ..< sourceStart + rangeByteCount]))
            }
            for offset in 0 ..< rangeRowCount {
                let sliceStart = offset * elementsPerRow
                result[firstRow + offset] = Array(rangeValues[sliceStart ..< sliceStart + elementsPerRow])
            }
            i = j + 1
        }
        return result
    }

    /// `FileHandle` counterpart of `readContiguousRuns` for the (untested in
    /// production — every shard is mmap-backed today) non-mapped fallback:
    /// one `read(upToCount:)` per contiguous run instead of one per row.
    private static func readContiguousRunsFromFile<Element: FixedWidthInteger & UnsignedInteger>(
        _ type: Element.Type, handle: FileHandle, dataStart: UInt64, byteWidth: Int, rows: [Int],
        url: URL
    ) -> [Int: [Element]] {
        var result: [Int: [Element]] = [:]
        guard !rows.isEmpty else { return result }
        result.reserveCapacity(rows.count)
        let elementsPerRow = byteWidth / MemoryLayout<Element>.size
        let sortedRows = rows.sorted()
        var i = 0
        while i < sortedRows.count {
            var j = i
            while j + 1 < sortedRows.count, sortedRows[j + 1] == sortedRows[j] + 1 {
                j += 1
            }
            let firstRow = sortedRows[i]
            let rangeRowCount = j - i + 1
            let rangeByteCount = rangeRowCount * byteWidth
            do {
                try handle.seek(toOffset: dataStart + UInt64(firstRow * byteWidth))
                guard let data = try handle.read(upToCount: rangeByteCount),
                      data.count == rangeByteCount else {
                    preconditionFailure("Lecture safetensors tronquée: \(url.path)")
                }
                data.withUnsafeBytes { raw in
                    for offset in 0 ..< rangeRowCount {
                        let elementStart = offset * elementsPerRow
                        let byteStart = elementStart * MemoryLayout<Element>.size
                        let byteEnd = byteStart + elementsPerRow * MemoryLayout<Element>.size
                        var rowValues = [Element](repeating: 0, count: elementsPerRow)
                        rowValues.withUnsafeMutableBytes { destination in
                            destination.copyMemory(
                                from: UnsafeRawBufferPointer(rebasing: raw[byteStart ..< byteEnd]))
                        }
                        result[firstRow + offset] = rowValues
                    }
                }
            } catch {
                preconditionFailure("Lecture safetensors impossible: \(error)")
            }
            i = j + 1
        }
        return result
    }

    private func readUInt16Rows(_ location: TensorLocation, rows: [Int]) -> [UInt16] {
        guard !rows.isEmpty else { return [] }
        var cached = [[UInt16]?](repeating: nil, count: rows.count)
        var missing = [Int]()
        var missingSet = Set<Int>()
        for (index, row) in rows.enumerated() {
            if let value = rowCache.half(location: location, row: row) {
                cached[index] = value
            } else if missingSet.insert(row).inserted {
                missing.append(row)
            }
        }
        if !missing.isEmpty {
            let missStart = ContinuousClock.now
            let fetched = readUInt16RowsUncached(location, rows: missing)
            rowCache.recordMissDuration((ContinuousClock.now - missStart).seconds)
            for row in missing {
                guard let value = fetched[row] else {
                    preconditionFailure("Lecture n-gram incomplète: \(location.url.path):\(row)")
                }
                rowCache.storeHalf(value, location: location, row: row)
            }
        }
        return rows.enumerated().flatMap { index, row in
            if let value = cached[index] { return value }
            guard let value = rowCache.half(location: location, row: row) else {
                preconditionFailure("Cache n-gram incohérent pour \(location.url.path):\(row)")
            }
            return value
        }
    }

    private func readUInt16RowsUncached(_ location: TensorLocation, rows: [Int]) -> [Int: [UInt16]] {
        if let mapping = mappedFiles[location.url] {
            let raw = UnsafeRawBufferPointer(
                start: UnsafeRawPointer(mapping.baseAddress), count: mapping.byteCount)
            let start = Int(location.dataStart)
            let byteWidth = location.rowBytes
            let end = start + (rows.max() ?? 0) * byteWidth + byteWidth
            guard start >= 0, end <= raw.count else {
                preconditionFailure("Mapping safetensors invalide: \(location.url.path)")
            }
            return Self.readContiguousRuns(UInt16.self, mapping: raw, start: start, byteWidth: byteWidth, rows: rows)
        }
        return Self.readUInt16RowsFromFile(location, rows: rows)
    }

    private static func readUInt16RowsFromFile(_ location: TensorLocation, rows: [Int]) -> [Int: [UInt16]] {
        guard let handle = try? FileHandle(forReadingFrom: location.url) else {
            preconditionFailure("Impossible d'ouvrir \(location.url.path)")
        }
        defer { try? handle.close() }
        return Self.readContiguousRunsFromFile(
            UInt16.self, handle: handle, dataStart: location.dataStart, byteWidth: location.rowBytes,
            rows: rows, url: location.url)
    }

    private struct HeaderDescriptor {
        let dataStart: UInt64
        let rowBytes: Int
        let rowCount: Int
        let dtype: String
    }

    /// Adapts the shared `Qwen4ExpSafetensorsHeader` parser to this reader's
    /// row-wise shape (`dataStart`/`rowBytes`/`rowCount`). Only `U32`/`BF16`
    /// tensors are kept, matching this type's existing contract — the n-gram
    /// table's `.ngram_embedding.shard_*` weight/scales/biases tensors are
    /// the only ones it ever looks up.
    private static func readHeader(url: URL) throws -> [String: HeaderDescriptor] {
        let header = try Qwen4ExpSafetensorsHeader.read(url: url)
        var result = [String: HeaderDescriptor]()
        for (key, descriptor) in header.tensors {
            guard descriptor.shape.count >= 1 else { continue }
            switch descriptor.dtype {
            case .uint32, .bfloat16: break
            default: continue
            }
            let rowCount = descriptor.shape[0]
            let elementCount = descriptor.shape.dropFirst().reduce(1, *)
            result[key] = HeaderDescriptor(
                dataStart: header.dataSectionStart + descriptor.dataOffsets.start,
                rowBytes: elementCount * descriptor.dtype.byteWidth,
                rowCount: rowCount,
                dtype: descriptor.dtype.rawValue)
        }
        return result
    }
}
import MLXLMCommon

/// The deterministic hash used by Flash-Next's n-gram embedding table.
///
/// The table is too large to join into one MLX array (the released checkpoint
/// keeps it in 128 row shards), so the implementation preserves the shard
/// boundary in the module tree as well as in the lookup path.
public final class Qwen4ExpNGramShardTable: Module {
    @ModuleInfo(key: "shards") public var shards: [Embedding]
    private let lazyStorage: Qwen4ExpLazyNGramStorage?

    public init(
        shardSizes: [Int],
        dimensions: Int,
        quantization: Qwen4ExpQuantizationSpec? = nil,
        lazyStorage: Qwen4ExpLazyNGramStorage? = nil
    ) {
        self.lazyStorage = lazyStorage
        _shards.wrappedValue = lazyStorage == nil ? shardSizes.map {
            if let quantization {
                return Qwen4ExpPrequantizedEmbedding(
                    embeddingCount: $0, dimensions: dimensions,
                    quantization: quantization) as Embedding
            }
            return Embedding(embeddingCount: $0, dimensions: dimensions)
        } : []
        super.init()
    }

    public func lookup(shard: Int, rows: [Int32]) -> MLXArray {
        if let lazyStorage {
            return lazyStorage.lookup(shard: shard, rows: rows)
        }
        return shards[shard](MLXArray(rows).asType(.int32))
    }

    /// P6.2: batched lookup across every shard touched by a token batch, in
    /// request order. The lazy (real-checkpoint) path forwards to
    /// `Qwen4ExpLazyNGramStorage.lookupBatch` (one packed/scales/biases
    /// array + one dequantize for the whole batch). The module-backed path
    /// (small-vocab tests, no lazy storage) has no shard-dispatch cost worth
    /// batching — it stays one `Embedding` call per request, concatenated in
    /// order.
    public func lookupBatch(_ requests: [(shard: Int, row: Int32)]) -> MLXArray {
        if let lazyStorage {
            return lazyStorage.lookupBatch(requests)
        }
        let pieces = requests.map { request in
            shards[request.shard](MLXArray([request.row]).asType(.int32))
        }
        return concatenated(pieces, axis: 0)
    }

    public func cacheStats() -> Qwen4ExpNGramCacheStats? {
        lazyStorage?.cacheStats().publicStats
    }

    public func lookupStats() -> Qwen4ExpPLELookupStats? {
        lazyStorage?.lookupStats()
    }

    public func resetLookupStats() {
        lazyStorage?.resetLookupStats()
    }
}

public final class Qwen4ExpNGramEmbedding: Module {
    public let ngramSize: Int
    public let contextLength: Int
    public let headsPerNgram: Int
    public let ngramHeads: Int
    public let embeddingDimension: Int
    public let eosTokenID: Int32
    public let headVocabSizes: [Int]
    public let headOffsets: [Int]
    public let shardSizes: [Int]
    public let shardOffsets: [Int]

    @ParameterInfo(key: "layer_multipliers") public var layerMultipliers: MLXArray
    @ParameterInfo(key: "ngram_heads_vocab_sizes") public var ngramHeadsVocabSizes: MLXArray
    @ParameterInfo(key: "ngram_heads_offsets") public var ngramHeadsOffsets: MLXArray
    @ModuleInfo(key: "ngram_embedding") public var ngramEmbedding: Qwen4ExpNGramShardTable

    public init(
        configuration: Qwen4ExpTextConfiguration,
        embeddingDimension: Int,
        pleLayerIndex: Int,
        seed: UInt64 = 0,
        quantization: Qwen4ExpQuantizationSpec? = nil,
        lazyStorage: Qwen4ExpLazyNGramStorage? = nil
    ) {
        ngramSize = configuration.ngramSize
        contextLength = max(0, ngramSize - 1)
        headsPerNgram = configuration.headsPerNgram ?? 8
        ngramHeads = contextLength * headsPerNgram
        self.embeddingDimension = embeddingDimension
        eosTokenID = configuration.eosTokenID ?? 0

        precondition(ngramSize >= 2)
        precondition(embeddingDimension % ngramHeads == 0)

        var sizes: [Int] = []
        var offsets: [Int] = []
        var total = 0
        for head in 0 ..< ngramHeads {
            let globalHead = pleLayerIndex * ngramHeads + head
            let prime = Self.nthPrime(after: configuration.ngramVocabSizeBase - 1, count: globalHead + 1)
            sizes.append(prime)
            offsets.append(total)
            total += prime
        }
        let divisor = configuration.makeNgramVocabSizeDivisibleBy ?? 128
        let padded = ((total + divisor - 1) / divisor) * divisor
        headVocabSizes = sizes
        headOffsets = offsets
        let requestedShards = configuration.splitNgramParts
        precondition(requestedShards > 0 && requestedShards <= padded)
        let base = padded / requestedShards
        let remainder = padded % requestedShards
        let computedShardSizes = (0 ..< requestedShards).map { base + ($0 < remainder ? 1 : 0) }
        shardSizes = computedShardSizes
        var calculatedOffsets = [0]
        for size in computedShardSizes { calculatedOffsets.append(calculatedOffsets.last! + size) }
        shardOffsets = calculatedOffsets

        let multipliers = Self.buildLayerMultipliers(
            unigramVocabularySize: configuration.vocabSize,
            ngramSize: ngramSize,
            pleLayerIndex: pleLayerIndex,
            seed: seed)
        _layerMultipliers.wrappedValue = multipliers
        let shardDimension = embeddingDimension / ngramHeads
        _ngramHeadsVocabSizes.wrappedValue = MLXArray(headVocabSizes).asType(.int64)
        _ngramHeadsOffsets.wrappedValue = MLXArray(headOffsets).asType(.int64)
        _ngramEmbedding.wrappedValue = Qwen4ExpNGramShardTable(
            shardSizes: computedShardSizes, dimensions: shardDimension,
            quantization: quantization, lazyStorage: lazyStorage)
        super.init()
    }

    public func callAsFunction(
        _ inputIDs: MLXArray, cache: ArraysCache? = nil,
        verificationSink: Qwen4ExpVerificationSink? = nil
    ) -> MLXArray {
        precondition(inputIDs.ndim == 2)
        let input = inputIDs.asType(.int64)
        let batch = input.dim(0)
        let sequence = input.dim(1)
        let previous: MLXArray
        if let state = cache?[3] {
            previous = state
        } else {
            previous = MLXArray.full(
                [batch, contextLength], values: MLXArray(eosTokenID), dtype: .int64)
        }
        let history = concatenated([previous, input], axis: 1)
        if let cache {
            cache[3] = contiguous(history[.ellipsis, (-contextLength)...])
        }
        // PM4.2: `history` is a pure concatenation of the old ID window and
        // the new tokens — no recurrence beyond that — so any prefix window
        // can be read back after the walk decides how many new tokens were
        // committed. See `Qwen4ExpVerificationCapture`.
        verificationSink?.record(
            slot: 3, entry: .window(source: history, length: contextLength))

        var shifted: [MLXArray] = []
        for shift in 0 ..< ngramSize {
            shifted.append(shiftRightIgnoringEOS(history, shift: shift))
        }

        var blocks: [MLXArray] = []
        for ngram in 2 ... ngramSize {
            let start = (ngram - 2) * headsPerNgram
            let end = start + headsPerNgram
            var mixed = shifted[0] * layerMultipliers[0]
            for position in 1 ..< ngram {
                mixed = MLX.bitwiseXOr(mixed, shifted[position] * layerMultipliers[position])
            }
            let sizes = MLXArray(Array(headVocabSizes[start ..< end])).asType(.int64)
            let offsets = MLXArray(Array(headOffsets[start ..< end])).asType(.int64)
            let IDs = remainder(
                mixed.expandedDimensions(axis: -1), sizes.expandedDimensions(axis: 0))
                + offsets.expandedDimensions(axis: 0)
            blocks.append(IDs)
        }

        let allIDs = concatenated(blocks, axis: -1)
        let sequenceIndices = MLXArray.arange(
            allIDs.dim(1) - sequence, allIDs.dim(1), dtype: .int32)
        let sequenceGather = broadcast(
            sequenceIndices[.newAxis, 0..., .newAxis],
            to: [batch, sequence, ngramHeads])
        let IDs = takeAlong(allIDs, sequenceGather, axis: 1)
        precondition(IDs.shape == [batch, sequence, ngramHeads], "PLE IDs inattendus: \(IDs.shape)")
        let lookedUp = lookup(IDs)
        precondition(lookedUp.shape == [batch, sequence, embeddingDimension],
                     "PLE embeddings inattendus: shape=\(lookedUp.shape)")
        return lookedUp
    }

    private func lookup(_ IDs: MLXArray) -> MLXArray {
        let flat = IDs.reshaped([-1])
        eval(flat)
        let hostIDs = flat.asArray(Int64.self)
        guard !hostIDs.isEmpty else {
            return MLXArray.zeros([IDs.dim(0), IDs.dim(1), embeddingDimension])
        }

        // P6.2 mesure (docs/knowledge/log.md "P6.2") : chaque shard distinct
        // touché par le batch (jusqu'à 231 mesurés sur le checkpoint réel,
        // constant quel que soit le nombre de tokens dans le préfill)
        // construit ici ses propres `MLXArray(...)` × 2-3 + un
        // `MLX.dequantized` (`lookup(shard:rows:)` ci-dessus), puis un
        // scatter/add sur device par shard — O(shards) constructions de
        // tenseurs et dispatches GPU par appel PLE. Correctif (une seule
        // construction de tenseurs par appel) dans le commit suivant.
        var shardIndices: [Int] = []
        shardIndices.reserveCapacity(hostIDs.count)
        for ID in hostIDs {
            let value = Int(ID)
            precondition(value >= 0 && value < shardOffsets.last!)
            shardIndices.append(shardIndex(for: value))
        }
        var positionsByShard: [Int: [Int32]] = [:]
        positionsByShard.reserveCapacity(min(shardOffsets.count, shardIndices.count))
        for (index, shard) in shardIndices.enumerated() {
            positionsByShard[shard, default: []].append(Int32(index))
        }
        var groupedValues: [MLXArray] = []
        var groupedPositions: [[Int32]] = []
        for shard in positionsByShard.keys.sorted() {
            let positions = positionsByShard[shard]!
            let local = positions.map { Int32(hostIDs[Int($0)]) - Int32(shardOffsets[shard]) }
            let values = ngramEmbedding.lookup(shard: shard, rows: local)
            groupedValues.append(values)
            groupedPositions.append(positions)
        }
        let dimension = embeddingDimension / ngramHeads
        guard let firstValues = groupedValues.first else {
            return MLXArray.zeros([IDs.dim(0), IDs.dim(1), embeddingDimension])
        }
        var result = MLXArray.zeros([hostIDs.count, dimension], dtype: firstValues.dtype)
        for (values, positions) in zip(groupedValues, groupedPositions) {
            result = result.at[MLXArray(positions).asType(.int32)].add(values)
        }
        result = result.reshaped([IDs.dim(0), IDs.dim(1), embeddingDimension])
        precondition(result.shape == [IDs.dim(0), IDs.dim(1), embeddingDimension],
                     "PLE lookup inattendu: \(result.shape)")
        return result
    }

    public func cacheStats() -> Qwen4ExpNGramCacheStats? {
        ngramEmbedding.cacheStats()
    }

    public func ngramLookupStats() -> Qwen4ExpPLELookupStats? {
        ngramEmbedding.lookupStats()
    }

    public func resetNgramLookupStats() {
        ngramEmbedding.resetLookupStats()
    }

    private func shardIndex(for ID: Int) -> Int {
        var low = 0
        var high = shardSizes.count
        while low + 1 < high {
            let middle = (low + high) / 2
            if shardOffsets[middle] <= ID { low = middle } else { high = middle }
        }
        return low
    }

    private func shiftRightIgnoringEOS(_ tokens: MLXArray, shift: Int) -> MLXArray {
        guard shift > 0 else { return tokens }
        let positions = MLXArray.arange(tokens.dim(1), dtype: .int64)
        let eosPositions = MLX.where(tokens .== eosTokenID, positions, -1)
        let previousInclusive = MLX.cummax(eosPositions, axis: 1)
        let previous = concatenated([
            MLXArray.full([tokens.dim(0), 1], values: MLXArray(Int64(-1)), dtype: .int64),
            previousInclusive[.ellipsis, ..<(tokens.dim(1) - 1)]
        ], axis: 1)
        let segmentStart = previous + 1
        let positionInSegment = positions[.newAxis, 0...] - segmentStart
        let sourcePositions = positions - Int64(shift)
        let gather = broadcast(
            MLX.maximum(sourcePositions, 0)[.newAxis, 0...],
            to: tokens.shape)
        let shifted = takeAlong(tokens, gather, axis: 1)
        let valid = (positionInSegment .>= shift) .&& (sourcePositions[.newAxis, 0...] .>= 0)
        return MLX.where(valid, shifted, eosTokenID)
    }

    private static func splitMix64(_ value: UInt64) -> UInt64 {
        var result = value &+ 0x9E3779B97F4A7C15
        result = (result ^ (result >> 30)) &* 0xBF58476D1CE4E5B9
        result = (result ^ (result >> 27)) &* 0x94D049BB133111EB
        return result ^ (result >> 31)
    }

    private static func buildLayerMultipliers(
        unigramVocabularySize: Int, ngramSize: Int, pleLayerIndex: Int, seed: UInt64
    ) -> MLXArray {
        let maxLong = UInt64(Int64.max)
        let multiplierMax = maxLong / UInt64(max(unigramVocabularySize, 1))
        let halfBound = max(1, multiplierMax / 2)
        let baseSeed = seed &+ UInt64(10007 * pleLayerIndex)
        let values = (0 ..< ngramSize).map { index in
            let value = baseSeed &+ 0x9E3779B97F4A7C15 &* UInt64(index + 1)
            return Int64(2 * (Self.splitMix64(value) % halfBound) + 1)
        }
        return MLXArray(values).asType(.int64)
    }

    private static func nthPrime(after start: Int, count: Int) -> Int {
        var value = max(1, start)
        var found = 0
        while found < count {
            value += 1
            if Self.isPrime(value) { found += 1 }
        }
        return value
    }

    private static func isPrime(_ value: Int) -> Bool {
        guard value >= 2 else { return false }
        if value % 2 == 0 { return value == 2 }
        var divisor = 3
        while divisor * divisor <= value {
            if value % divisor == 0 { return false }
            divisor += 2
        }
        return true
    }
}

/// Per-layer n-gram injection (PLE) used at layer 2 of Flash-Next.
public final class Qwen4ExpPLELayer: Module {
    public let hiddenSize: Int
    public let streamCount: Int
    public let dilation: Int
    public let shortConvStateLength: Int

    @ModuleInfo(key: "ple_embedding") public var pleEmbedding: Qwen4ExpNGramEmbedding
    @ModuleInfo(key: "key_proj") public var keyProj: Linear
    @ModuleInfo(key: "value_proj") public var valueProj: Linear
    @ModuleInfo(key: "norm_key") public var normKey: Qwen4ExpRMSNorm
    @ModuleInfo(key: "norm_query") public var normQuery: Qwen4ExpRMSNorm
    @ModuleInfo(key: "norm_conv") public var normConv: Qwen4ExpRMSNorm
    @ModuleInfo(key: "conv1d") public var conv1d: Conv1d

    public init(
        configuration: Qwen4ExpTextConfiguration,
        layerIndex: Int,
        pleLayerIndex: Int,
        seed: UInt64 = 0,
        quantization: Qwen4ExpQuantizationSpec? = nil,
        lazyStorage: Qwen4ExpLazyNGramStorage? = nil
    ) {
        hiddenSize = configuration.hiddenSize
        streamCount = configuration.hcCount
        dilation = configuration.ngramSize
        shortConvStateLength = max(0, configuration.pleConvKernelSize - 1) * dilation
        let width = hiddenSize * streamCount
        let embeddingDimension = configuration.pleEmbedDim ?? hiddenSize
        _pleEmbedding.wrappedValue = Qwen4ExpNGramEmbedding(
            configuration: configuration, embeddingDimension: embeddingDimension,
            pleLayerIndex: pleLayerIndex, seed: seed, quantization: quantization,
            lazyStorage: lazyStorage)
        _keyProj.wrappedValue = qwen4ExpLinear(
            inputDimensions: embeddingDimension, outputDimensions: width,
            quantization: quantization)
        _valueProj.wrappedValue = qwen4ExpLinear(
            inputDimensions: embeddingDimension, outputDimensions: hiddenSize,
            quantization: quantization)
        _normKey.wrappedValue = Qwen4ExpRMSNorm(dimensions: width, groupSize: hiddenSize)
        _normQuery.wrappedValue = Qwen4ExpRMSNorm(dimensions: width, groupSize: hiddenSize)
        _normConv.wrappedValue = Qwen4ExpRMSNorm(dimensions: width, groupSize: hiddenSize)
        _conv1d.wrappedValue = Conv1d(
            inputChannels: width, outputChannels: width,
            kernelSize: configuration.pleConvKernelSize, stride: 1, padding: 0,
            dilation: dilation, groups: width, bias: false)
        super.init()
        _ = layerIndex
    }

    public func callAsFunction(
        hiddenStates: MLXArray,
        inputIDs: MLXArray,
        cache: ArraysCache? = nil,
        mask: MLXArray? = nil,
        verificationSink: Qwen4ExpVerificationSink? = nil
    ) -> MLXArray {
        let batch = hiddenStates.dim(0)
        let sequence = hiddenStates.dim(1)
        let embeddings = pleEmbedding(inputIDs, cache: cache, verificationSink: verificationSink)
        let keys = normKey(keyProj(embeddings)).reshaped(
            [batch, sequence, streamCount, hiddenSize])
        let values = valueProj(embeddings)
        let queries = normQuery(hiddenStates).reshaped(
            [batch, sequence, streamCount, hiddenSize])
        var gate = (keys * queries).sum(axis: -1, keepDims: true) / Float(hiddenSize).squareRoot()
        gate = sign(gate) * sqrt(maximum(abs(gate), 1e-6))
        var gated = sigmoid(gate) * values[.ellipsis, .newAxis, 0...]
        gated = gated.reshaped([batch, sequence, streamCount * hiddenSize])
        var normed = normConv(gated)
        if let mask {
            let expanded = mask[.ellipsis, .newAxis]
            gated = MLX.where(expanded, gated, 0)
            normed = MLX.where(expanded, normed, 0)
        }
        let state = cache?[2] ?? MLXArray.zeros(
            [batch, shortConvStateLength, streamCount * hiddenSize], dtype: normed.dtype)
        let convInput = concatenated([state, normed], axis: 1)
        if let cache {
            cache[2] = contiguous(
                convInput[0..., (-shortConvStateLength)..., 0...])
        }
        if shortConvStateLength > 0 {
            // PM4.2: same reasoning as the ID history above — `convInput` is
            // a plain concatenation, any prefix window is a valid rollback
            // target.
            verificationSink?.record(
                slot: 2, entry: .window(source: convInput, length: shortConvStateLength))
        }
        return gated + silu(conv1d(convInput))
    }

    public func ngramCacheStats() -> Qwen4ExpNGramCacheStats? {
        pleEmbedding.cacheStats()
    }

    public func ngramLookupStats() -> Qwen4ExpPLELookupStats? {
        pleEmbedding.ngramLookupStats()
    }

    public func resetNgramLookupStats() {
        pleEmbedding.resetNgramLookupStats()
    }
}

private extension Duration {
    var seconds: Double {
        let components = self.components
        return Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}
