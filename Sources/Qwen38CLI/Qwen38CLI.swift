import ArgumentParser
import Foundation
import MLX
import MLXLMCommon
import MLXProfiler
import Qwen38Core
import Qwen38Server
import Tokenizers

private func durationSeconds(_ duration: Duration) -> Double {
    let components = duration.components
    return Double(components.seconds) + Double(components.attoseconds) / 1e18
}

@main
struct Qwen38CLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "qwen38",
        abstract: "Inférence locale Qwen3.8 sur MLX",
        subcommands: [
            Info.self, FlashSliceProbe.self, FlashStreamProbe.self, FlashGlobalProbe.self,
            FlashTextProbe.self, FlashVisionProbe.self, FlashMergeProbe.self,
            FlashMultimodalProbe.self, FlashGenerateProbe.self, FlashLayerBench.self,
            FlashQSAParity.self,
            FlashMRoPEParity.self,
            FlashVisionParity.self,
            FlashLanguageParity.self,
            FlashGlobalParity.self,
            FlashSingleLayerParity.self,
            FlashPublicLayerParity.self,
            FlashSelectedLayersParity.self,
            FlashTemplateProbe.self,
            FlashNGramParity.self,
            FlashTeacherForcedScore.self,
            FlashMTPProbe.self,
            FlashChatProbe.self,
            Generate.self, MTPProbe.self, MTPParity.self,
            MTPConversationProbe.self,
            ConversationBenchmark.self,
            ConversationParity.self, Download.self, Serve.self
        ]
    )
}

struct FlashSliceProbe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flash-slice-probe",
        abstract: "Charger une couche réelle du checkpoint Flash-Next sans charger le modèle complet")

    @Argument(help: "Répertoire local du checkpoint qwen4_exp")
    var modelPath: String

    @Option(name: .long, help: "Index de couche à charger (la couche 0 est le chemin sûr par défaut)")
    var layer: Int = 0

    @Option(name: .long, help: "Nombre de tokens du forward de fumée optionnel")
    var sequenceLength: Int = 4

    @Flag(name: .long, help: "Exécuter aussi un forward GDN réel après le chargement")
    var runForward = false

    @Flag(name: .long, help: "Ne pas matérialiser les poids après update")
    var lazy = false

    func run() async throws {
        let directory = URL(fileURLWithPath: modelPath, isDirectory: true)
        print("Sélection de la couche \(layer)…")
        let loaded = try Qwen4ExpCheckpointSliceLoader.loadLayer(
            layer,
            from: directory,
            materialize: !lazy,
            runForward: runForward,
            sequenceLength: sequenceLength)
        let report = loaded.report
        print("checkpoint slice: couche \(report.layerIndex)")
        print("tenseurs chargés: \(report.tensorCount)")
        print("shards lus: \(report.shardCount)")
        if report.materializedBytes > 0 {
            print("poids matérialisés: \(ByteCountFormatter.string(fromByteCount: report.materializedBytes, countStyle: .file))")
        } else {
            print("poids matérialisés: non (mode lazy)")
        }
        if let outputShape = report.outputShape {
            print("forward: OK, sortie \(outputShape)")
        }
    }
}

struct FlashStreamProbe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flash-stream-probe",
        abstract: "Exécuter plusieurs couches Flash-Next en ne gardant qu’une couche chargée")

    @Argument(help: "Répertoire local du checkpoint qwen4_exp")
    var modelPath: String

    @Option(name: .long, help: "Indices séparés par des virgules, par exemple 0,1,3")
    var layers: String = "0,1,3"

    @Option(name: .long, help: "Nombre de tokens du forward de fumée")
    var sequenceLength: Int = 4

    @Option(name: .long, help: "Nombre d'appels successifs avec les caches conservés")
    var repeatCount: Int = 1

    @Flag(name: .long, help: "Garder les poids lazy pendant les forwards")
    var lazy = false

    func run() async throws {
        let indices = try layers.split(separator: ",").map { part -> Int in
            guard let value = Int(part.trimmingCharacters(in: .whitespaces)) else {
                throw ValidationError("Indice de couche invalide : \(part)")
            }
            return value
        }
        guard !indices.isEmpty else {
            throw ValidationError("Au moins une couche est nécessaire.")
        }
        guard repeatCount > 0 else {
            throw ValidationError("--repeat-count doit être positif")
        }
        let directory = URL(fileURLWithPath: modelPath, isDirectory: true)
        let decoder = try Qwen4ExpStreamingDecoder(directory: directory)
        let ids = MLXArray(Array(repeating: Int32(1), count: sequenceLength))
            .reshaped([1, sequenceLength])
        for pass in 0..<repeatCount {
            let hidden = MLXArray.zeros(
                [1, sequenceLength, decoder.configuration.hiddenSize * decoder.configuration.hcCount],
                dtype: .bfloat16)
            let result = try decoder.forward(
                hidden,
                inputIDs: ids,
                layerIndices: indices,
                materializeLayers: !lazy)
            for report in result.reports {
                let memory = report.materializedBytes > 0
                    ? ByteCountFormatter.string(
                        fromByteCount: report.materializedBytes, countStyle: .file)
                    : "lazy"
                print(
                    "tour \(pass + 1) · couche \(report.layerIndex): "
                        + "\(report.loadedTensorCount) tenseurs, "
                        + "\(report.loadedShardCount) shards, poids=\(memory), "
                        + String(format: "load=%.2fs forward=%.2fs", report.loadDuration, report.forwardDuration))
            }
            print("tour \(pass + 1): sortie \(result.output.shape)")
        }
        print("stream forward: OK · \(repeatCount) appels · caches conservés")
        print("MLX mémoire active: \(ByteCountFormatter.string(fromByteCount: Int64(Memory.activeMemory), countStyle: .file))")
        print("MLX mémoire peak: \(ByteCountFormatter.string(fromByteCount: Int64(Memory.peakMemory), countStyle: .file))")
    }
}

struct FlashGlobalProbe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flash-global-probe",
        abstract: "Charger les poids globaux Flash-Next et vérifier embedding/lm-head")

    @Argument(help: "Répertoire local du checkpoint qwen4_exp")
    var modelPath: String

    @Option(name: .long, help: "Nombre de tokens à projeter")
    var sequenceLength: Int = 4

    @Flag(name: .long, help: "Ne pas matérialiser les poids après update")
    var lazy = false

    func run() async throws {
        guard sequenceLength > 0 else {
            throw ValidationError("--sequence-length doit être positif")
        }
        let directory = URL(fileURLWithPath: modelPath, isDirectory: true)
        let start = ContinuousClock.now
        let loaded = try Qwen4ExpGlobalCheckpointLoader.load(
            from: directory, materialize: !lazy)
        let loadSeconds = ContinuousClock.now - start
        let ids = MLXArray(Array(repeating: Int32(1), count: sequenceLength))
            .reshaped([1, sequenceLength])
        let embeddings = loaded.model.embed(ids)
        let hyper = tiled(
            embeddings,
            repetitions: [1, 1, loaded.model.configuration.hcCount])
        let reduced = loaded.model.reduceHyperStreams(hyper)
        let logits = loaded.model.logits(from: reduced)
        eval(logits)
        print("poids globaux: \(loaded.report.tensorCount) tenseurs, \(loaded.report.shardCount) shards")
        let materialized = loaded.report.materializedBytes > 0
            ? ByteCountFormatter.string(fromByteCount: loaded.report.materializedBytes, countStyle: .file)
            : "non (mode lazy)"
        print("poids matérialisés: \(materialized)")
        print("chargement: \(String(format: "%.2fs", durationSeconds(loadSeconds)))")
        print("embedding: \(embeddings.shape) · réduction: \(reduced.shape) · logits: \(logits.shape)")
        print("MLX mémoire active: \(ByteCountFormatter.string(fromByteCount: Int64(Memory.activeMemory), countStyle: .file))")
        print("MLX mémoire peak: \(ByteCountFormatter.string(fromByteCount: Int64(Memory.peakMemory), countStyle: .file))")
    }
}

struct FlashTextProbe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flash-text-probe",
        abstract: "Produire des logits Flash-Next avec un sous-ensemble de couches streamées")

    @Argument(help: "Répertoire local du checkpoint qwen4_exp")
    var modelPath: String

    @Option(name: .long, help: "Indices séparés par des virgules; par défaut 0,1,3")
    var layers: String = "0,1,3"

    @Option(name: .long, help: "Nombre de tokens du forward")
    var sequenceLength: Int = 4

    @Option(name: .long, help: "Répéter le forward pour vérifier la réutilisation des caches")
    var repeatCount: Int = 1

    @Flag(name: .long, help: "Garder les poids de couche lazy")
    var lazy = false

    func run() async throws {
        guard sequenceLength > 0 else {
            throw ValidationError("--sequence-length doit être positif")
        }
        guard repeatCount > 0 else {
            throw ValidationError("--repeat-count doit être positif")
        }
        let indices = try layers.split(separator: ",").map { part -> Int in
            guard let value = Int(part.trimmingCharacters(in: .whitespaces)) else {
                throw ValidationError("Indice de couche invalide : \(part)")
            }
            return value
        }
        guard !indices.isEmpty else {
            throw ValidationError("Au moins une couche est nécessaire.")
        }
        let directory = URL(fileURLWithPath: modelPath, isDirectory: true)
        let start = ContinuousClock.now
        let model = try Qwen4ExpStreamingTextModel(directory: directory)
        let ids = MLXArray(Array(repeating: Int32(1), count: sequenceLength))
            .reshaped([1, sequenceLength])
        var lastLogits: [Int] = []
        for pass in 0..<repeatCount {
            let result = try model.forward(
                inputIDs: ids,
                layerIndices: indices,
                materializeLayers: !lazy)
            for report in result.reports {
                let memory = report.materializedBytes > 0
                    ? ByteCountFormatter.string(
                        fromByteCount: report.materializedBytes, countStyle: .file)
                    : "lazy"
                print(
                    "tour \(pass + 1) · couche \(report.layerIndex): poids=\(memory), "
                        + String(format: "load=%.2fs forward=%.2fs", report.loadDuration, report.forwardDuration))
            }
            lastLogits = result.logits.shape
        }
        print("global: \(model.globalLoadReport.tensorCount) tenseurs")
        print("text forward: OK, logits \(lastLogits) · tours \(repeatCount)")
        print("durée totale: \(String(format: "%.2fs", durationSeconds(ContinuousClock.now - start)))")
        print("MLX mémoire active: \(ByteCountFormatter.string(fromByteCount: Int64(Memory.activeMemory), countStyle: .file))")
        print("MLX mémoire peak: \(ByteCountFormatter.string(fromByteCount: Int64(Memory.peakMemory), countStyle: .file))")
    }
}

struct FlashVisionProbe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flash-vision-probe",
        abstract: "Charger la tour vision Flash-Next et produire ses embeddings")

    @Argument(help: "Répertoire local du checkpoint qwen4_exp")
    var modelPath: String

    @Option(name: .long, help: "Hauteur de l'image synthétique, multiple de 32")
    var height: Int = 224

    @Option(name: .long, help: "Largeur de l'image synthétique, multiple de 32")
    var width: Int = 224

    @Option(name: .long, help: "Image réelle à prétraiter; sinon une image noire est utilisée")
    var image: String?

    @Flag(name: .long, help: "Ne pas matérialiser les poids après update")
    var lazy = false

    func run() async throws {
        guard height > 0, width > 0, height % 32 == 0, width % 32 == 0 else {
            throw ValidationError("--height et --width doivent être positifs et multiples de 32")
        }
        let directory = URL(fileURLWithPath: modelPath, isDirectory: true)
        let start = ContinuousClock.now
        let loaded = try Qwen4ExpVisionCheckpointLoader.load(
            from: directory, materialize: !lazy)
        let pixels: MLXArray
        let inputDescription: String
        if let image {
            let processed = try Qwen4ExpImageProcessor.load(
                from: URL(fileURLWithPath: image),
                patchSize: loaded.model.configuration.patchSize,
                mergeSize: loaded.model.configuration.spatialMergeSize)
            pixels = processed.pixels
            inputDescription = "image \(processed.width)x\(processed.height)"
        } else {
            pixels = MLXArray.zeros([1, height, width, 3], dtype: .bfloat16)
            inputDescription = "zeros \(width)x\(height)"
        }
        let output = loaded.model(pixels)
        eval(output)
        print("vision: \(loaded.report.tensorCount) tenseurs, \(loaded.report.shardCount) shard(s)")
        let memory = loaded.report.materializedBytes > 0
            ? ByteCountFormatter.string(
                fromByteCount: loaded.report.materializedBytes, countStyle: .file)
            : "non (mode lazy)"
        print("poids matérialisés: \(memory)")
        print("entrée: \(inputDescription) · embeddings vision: \(output.shape)")
        print("durée: \(String(format: "%.2fs", durationSeconds(ContinuousClock.now - start)))")
        print("MLX mémoire active: \(ByteCountFormatter.string(fromByteCount: Int64(Memory.activeMemory), countStyle: .file))")
        print("MLX mémoire peak: \(ByteCountFormatter.string(fromByteCount: Int64(Memory.peakMemory), countStyle: .file))")
    }
}

struct FlashMergeProbe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flash-merge-probe",
        abstract: "Valider le remplacement des marqueurs image par les embeddings vision")

    @Argument(help: "Répertoire local du checkpoint qwen4_exp")
    var modelPath: String

    @Option(name: .long, help: "Image locale à encoder")
    var image: String

    func run() async throws {
        let directory = URL(fileURLWithPath: modelPath, isDirectory: true)
        let configuration = try Qwen4ExpConfiguration.load(from: directory)
        guard let imageTokenID = configuration.imageTokenID else {
            throw Qwen4ExpInputMergeError.missingImageTokenID
        }
        let vision = try Qwen4ExpVisionCheckpointLoader.load(from: directory)
        let processed = try Qwen4ExpImageProcessor.load(
            from: URL(fileURLWithPath: image),
            patchSize: configuration.visionConfiguration.patchSize,
            mergeSize: configuration.visionConfiguration.spatialMergeSize)
        let imageEmbeddings = vision.model(processed.pixels)
        eval(imageEmbeddings)
        let global = try Qwen4ExpGlobalCheckpointLoader.load(from: directory)
        let tokenCount = imageEmbeddings.dim(1)
        let ids = MLXArray(
            Array(repeating: imageTokenID, count: tokenCount)).reshaped([1, tokenCount])
        let textEmbeddings = global.model.embed(ids)
        let merged = try Qwen4ExpInputMerger.merge(
            inputIDs: ids,
            textEmbeddings: textEmbeddings,
            visionEmbeddings: imageEmbeddings,
            imageTokenID: imageTokenID)
        eval(merged)
        print("image: \(processed.width)x\(processed.height) · marqueurs: \(tokenCount)")
        print("vision: \(imageEmbeddings.shape) · merge: \(merged.shape)")
        print("merge image: OK · image_token_id \(imageTokenID)")
        print("MLX mémoire active: \(ByteCountFormatter.string(fromByteCount: Int64(Memory.activeMemory), countStyle: .file))")
        print("MLX mémoire peak: \(ByteCountFormatter.string(fromByteCount: Int64(Memory.peakMemory), countStyle: .file))")
    }
}

struct FlashMultimodalProbe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flash-multimodal-probe",
        abstract: "Exécuter un premier forward texte avec des embeddings image fusionnés")

    @Argument(help: "Répertoire local du checkpoint qwen4_exp")
    var modelPath: String

    @Option(name: .long, help: "Image locale à encoder")
    var image: String

    @Option(name: .long, help: "Indices séparés par des virgules; par défaut 0,1,3")
    var layers: String = "0,1,3"

    @Flag(name: .long, help: "Garder les poids de couche lazy")
    var lazy = false

    @Option(name: .long, help: "Écrire une trace Chrome/Perfetto à ce chemin")
    var trace: String?

    func run() async throws {
        let indices = try layers.split(separator: ",").map { part -> Int in
            guard let value = Int(part.trimmingCharacters(in: .whitespaces)) else {
                throw ValidationError("Indice de couche invalide : \(part)")
            }
            return value
        }
        guard !indices.isEmpty else {
            throw ValidationError("Au moins une couche est nécessaire.")
        }

        let directory = URL(fileURLWithPath: modelPath, isDirectory: true)
        let configuration = try Qwen4ExpConfiguration.load(from: directory)
        guard let imageTokenID = configuration.imageTokenID else {
            throw Qwen4ExpInputMergeError.missingImageTokenID
        }
        let started = ContinuousClock.now
        let profiler = MLXProfiler.shared
        let profileSession: ProfilingSession?
        if trace != nil {
            let session = ProfilingSession(config: .singleRun, subsystem: "com.qwen38mlx")
            session.title = "QWEN3.8 FLASH-NEXT MULTIMODAL PROBE"
            session.metadata["model"] = directory.lastPathComponent
            session.metadata["image"] = image
            session.metadata["layers"] = indices.map(String.init).joined(separator: ",")
            profiler.activeSession = session
            profiler.enable()
            profileSession = session
        } else {
            profileSession = nil
        }
        defer {
            if let profileSession {
                profiler.disable()
                if let trace {
                    try? ChromeTraceExporter.export(session: profileSession).write(
                        to: URL(fileURLWithPath: trace))
                }
            }
        }
        profiler.start("Flash vision")
        let vision = try Qwen4ExpVisionCheckpointLoader.load(from: directory)
        let processed = try Qwen4ExpImageProcessor.load(
            from: URL(fileURLWithPath: image),
            patchSize: configuration.visionConfiguration.patchSize,
            mergeSize: configuration.visionConfiguration.spatialMergeSize)
        let imageEmbeddings = vision.model(processed.pixels)
        eval(imageEmbeddings)
        profiler.end("Flash vision")

        profiler.start("Flash globals")
        let textModel = try Qwen4ExpStreamingTextModel(directory: directory)
        profiler.end("Flash globals")
        // This is a deliberately minimal multimodal sequence.  The image
        // markers occupy the exact rows produced by the vision merger; the
        // trailing token proves that text after the image remains in place.
        let markerCount = imageEmbeddings.dim(1)
        guard let visionStartTokenID = configuration.visionStartTokenID,
              let visionEndTokenID = configuration.visionEndTokenID else {
            throw Qwen4ExpMRoPEPositionError.missingVisionStart
        }
        let ids = MLXArray(
            [visionStartTokenID]
                + Array(repeating: imageTokenID, count: markerCount)
                + [visionEndTokenID, Int32(1)])
            .reshaped([1, markerCount + 3])
        let positionIDs = try Qwen4ExpMRoPE.multimodalPositionIDs(
            inputIDs: ids.asArray(Int32.self),
            imageTokenID: imageTokenID,
            visionStartTokenID: visionStartTokenID,
            grids: [Qwen4ExpVisionGrid(
                height: processed.patchGrid.height,
                width: processed.patchGrid.width)])
        profiler.start("Flash forward multimodal")
        let result = try textModel.forward(
            inputIDs: ids,
            layerIndices: indices,
            positionIDs: positionIDs,
            visionEmbeddings: imageEmbeddings,
            imageTokenID: imageTokenID,
            materializeLayers: !lazy)
        eval(result.logits)
        profiler.end("Flash forward multimodal")

        print("image: \(processed.width)x\(processed.height) · marqueurs: \(markerCount)")
        print("séquence multimodale: \(ids.shape) · logits: \(result.logits.shape)")
        print("forward multimodal: OK · couches \(indices)")
        print("durée: \(String(format: "%.2fs", durationSeconds(ContinuousClock.now - started)))")
        if let profileSession {
            print(profileSession.generateReport())
            if let trace {
                print("trace profiler: \(trace)")
           }
       }
      print("MLX mémoire active: \(ByteCountFormatter.string(fromByteCount: Int64(Memory.activeMemory), countStyle: .file))")
      print("MLX mémoire peak: \(ByteCountFormatter.string(fromByteCount: Int64(Memory.peakMemory), countStyle: .file))")
    }
}

struct FlashGenerateProbe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flash-generate-probe",
        abstract: "Générer des tokens greedy avec le modèle Flash-Next réel")

    @Argument(help: "Répertoire local du checkpoint qwen4_exp")
    var modelPath: String

    @Option(name: .long, help: "Prompt utilisateur")
    var prompt: String

    @Option(name: .long, help: "Image locale optionnelle à joindre au premier tour")
    var image: String?

    @Option(name: .long, help: "Nombre maximum de nouveaux tokens (1 par défaut pour le probe)")
    var maxNewTokens: Int = 1

    @Option(name: .long, help: "Afficher les top-k logits du premier token (0 désactive)")
    var reportTopK: Int = 0

    @Flag(name: .long, help: "Insérer le préfixe thinking dans le prompt")
    var thinking = false

    @Flag(
        name: .long,
        help: "Conserver toutes les couches Flash-Next en mémoire entre les tokens (expérimental)")
    var residentLayers = false

    @Option(
        name: .long,
        help: "Évaluer le graphe résident toutes les N couches (défaut: 1 — mesuré plus rapide et stable que le batching, voir docs/knowledge/log.md 2026-09-05)")
    var residentEvalInterval = 1

    @Flag(
        name: .long,
        help: "Activer le générateur MTP Flash-Next local (texte uniquement, opt-in)")
    var mtp = false

    @Option(name: .long, help: "Taille du bloc MTP Flash-Next")
    var mtpBlockSize = 2

    @Option(name: .long, help: "Écrire une trace Chrome/Perfetto à ce chemin")
    var trace: String?

    @Flag(
        name: .long,
        help:
            "Profiler chaque couche (\"Flash couche N\") au lieu des seules phases Prefill/Generation — coûte ~4,7 ms par frontière de phase (P0-c)"
    )
    var profileLayers = false

    @Flag(
        name: .long,
        help:
            "Mode résident : `asyncEval` sur chaque couche intermédiaire, `eval` bloquant sur la dernière seulement (P2-code (d), à trancher par P1)"
    )
    var residentAsync = false

    @Flag(
        name: .long,
        help:
            "Revenir à `loadArraysAndMetadata` (cache de fichiers du noyau) au lieu de la lecture F_NOCACHE des tenseurs résidents (P2-mem-a, défaut : F_NOCACHE actif)"
    )
    var cachedIO = false

    func run() async throws {
        guard maxNewTokens > 0 else {
            throw ValidationError("--max-new-tokens doit être positif")
        }
        guard reportTopK >= 0 else {
            throw ValidationError("--report-top-k doit être positif ou nul")
        }
        guard mtpBlockSize >= 2 else {
            throw ValidationError("--mtp-block-size doit être au moins 2")
        }
        guard residentEvalInterval > 0 else {
            throw ValidationError("--resident-eval-interval doit être positif")
        }
        if mtp && image != nil {
            throw ValidationError(
                "--mtp est temporairement limité au texte : les deltas M-RoPE multimodaux restent à brancher")
        }
        let directory = URL(fileURLWithPath: modelPath, isDirectory: true)
        let configuration = try Qwen4ExpConfiguration.load(from: directory)
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        // The profiler samples MLX memory at the beginning of a phase.  On a
        // fresh CLI process that sample can happen before MLX has created its
        // default Metal device; mlx-swift then sees an empty device list and
        // the native allocator aborts.  Touch the device explicitly before
        // enabling profiler phases (the same process is still GPU-backed).
        _ = Device.defaultDevice()
        let profiler = MLXProfiler.shared
        let profileSession: ProfilingSession?
        if trace != nil {
            let session = ProfilingSession(config: .singleRun, subsystem: "com.qwen38mlx")
            session.title = "QWEN3.8 FLASH-NEXT GREEDY PROBE"
            session.metadata["model"] = directory.lastPathComponent
            session.metadata["image"] = image ?? "none"
            session.metadata["thinking"] = thinking ? "true" : "false"
            session.metadata["max_new_tokens"] = String(maxNewTokens)
            profiler.activeSession = session
            profiler.enable()
            profileSession = session
        } else {
            profileSession = nil
        }
        defer {
            if let profileSession {
                profiler.disable()
                if let trace {
                    try? ChromeTraceExporter.export(session: profileSession).write(
                        to: URL(fileURLWithPath: trace))
                }
            }
        }

        let promptIDs: [Int32]
        let positionIDs: MLXArray?
        let visionEmbeddings: MLXArray?
        let imageTokenID: Int32?
        if let image {
            guard let imageToken = configuration.imageTokenID,
                  let visionStart = configuration.visionStartTokenID,
                  let visionEnd = configuration.visionEndTokenID else {
                throw Qwen4ExpInputMergeError.missingImageTokenID
            }
            profiler.start("Flash vision")
            let vision = try Qwen4ExpVisionCheckpointLoader.load(
                from: directory, uncachedIO: !cachedIO)
            let processed = try Qwen4ExpImageProcessor.load(
                from: URL(fileURLWithPath: image),
                patchSize: configuration.visionConfiguration.patchSize,
                mergeSize: configuration.visionConfiguration.spatialMergeSize)
            let embeddings = vision.model(processed.pixels)
            eval(embeddings)
            profiler.end("Flash vision")

            func encode(_ value: String) -> [Int32] {
                tokenizer.encode(text: value, addSpecialTokens: false).map(Int32.init)
            }
            let markerCount = embeddings.dim(1)
            promptIDs = encode("<|im_start|>user\n")
                + [visionStart]
                + Array(repeating: imageToken, count: markerCount)
                + [visionEnd]
                + encode(prompt + "\n<|im_end|>\n<|im_start|>assistant\n")
                + (thinking ? encode("<think>\n") : encode("<think>\n\n</think>\n\n"))
            positionIDs = try Qwen4ExpMRoPE.multimodalPositionIDs(
                inputIDs: promptIDs,
                imageTokenID: imageToken,
                visionStartTokenID: visionStart,
                grids: [Qwen4ExpVisionGrid(
                    height: processed.patchGrid.height,
                    width: processed.patchGrid.width)])
            visionEmbeddings = embeddings
            imageTokenID = imageToken
            print("image: \(processed.width)x\(processed.height) · marqueurs: \(markerCount)")
        } else {
            let messages: [[String: any Sendable]] = [[
                "role": "user",
                "content": prompt
            ]]
            promptIDs = try tokenizer.applyChatTemplate(
                messages: messages,
                tools: nil,
                additionalContext: [
                    "enable_thinking": thinking,
                    "reasoning_effort": "low"
                ]).map(Int32.init)
            positionIDs = nil
            visionEmbeddings = nil
            imageTokenID = nil
        }

        profiler.start("Flash globals")
        let model = try Qwen4ExpStreamingTextModel(
            directory: directory,
            layerLoadingMode: residentLayers ? .resident : .streamed,
            residentEvaluationInterval: residentEvalInterval,
            profileLayers: profileLayers,
            residentAsyncEval: residentAsync,
            uncachedIO: !cachedIO)
        profiler.end("Flash globals")
        let generator = Qwen4ExpGreedyGenerator(model: model)
        let stopTokens: Set<Int32> = [
            configuration.textConfiguration.eosTokenID,
            Int32(248044), Int32(248046)
        ].compactMap { $0 }.reduce(into: Set<Int32>()) { $0.insert($1) }
        if mtp {
            let loadedMTP = try Qwen4ExpMTPLoader.load(from: directory, uncachedIO: !cachedIO)
            let result = try generator.generateMTP(
                promptTokenIDs: promptIDs,
                predictor: loadedMTP.model,
                options: .init(maxNewTokens: maxNewTokens, stopTokenIDs: stopTokens),
                blockSize: mtpBlockSize,
                profiler: profiler)
            let decoded = tokenizer.decode(
                tokens: result.tokenIDs.map(Int.init), skipSpecialTokens: false)
            print("mode : MTP Flash-Next local · bloc \(mtpBlockSize)")
            print("prompt tokens: \(result.promptTokenCount)")
            print("generated ids: \(result.tokenIDs)")
            print("decoded: \(decoded.debugDescription)")
            print("TTFT: \(String(format: "%.3fs", result.timeToFirstToken ?? 0))")
            print("prefill: \(String(format: "%.3fs", result.prefillTime))")
            print("decode: \(String(format: "%.3fs", result.generationTime))")
            print("couches: \(result.layerVisitCount) visites · load cumulé \(String(format: "%.3fs", result.layerLoadTime)) · forward cumulé \(String(format: "%.3fs", result.layerForwardTime))")
            print("MTP : \(result.stats.acceptedTokens)/\(result.stats.proposedTokens) acceptés · \(result.stats.rounds) rounds · \(result.stats.rollbacks) rollbacks · \(result.stats.replayedTokens) tokens rejoués · \(result.stats.targetVerifiedTokens) tokens vérifiés")
            if let rate = result.stats.acceptanceRate {
                print("accept rate MTP : \(String(format: "%.1f%%", rate * 100))")
            }
            if let profileSession {
                profileSession.metadata["mtp"] = "true"
                profileSession.metadata["prompt_tokens"] = String(result.promptTokenCount)
                profileSession.metadata["generated_ids"] = result.tokenIDs.map(String.init).joined(separator: ",")
                profileSession.metadata["decoded"] = decoded
                profileSession.metadata["mtp_block_size"] = String(mtpBlockSize)
                profileSession.metadata["mtp_rounds"] = String(result.stats.rounds)
                profileSession.metadata["mtp_proposed"] = String(result.stats.proposedTokens)
                profileSession.metadata["mtp_accepted"] = String(result.stats.acceptedTokens)
                profileSession.metadata["layer_visits"] = String(result.layerVisitCount)
                profileSession.metadata["layer_load_seconds"] = String(format: "%.6f", result.layerLoadTime)
                profileSession.metadata["layer_forward_seconds"] = String(format: "%.6f", result.layerForwardTime)
            }
        } else {
            let result = try generator.generate(
                promptTokenIDs: promptIDs,
                positionIDs: positionIDs,
                visionEmbeddings: visionEmbeddings,
                imageTokenID: imageTokenID,
                options: .init(maxNewTokens: maxNewTokens, stopTokenIDs: stopTokens),
                firstTokenTopK: reportTopK,
                profiler: profiler)
            let decoded = tokenizer.decode(
                tokens: result.tokenIDs.map(Int.init), skipSpecialTokens: false)
            print("prompt tokens: \(result.promptTokenCount)")
            print("generated ids: \(result.tokenIDs)")
            print("decoded: \(decoded.debugDescription)")
            print("TTFT: \(result.timeToFirstToken.map { String(format: "%.3fs", $0) } ?? "-")")
            print("prefill: \(String(format: "%.3fs", result.prefillTime))")
            print("decode: \(String(format: "%.3fs", result.generationTime))")
            print("couches: \(result.layerVisitCount) visites · load cumulé \(String(format: "%.3fs", result.layerLoadTime)) · forward cumulé \(String(format: "%.3fs", result.layerForwardTime))")
            if !result.firstTokenCandidates.isEmpty {
                print("top-\(result.firstTokenCandidates.count) premier token:")
                for (rank, candidate) in result.firstTokenCandidates.enumerated() {
                    print("  #\(rank + 1) id=\(candidate.tokenID) logit=\(String(format: "%.6f", candidate.logit))")
                }
            }
            if let profileSession {
                profileSession.metadata["mtp"] = "false"
                profileSession.metadata["prompt_tokens"] = String(result.promptTokenCount)
                profileSession.metadata["generated_ids"] = result.tokenIDs.map(String.init).joined(separator: ",")
                profileSession.metadata["decoded"] = decoded
                profileSession.metadata["layer_visits"] = String(result.layerVisitCount)
                profileSession.metadata["layer_load_seconds"] = String(format: "%.6f", result.layerLoadTime)
                profileSession.metadata["layer_forward_seconds"] = String(format: "%.6f", result.layerForwardTime)
            }
        }
        print("MLX mémoire active: \(ByteCountFormatter.string(fromByteCount: Int64(Memory.activeMemory), countStyle: .file))")
        let ngramCacheStats = model.ngramCacheStats()
        if let profileSession {
            profileSession.metadata["ngram_cache_hits"] = String(ngramCacheStats.hits)
            profileSession.metadata["ngram_cache_misses"] = String(ngramCacheStats.misses)
            profileSession.metadata["ngram_cache_entries"] = String(ngramCacheStats.entries)
            profileSession.metadata["ngram_cache_hit_rate"] = ngramCacheStats.hitRate.map {
                String(format: "%.4f", $0)
            } ?? "n/a"
        }
        print("MLX mémoire peak: \(ByteCountFormatter.string(fromByteCount: Int64(Memory.peakMemory), countStyle: .file))")
        if let profileSession {
            print(profileSession.generateReport())
            if let trace { print("trace profiler: \(trace)") }
        }
    }
}

/// Which Flash-Next layer kind(s) `flash-layer-bench` should run.
enum FlashLayerBenchKindOption: String, ExpressibleByArgument, CaseIterable {
    case gdn
    case qsa
    case both
}

struct FlashLayerBench: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flash-layer-bench",
        abstract:
            "P0 : micro-bench synthétique d'une couche Flash-Next (poids empaquetés aléatoires, sans checkpoint)")

    @Option(name: .long, help: "Nombre de pas mesurés par type de couche")
    var steps: Int = 200

    @Option(name: .long, help: "Nombre de pas de warm-up non mesurés")
    var warmup: Int = 20

    @Option(name: .long, help: "Type de couche à bencher : gdn, qsa ou both")
    var layerKind: FlashLayerBenchKindOption = .both

    @Option(name: .long, help: "Écrire une trace Chrome/Perfetto à ce chemin")
    var trace: String?

    @Flag(
        name: .long,
        help:
            "P2-code (c) : envelopper le forward de couche dans MLX.compile (état de cache comme entrée/sortie)"
    )
    var compiled = false

    @Flag(
        name: .long,
        help:
            "Avec --compiled : ne pas recompiler seulement à cause d'un changement de forme (compile(shapeless: true))"
    )
    var shapeless = false

    @Option(
        name: .long,
        help:
            "P2-code (d) : asyncEval() pendant N-1 pas puis un eval() bloquant (défaut 1 = eval à chaque pas, comme la production)"
    )
    var asyncInterval = 1

    @Flag(
        name: .long,
        help:
            "P2-code (e) : ne pas construire le masque causal QSA (provablement toujours vrai en décodage à un jeton)"
    )
    var skipTrivialMask = false

    @Option(
        name: .long,
        help:
            "Q3.2 : bits des experts routés (switch_mlp) pour ce bench ; défaut = même valeur que la quantification globale (4)"
    )
    var expertBits: Int?

    @Option(
        name: .long,
        help:
            "Q3.2 : group_size des experts routés (switch_mlp) pour ce bench ; défaut = même valeur que la quantification globale (32)"
    )
    var expertGroupSize: Int?

    @Option(
        name: .long,
        help:
            "P2-fusion : niveau cumulatif F1-F6 appliqué à la couche après chargement (0 = chemin d'origine, défaut)"
    )
    var fusionLevel: Int = 0

    @Flag(
        name: .long,
        help:
            "P2-fusion : au lieu de mesurer, compare la sortie du chemin fusionné (--fusion-level) à celle du chemin d'origine sur des poids aléatoires seedés (32 pas), plus l'argmax d'un lm_head synthétique"
    )
    var checkParity = false

    func run() async throws {
        guard steps > 0 else {
            throw ValidationError("--steps doit être positif")
        }
        guard warmup >= 0 else {
            throw ValidationError("--warmup doit être positif ou nul")
        }
        guard asyncInterval > 0 else {
            throw ValidationError("--async-interval doit être positif")
        }
        if let expertBits {
            guard [2, 3, 4, 5, 6, 8].contains(expertBits) else {
                throw ValidationError("--expert-bits doit appartenir à {2,3,4,5,6,8}")
            }
        }
        if let expertGroupSize {
            guard expertGroupSize > 0 else {
                throw ValidationError("--expert-group-size doit être positif")
            }
        }
        guard let resolvedFusionLevel = Qwen4ExpFusionLevel(rawValue: fusionLevel) else {
            throw ValidationError("--fusion-level doit appartenir à 0-6 (P2-fusion F1-F6)")
        }
        if checkParity && resolvedFusionLevel == .none {
            throw ValidationError("--check-parity exige --fusion-level 1-6 (comparaison à .none)")
        }
        let expertsQuantization: Qwen4ExpQuantizationSpec? =
            (expertBits != nil || expertGroupSize != nil)
            ? Qwen4ExpQuantizationSpec(
                groupSize: expertGroupSize ?? 32, bits: expertBits ?? 4)
            : nil

        _ = Device.defaultDevice()

        let kinds: [Qwen4ExpLayerBenchKind]
        switch layerKind {
        case .both: kinds = [.gdn, .qsa]
        case .gdn: kinds = [.gdn]
        case .qsa: kinds = [.qsa]
        }

        if checkParity {
            for kind in kinds {
                let result = Qwen4ExpLayerBench.checkParity(
                    kind: kind, expertsQuantization: expertsQuantization,
                    fusionLevel: resolvedFusionLevel)
                print(
                    "parité \(kind.rawValue) niveau \(resolvedFusionLevel.rawValue) : "
                        + "\(result.steps) pas, diff abs max \(result.maxAbsoluteDifference), "
                        + "diff rel max \(result.maxRelativeDifference), "
                        + "argmax lm_head désaccords \(result.argmaxMismatches)/\(result.steps) — "
                        + (result.passed ? "PASS" : "FAIL"))
            }
            return
        }

        let profiler = MLXProfiler.shared
        let profileSession: ProfilingSession?
        if trace != nil {
            let session = ProfilingSession(config: .singleRun, subsystem: "com.qwen38mlx")
            session.title = "QWEN3.8 FLASH-NEXT LAYER BENCH (P0)"
            session.metadata["warmup_steps"] = String(warmup)
            session.metadata["measured_steps"] = String(steps)
            session.metadata["layer_kind"] = layerKind.rawValue
            session.metadata["compiled"] = compiled ? "true" : "false"
            session.metadata["shapeless"] = shapeless ? "true" : "false"
            session.metadata["async_interval"] = String(asyncInterval)
            session.metadata["skip_trivial_mask"] = skipTrivialMask ? "true" : "false"
            profiler.activeSession = session
            profiler.enable()
            profileSession = session
        } else {
            profileSession = nil
        }
        defer {
            if let profileSession {
                profiler.disable()
                if let trace {
                    try? ChromeTraceExporter.export(session: profileSession).write(
                        to: URL(fileURLWithPath: trace))
                }
            }
        }

        func percentile(_ sortedMs: [Double], _ p: Double) -> Double {
            guard !sortedMs.isEmpty else { return .nan }
            let index = Int((Double(sortedMs.count - 1) * p).rounded())
            return sortedMs[min(max(index, 0), sortedMs.count - 1)]
        }

        if let expertsQuantization {
            print("experts (switch_mlp): bits=\(expertsQuantization.bits) group_size=\(expertsQuantization.groupSize)")
        }
        for kind in kinds {
            print("--- couche \(kind.rawValue) ---")
            let result = Qwen4ExpLayerBench.run(
                kind: kind,
                warmupSteps: warmup,
                measuredSteps: steps,
                expertsQuantization: expertsQuantization,
                profiler: profiler,
                computeMode: Qwen4ExpLayerBenchComputeMode(
                    compiled: compiled, shapeless: shapeless, syncEvery: asyncInterval,
                    skipTrivialCausalMask: skipTrivialMask),
                fusionLevel: resolvedFusionLevel)
            let sortedMs = result.steps.map { $0.durationSeconds * 1000 }.sorted()
            let median = percentile(sortedMs, 0.5)
            let p10 = percentile(sortedMs, 0.1)
            let p90 = percentile(sortedMs, 0.9)
            let minMs = sortedMs.first ?? .nan
            let maxMs = sortedMs.last ?? .nan
            let meanCPU = result.steps.map(\.cpuPercent).reduce(0, +) / Double(result.steps.count)
            let meanGPU = result.steps.map(\.gpuPercent).reduce(0, +) / Double(result.steps.count)
            print("mémoire couche: \(ByteCountFormatter.string(fromByteCount: result.materializedBytes, countStyle: .file))")
            print("ms/pas — médiane \(String(format: "%.2f", median))  p10 \(String(format: "%.2f", p10))  p90 \(String(format: "%.2f", p90))  min \(String(format: "%.2f", minMs))  max \(String(format: "%.2f", maxMs))")
            print("CPU moyen: \(String(format: "%.1f", meanCPU)) %   GPU moyen: \(String(format: "%.1f", meanGPU)) %")
            if let profileSession {
                let prefix = "bench_\(kind.rawValue)_"
                profileSession.metadata[prefix + "median_ms"] = String(format: "%.3f", median)
                profileSession.metadata[prefix + "p10_ms"] = String(format: "%.3f", p10)
                profileSession.metadata[prefix + "p90_ms"] = String(format: "%.3f", p90)
                profileSession.metadata[prefix + "min_ms"] = String(format: "%.3f", minMs)
                profileSession.metadata[prefix + "max_ms"] = String(format: "%.3f", maxMs)
                profileSession.metadata[prefix + "cpu_percent_mean"] = String(format: "%.1f", meanCPU)
                profileSession.metadata[prefix + "gpu_percent_mean"] = String(format: "%.1f", meanGPU)
                profileSession.metadata[prefix + "materialized_bytes"] = String(result.materializedBytes)
            }
        }
        print("MLX mémoire active: \(ByteCountFormatter.string(fromByteCount: Int64(Memory.activeMemory), countStyle: .file))")
        print("MLX mémoire peak: \(ByteCountFormatter.string(fromByteCount: Int64(Memory.peakMemory), countStyle: .file))")
        if let profileSession {
            print(profileSession.generateReport())
            if let trace { print("trace profiler: \(trace)") }
        }
    }
}

struct FlashChatProbe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flash-chat-probe",
        abstract:
            "Discuter avec Flash-Next en streaming (générateur H2 : sampling, multi-tour, mesures)")

    @Argument(help: "Répertoire local du checkpoint qwen4_exp")
    var modelPath: String

    @Option(name: .long, help: "Prompt utilisateur du premier tour")
    var prompt: String

    @Option(name: .long, help: "Prompt utilisateur du second tour, en continuation (optionnel)")
    var secondPrompt: String?

    @Option(name: .long, help: "Image locale optionnelle à joindre au premier tour")
    var image: String?

    @Option(name: .long, help: "Preset d'échantillonnage : thinking ou instruct")
    var preset: String = "instruct"

    @Option(name: .long, help: "Température (avec --top-p/--top-k, surcharge le preset)")
    var temperature: Float?

    @Option(name: .long, help: "Top-p (avec --temperature/--top-k, surcharge le preset)")
    var topP: Float?

    @Option(name: .long, help: "Top-k (avec --temperature/--top-p, surcharge le preset)")
    var topK: Int?

    @Flag(name: .long, help: "Insérer le préfixe thinking dans le prompt")
    var thinking = false

    @Option(name: .long, help: "Nombre maximum de nouveaux tokens par tour")
    var maxNewTokens: Int = 256

    @Flag(
        name: .long,
        help: "Conserver toutes les couches Flash-Next en mémoire entre les tokens (expérimental)")
    var residentLayers = false

    @Option(
        name: .long,
        help: "Évaluer le graphe résident toutes les N couches (défaut: 1)")
    var residentEvalInterval = 1

    @Option(name: .long, help: "Écrire une trace Chrome/Perfetto à ce chemin")
    var trace: String?

    @Flag(
        name: .long,
        help:
            "Profiler chaque couche (\"Flash couche N\") au lieu des seules phases Prefill/Generation — coûte ~4,7 ms par frontière de phase (P0-c)"
    )
    var profileLayers = false

    @Flag(
        name: .long,
        help:
            "Mode résident : `asyncEval` sur chaque couche intermédiaire, `eval` bloquant sur la dernière seulement (P2-code (d), à trancher par P1)"
    )
    var residentAsync = false

    @Flag(
        name: .long,
        help:
            "Revenir à `loadArraysAndMetadata` (cache de fichiers du noyau) au lieu de la lecture F_NOCACHE des tenseurs résidents (P2-mem-a, défaut : F_NOCACHE actif)"
    )
    var cachedIO = false

    @Option(
        name: .long,
        help:
            "P2-fusion : niveau cumulatif F1-F6 appliqué à chaque couche (0 = chemin d'origine, défaut — F7 valide ce flag sur le vrai checkpoint avant tout changement de défaut)"
    )
    var fusionLevel: Int = 0

    func run() async throws {
        guard maxNewTokens > 0 else {
            throw ValidationError("--max-new-tokens doit être positif")
        }
        guard residentEvalInterval > 0 else {
            throw ValidationError("--resident-eval-interval doit être positif")
        }
        guard let resolvedFusionLevel = Qwen4ExpFusionLevel(rawValue: fusionLevel) else {
            throw ValidationError("--fusion-level doit appartenir à 0-6 (P2-fusion F1-F6)")
        }
        let samplingPreset: Qwen4ExpSamplingPreset
        if temperature != nil || topP != nil || topK != nil {
            samplingPreset = .custom(
                temperature: temperature ?? 0.7, topP: topP ?? 0.80, topK: topK ?? 20)
        } else {
            switch preset {
            case "thinking": samplingPreset = .thinking
            case "instruct": samplingPreset = .instruct
            default: throw ValidationError("--preset doit être 'thinking' ou 'instruct'")
            }
        }

        let directory = URL(fileURLWithPath: modelPath, isDirectory: true)
        let configuration = try Qwen4ExpConfiguration.load(from: directory)
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        // See FlashGenerateProbe: touch the Metal device before enabling
        // profiler phases on a fresh CLI process.
        _ = Device.defaultDevice()
        let profiler = MLXProfiler.shared
        let profileSession: ProfilingSession?
        if trace != nil {
            let session = ProfilingSession(config: .singleRun, subsystem: "com.qwen38mlx")
            session.title = "QWEN3.8 FLASH-NEXT CHAT PROBE"
            session.metadata["model"] = directory.lastPathComponent
            session.metadata["preset"] = preset
            session.metadata["thinking"] = thinking ? "true" : "false"
            profiler.activeSession = session
            profiler.enable()
            profileSession = session
        } else {
            profileSession = nil
        }
        defer {
            if let profileSession {
                profiler.disable()
                if let trace {
                    try? ChromeTraceExporter.export(session: profileSession).write(
                        to: URL(fileURLWithPath: trace))
                }
            }
        }

        let stopTokens: Set<Int32> = [
            configuration.textConfiguration.eosTokenID, Int32(248044), Int32(248046),
        ].compactMap { $0 }.reduce(into: Set<Int32>()) { $0.insert($1) }

        profiler.start("Flash globals")
        let model = try Qwen4ExpStreamingTextModel(
            directory: directory,
            layerLoadingMode: residentLayers ? .resident : .streamed,
            residentEvaluationInterval: residentEvalInterval,
            profileLayers: profileLayers,
            residentAsyncEval: residentAsync,
            uncachedIO: !cachedIO,
            fusionLevel: resolvedFusionLevel)
        profiler.end("Flash globals")
        let generator = Qwen4ExpStreamingGenerator(model: model)

        func runTurn(label: String, promptText: String, imageURL: URL?, continueConversation: Bool)
            async throws
        {
            let built: Qwen4ExpBuiltPrompt
            if continueConversation {
                built = Qwen4ExpPromptBuilder.buildContinuationTurn(
                    tokenizer: tokenizer, prompt: promptText, thinking: thinking)
            } else {
                built = try Qwen4ExpPromptBuilder.buildFirstTurn(
                    tokenizer: tokenizer, configuration: configuration, directory: directory,
                    prompt: promptText, imageURL: imageURL, thinking: thinking)
            }
            print("=== \(label) ===")
            var generatedIDs: [Int32] = []
            for try await event in generator.generate(
                promptTokenIDs: built.tokenIDs, positionIDs: built.positionIDs,
                visionEmbeddings: built.visionEmbeddings, imageTokenID: built.imageTokenID,
                options: .init(
                    maxNewTokens: maxNewTokens, stopTokenIDs: stopTokens, preset: samplingPreset,
                    continueConversation: continueConversation)
            ) {
                switch event {
                case .token(let token):
                    generatedIDs.append(token)
                    print(
                        tokenizer.decode(tokens: [Int(token)], skipSpecialTokens: false),
                        terminator: "")
                    fflush(stdout)
                case .finished(let summary):
                    print("")
                    print("prompt tokens: \(summary.promptTokenCount)")
                    print("generated ids: \(generatedIDs)")
                    print("TTFT: \(summary.timeToFirstToken.map { String(format: "%.3fs", $0) } ?? "-")")
                    print("prefill: \(String(format: "%.3fs", summary.prefillTime))")
                    print("decode: \(String(format: "%.3fs", summary.decodeTime))")
                    print(
                        "couches: \(summary.layerVisitCount) visites · load cumulé \(String(format: "%.3fs", summary.layerLoadTime))"
                    )
                    if let profileSession {
                        profileSession.metadata["\(label)_prompt_tokens"] = String(
                            summary.promptTokenCount)
                        profileSession.metadata["\(label)_generated_ids"] = generatedIDs.map(
                            String.init
                        ).joined(separator: ",")
                    }
                }
            }
        }

        try await runTurn(
            label: "tour 1", promptText: prompt,
            imageURL: image.map { URL(fileURLWithPath: $0) }, continueConversation: false)
        if let secondPrompt {
            try await runTurn(
                label: "tour 2", promptText: secondPrompt, imageURL: nil,
                continueConversation: true)
        }

        print("MLX mémoire active: \(ByteCountFormatter.string(fromByteCount: Int64(Memory.activeMemory), countStyle: .file))")
        print("MLX mémoire peak: \(ByteCountFormatter.string(fromByteCount: Int64(Memory.peakMemory), countStyle: .file))")
        let ngramCacheStats = model.ngramCacheStats()
        if let profileSession {
            profileSession.metadata["ngram_cache_hits"] = String(ngramCacheStats.hits)
            profileSession.metadata["ngram_cache_misses"] = String(ngramCacheStats.misses)
            profileSession.metadata["ngram_cache_entries"] = String(ngramCacheStats.entries)
            profileSession.metadata["ngram_cache_hit_rate"] = ngramCacheStats.hitRate.map {
                String(format: "%.4f", $0)
            } ?? "n/a"
            print(profileSession.generateReport())
            if let trace { print("trace profiler: \(trace)") }
        }
    }
}

/// Score a fixed continuation without sampling. This is the Swift half of
/// Q-B: the same prompt/continuation token IDs can be scored by Python and the
/// resulting mean log-probability, argmax agreement and ranks compared.
struct FlashTeacherForcedScore: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flash-teacher-forced-score",
        abstract: "Scorer une continuation Flash-Next en teacher-forcing")

    @Argument(help: "Répertoire local du checkpoint Flash-Next")
    var modelPath: String

    @Option(name: .long, help: "Prompt utilisateur, rendu avec le template ChatML")
    var prompt: String

    @Option(name: .long, help: "Continuation texte exacte à scorer")
    var continuation: String?

    @Option(name: .long, help: "IDs de continuation séparés par des virgules")
    var continuationIds: String?

    @Flag(name: .long, help: "Rendre le prompt avec thinking activé")
    var thinking = false

    @Flag(name: .long, help: "Conserver toutes les couches en mémoire")
    var residentLayers = false

    @Option(
        name: .long,
        help: "Évaluer le graphe résident toutes les N couches (défaut: 1 — mesuré plus rapide et stable que le batching, voir docs/knowledge/log.md 2026-09-05)")
    var residentEvalInterval = 1

    @Option(name: .long, help: "Marge minimale pour l'accord argmax confiant")
    var confidentMargin: Float = 0.5

    @Option(name: .long, help: "Fixture Python teacher-forced à comparer sans second forward")
    var pythonFixture: String?

    func run() async throws {
        guard confidentMargin >= 0 else {
            throw ValidationError("--confident-margin doit être positif ou nul")
        }
        guard continuation != nil || continuationIds != nil else {
            throw ValidationError("Fournir --continuation ou --continuation-ids")
        }
        guard residentEvalInterval > 0 else {
            throw ValidationError("--resident-eval-interval doit être positif")
        }
        if continuation != nil && continuationIds != nil {
            throw ValidationError("Utiliser une seule forme de continuation")
        }

        let directory = URL(fileURLWithPath: modelPath, isDirectory: true)
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        let messages: [[String: any Sendable]] = [[
            "role": "user",
            "content": prompt
        ]]
        let promptIDs = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: nil,
            additionalContext: [
                "enable_thinking": thinking,
                "reasoning_effort": "low"
            ]).map(Int32.init)
        let suffixIDs: [Int32]
        if let continuation {
            suffixIDs = tokenizer.encode(
                text: continuation, addSpecialTokens: false).map(Int32.init)
        } else {
            let values = continuationIds!.split(separator: ",").map {
                $0.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            guard !values.isEmpty, values.allSatisfy({ Int32($0) != nil }) else {
                throw ValidationError("--continuation-ids doit contenir des entiers séparés par des virgules")
            }
            suffixIDs = values.map { Int32($0)! }
        }
        guard !suffixIDs.isEmpty else {
            throw ValidationError("La continuation ne produit aucun token")
        }

        _ = Device.defaultDevice()
        let model = try Qwen4ExpStreamingTextModel(
            directory: directory,
            layerLoadingMode: residentLayers ? .resident : .streamed,
            residentEvaluationInterval: residentEvalInterval)
        let score = try model.scoreTeacherForced(
            promptTokenIDs: promptIDs,
            continuationTokenIDs: suffixIDs,
            confidentMargin: confidentMargin)
        print("prompt tokens: \(score.promptTokenCount)")
        print("continuation tokens: \(score.continuationTokenCount)")
        print("mean logprob: \(String(format: "%.6f", score.meanLogProbability))")
        print("accord argmax: \(String(format: "%.1f%%", score.argmaxAgreement * 100))")
        print("accord confiant (marge ≥ \(confidentMargin)): \(String(format: "%.1f%%", score.confidentAgreement * 100)) · \(score.confidentTokenCount) tokens")
        print("rang cible moyen: \(String(format: "%.2f", score.meanTargetRank))")
        print("prefill: \(String(format: "%.3fs", score.prefillTime))")
        for (index, token) in score.tokens.enumerated() {
            print("  #\(index + 1) cible=\(token.tokenID) argmax=\(token.argmaxTokenID) rang=\(token.targetRank) logprob=\(String(format: "%.5f", token.logProbability)) marge=\(String(format: "%.5f", token.argmaxMargin))")
        }
        if let pythonFixture {
            let report = try Qwen4ExpTeacherForcedParity.compare(
                score: score,
                fixtureURL: URL(fileURLWithPath: pythonFixture))
            print("Python Q-B: IDs alignés · \(report.tokenCount) tokens")
            print("  Δ logprob max/moy: \(String(format: "%.6f", report.maxLogProbabilityAbsoluteError))/\(String(format: "%.6f", report.meanLogProbabilityAbsoluteError))")
            print("  Δ marge max: \(String(format: "%.6f", report.maxMarginAbsoluteError)) · Δ rang max: \(report.maxTargetRankDifference)")
            print("  accord argmax Swift/Python: \(String(format: "%.1f%%", report.argmaxAgreement * 100))/\(String(format: "%.1f%%", report.pythonArgmaxAgreement * 100)) · Δ rang moyen: \(String(format: "%.3f", report.targetRankMeanDifference))")
        }
        let load = score.layerReports.reduce(0) { $0 + $1.loadDuration }
        let forward = score.layerReports.reduce(0) { $0 + $1.forwardDuration }
        print("couches: \(score.layerReports.count) · load cumulé \(String(format: "%.3fs", load)) · forward cumulé \(String(format: "%.3fs", forward))")
        print("MLX mémoire active: \(ByteCountFormatter.string(fromByteCount: Int64(Memory.activeMemory), countStyle: .file))")
        print("MLX mémoire peak: \(ByteCountFormatter.string(fromByteCount: Int64(Memory.peakMemory), countStyle: .file))")
    }
}

/// Inspect the exact ChatML prompt produced by swift-transformers without
/// loading any MLX weights. This keeps template/EOS diagnostics independent
/// from the very slow layer-streamed Flash forward.
struct FlashTemplateProbe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flash-template-probe",
        abstract: "Inspecter le template ChatML et les bornes thinking/EOS")

    @Argument(help: "Répertoire local du checkpoint Flash-Next")
    var modelPath: String

    @Option(name: .long, help: "Prompt utilisateur")
    var prompt: String

    @Flag(name: .long, help: "Demander un prompt de génération avec thinking activé")
    var thinking = false

    @Option(
        name: .long,
        help: "JSON de référence Python contenant {\"ids\": [..]} à comparer")
    var reference: String?

    func run() async throws {
        let directory = URL(fileURLWithPath: modelPath, isDirectory: true)
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        let messages: [[String: any Sendable]] = [[
            "role": "user",
            "content": prompt
        ]]
        let ids = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: nil,
            additionalContext: [
                "enable_thinking": thinking,
                "reasoning_effort": "low"
            ])
        let decoded = tokenizer.decode(tokens: ids, skipSpecialTokens: false)

        print("thinking: \(thinking ? "on" : "off")")
        print("tokens: \(ids.count)")
        print("ids: \(ids.map(String.init).joined(separator: ","))")
        print("decoded: \(decoded.debugDescription)")

        let names = [
            "<|im_start|>", "<|im_end|>", "<|endoftext|>",
            "<|user|>", "<|assistant|>", "<think>", "</think>"
        ]
        for name in names {
            let value = tokenizer.convertTokenToId(name).map(String.init) ?? "absent"
            print("token \(name): \(value)")
        }
        let suffixStart = max(0, ids.count - 24)
        print("suffix ids: \(ids[suffixStart...].map(String.init).joined(separator: ","))")

        if let reference {
            let data = try Data(contentsOf: URL(fileURLWithPath: reference))
            guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let rawIDs = object["ids"] as? [NSNumber]
            else {
                throw ValidationError("La référence doit contenir un tableau JSON `ids`.")
            }
            let expected = rawIDs.map(\.intValue)
            let commonCount = min(ids.count, expected.count)
            let firstDifference = (0..<commonCount).first { ids[$0] != expected[$0] }
            let same = ids == expected
            print("référence: \(reference)")
            print("comparaison: \(same ? "IDENTIQUE" : "DIFFÉRENTE")")
            print("référence tokens: \(expected.count)")
            if let firstDifference = firstDifference {
                print("première différence: index \(firstDifference), swift=\(ids[firstDifference]), python=\(expected[firstDifference])")
            } else if ids.count != expected.count {
                print("différence de longueur après le préfixe commun: swift=\(ids.count), python=\(expected.count)")
            }
        }
    }
}

/// Compare a handful of rows from the row-wise n-gram reader with the exact
/// dequantization of the corresponding safetensors rows. This deliberately
/// touches only a few rows of one shard, so it is safe to run on the full
/// Flash-Next checkpoint without paying the 32 GB table materialization.
struct FlashNGramParity: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flash-ngram-parity",
        abstract: "Comparer les lignes n-gram lazy à la référence eager")

    @Argument(help: "Répertoire local du checkpoint Flash-Next")
    var modelPath: String

    @Option(name: .long, help: "Index de shard à vérifier")
    var shard = 0

    @Option(name: .long, help: "Lignes à vérifier, séparées par des virgules")
    var rows = "0,1,12345"

    func run() async throws {
        let directory = URL(fileURLWithPath: modelPath, isDirectory: true)
        let configuration = try Qwen4ExpConfiguration.load(from: directory)
        guard let quantization = Qwen4ExpQuantizationSpec(configuration.quantization) else {
            throw ValidationError("Le checkpoint doit fournir une quantification explicite.")
        }
        let layerIndex = 1 // ple_layer_ids=[2] : couche 2 du modèle, index Swift 1
        let prefix = "language_model.model.layers.\(layerIndex)."
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        let indexData = try Data(contentsOf: indexURL)
        guard let object = try JSONSerialization.jsonObject(with: indexData)
                as? [String: Any],
              let weightMap = object["weight_map"] as? [String: String] else {
            throw ValidationError("Index safetensors Flash-Next invalide.")
        }

        var rawKeysByShardFile = [String: [String]]()
        for (key, file) in weightMap where key.hasPrefix(prefix)
            && key.contains(".ngram_embedding.shard_") {
            rawKeysByShardFile[file, default: []].append(key)
        }
        guard !rawKeysByShardFile.isEmpty else {
            throw ValidationError("Aucun shard n-gram trouvé pour la couche PLE.")
        }

        let heads = configuration.textConfiguration.headsPerNgram ?? 8
        let dimensions = (configuration.textConfiguration.pleEmbedDim
            ?? configuration.textConfiguration.hiddenSize)
            / ((configuration.textConfiguration.ngramSize - 1) * heads)
        let storage = try Qwen4ExpLazyNGramStorage(
            directory: directory,
            rawKeysByShardFile: rawKeysByShardFile,
            layerIndex: layerIndex,
            shardCount: configuration.textConfiguration.splitNgramParts,
            dimensions: dimensions,
            quantization: quantization)

        let requestedRows = try rows.split(separator: ",").map { part -> Int in
            guard let value = Int(part.trimmingCharacters(in: .whitespaces)), value >= 0 else {
                throw ValidationError("Ligne n-gram invalide : \(part)")
            }
            return value
        }
        guard !requestedRows.isEmpty else {
            throw ValidationError("Au moins une ligne n-gram est nécessaire.")
        }
        guard let shardFile = rawKeysByShardFile.first(where: { keys in
            keys.value.contains { $0.hasSuffix(".shard_\(shard).weight") }
        })?.key else {
            throw ValidationError("Shard n-gram inexistant : \(shard)")
        }
        let shardKeys = rawKeysByShardFile[shardFile] ?? []
        guard let weightKey = shardKeys.first(where: { $0.hasSuffix(".shard_\(shard).weight") }),
              let scalesKey = shardKeys.first(where: { $0.hasSuffix(".shard_\(shard).scales") }),
              let biasesKey = shardKeys.first(where: { $0.hasSuffix(".shard_\(shard).biases") }) else {
            throw ValidationError("Paramètres quantifiés incomplets pour le shard \(shard).")
        }
        let arrays = try loadArraysAndMetadata(
            url: directory.appendingPathComponent(shardFile), stream: .cpu).0
        guard let packed = arrays[weightKey],
              let scales = arrays[scalesKey],
              let biases = arrays[biasesKey] else {
            throw ValidationError("Données absentes du shard \(shardFile).")
        }
        let indices = MLXArray(requestedRows.map(Int32.init)).asType(.int32)
        let eager = MLX.dequantized(
            packed[indices], scales: scales[indices], biases: biases[indices],
            groupSize: quantization.groupSize, bits: quantization.bits,
            mode: quantization.mode)
        let lazy = storage.lookup(shard: shard, rows: requestedRows.map(Int32.init))
        // Repeat the same lookup to exercise the bounded raw-row cache. The
        // second call must remain bit-exact while avoiding another file read.
        let cached = storage.lookup(shard: shard, rows: requestedRows.map(Int32.init))
        eval(eager, lazy, cached)
        let delta = abs(eager.asType(.float32) - lazy.asType(.float32))
        let cacheDelta = abs(lazy.asType(.float32) - cached.asType(.float32))
        let cacheStats = storage.cacheStats()
        print("shard: \(shard) · lignes: \(requestedRows)")
        print("forme eager/lazy: \(eager.shape) / \(lazy.shape)")
        print("max |delta|: \(String(format: "%.9g", delta.max().item(Float.self)))")
        print("mean |delta|: \(String(format: "%.9g", delta.sum().item(Float.self) / Float(delta.size)))")
        print("parité n-gram lazy: \(delta.max().item(Float.self) == 0 ? "IDENTIQUE" : "DIFFÉRENTE")")
        print("cache lignes: hits=\(cacheStats.hits) · misses=\(cacheStats.misses) · entrées=\(cacheStats.entries) · max |delta| répétition=\(String(format: "%.9g", cacheDelta.max().item(Float.self)))")
    }
}

struct FlashMTPProbe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flash-mtp-probe",
        abstract: "Charger et sonder le predictor MTP natif de Flash-Next")

    @Argument(help: "Répertoire local du checkpoint qwen4_exp")
    var modelPath: String

    @Flag(name: .long, help: "Ne pas matérialiser les poids après leur chargement")
    var lazy = false

    @Flag(name: .long, help: "Exécuter un forward MTP réel sur un token synthétique")
    var forward = false

    @Flag(
        name: .long,
        help: "Exécuter un round cible→draft→vérification→commit borné (texte uniquement)")
    var round = false

    @Option(name: .long, help: "Prompt du round MTP réel")
    var prompt = "Explique en une phrase qui est Xi Jinping."

    @Option(name: .long, help: "Taille du bloc MTP du round réel (au moins 2)")
    var blockSize = 2

    func run() async throws {
        guard blockSize >= 2 else {
            throw ValidationError("--block-size doit être au moins 2")
        }
        let directory = URL(fileURLWithPath: modelPath, isDirectory: true)
        let started = ContinuousClock.now
        let loaded = try Qwen4ExpMTPLoader.load(
            from: directory,
            materialize: !lazy)
        print("MTP Flash-Next : predictor QSA chargé")
        print("tenseurs : \(loaded.tensorCount)")
        print("shards : \(loaded.shardCount)")
        if loaded.materializedBytes > 0 {
            print("poids matérialisés : \(ByteCountFormatter.string(fromByteCount: loaded.materializedBytes, countStyle: .file))")
        } else {
            print("poids matérialisés : non (mode lazy)")
        }
        print("durée : \(String(format: "%.3fs", durationSeconds(ContinuousClock.now - started)))")
        if forward {
            guard !lazy else {
                throw ValidationError("--forward nécessite le chargement matérialisé (retirer --lazy)")
            }
            let global = try Qwen4ExpGlobalCheckpointLoader.load(from: directory)
            let ids = MLXArray([Int32(1)]).reshaped([1, 1])
            let hidden = MLXArray.zeros(
                [1, 1, loaded.model.configuration.hiddenSize * loaded.model.configuration.hcCount],
                dtype: .bfloat16)
            let cache = loaded.model.makeCache()
            let output = loaded.model(
                inputEmbeddings: global.model.embed(ids),
                targetHidden: hidden,
                inputIDs: ids,
                cache: cache)
            eval(output)
            let logitsHidden = loaded.model.logitsHidden(from: output)
            eval(logitsHidden)
            print("forward MTP réel : OK · état \(output.shape) · logits hidden \(logitsHidden.shape) · offset \(cache.offset)")
        }
        if round {
            guard !lazy else {
                throw ValidationError("--round nécessite le chargement matérialisé (retirer --lazy)")
            }
            try await runRealRound(
                directory: directory,
                mtp: loaded.model,
                prompt: prompt,
                blockSize: blockSize)
        }
        print("statut : brique MTP chargée ; loop spéculatif Flash à qualifier séparément")
    }

    private func runRealRound(
        directory: URL,
        mtp: Qwen4ExpMTPPredictor,
        prompt: String,
        blockSize: Int
    ) async throws {
        _ = Device.defaultDevice()
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        let promptIDs = try tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": prompt]],
            tools: nil,
            additionalContext: [
                "enable_thinking": false,
                "reasoning_effort": "low"
            ]).map(Int32.init)
        guard !promptIDs.isEmpty else {
            throw ValidationError("Le template n'a produit aucun token")
        }

        print("round MTP réel : prompt \(promptIDs.count) tokens · bloc \(blockSize)")
        let target = try Qwen4ExpStreamingTextModel(directory: directory)
        let targetStarted = ContinuousClock.now
        let promptArray = MLXArray(promptIDs).reshaped([1, promptIDs.count])
        let prefill = try target.forward(inputIDs: promptArray)
        eval(prefill.logits)
        eval(prefill.preMixerHidden)
        let firstBonus = greedyToken(from: prefill.logits[0..., -1, 0...])

        let engine = Qwen4ExpFlashMTPDraftEngine(target: target, predictor: mtp)
        let state = engine.makeState()
        try engine.prepare(
            promptTokenIDs: promptArray,
            targetHidden: prefill.preMixerHidden,
            firstBonus: firstBonus,
            state: state)
        let lastHidden = prefill.preMixerHidden[0..., (-1)..., 0...]
        let drafts = try engine.draftBlock(
            lastToken: firstBonus,
            lastHidden: lastHidden,
            blockSize: blockSize,
            state: state)
        eval(drafts)
        let draftIDs = drafts.flattened().asArray(Int32.self)

        let targetSnapshot = target.snapshot()
        let verifyTokens = concatenated([
            firstBonus.reshaped([1, 1]), drafts
        ], axis: 1).asType(.int32)
        let verification = try target.forward(inputIDs: verifyTokens)
        eval(verification.logits)
        eval(verification.preMixerHidden)
        let targetIDs = (0 ..< verifyTokens.dim(1)).map { index in
            greedyToken(from: verification.logits[0..., index, 0...])
                .item(Int32.self)
        }
        let walk = Qwen38SpeculativeWalk.walk(
            drafts: draftIDs, targets: targetIDs, budget: Int.max)
        let finalToken = MLXArray([walk.emitted.last ?? targetIDs[walk.accepted]])

        let hiddenForCommit: MLXArray
        if walk.accepted < draftIDs.count {
            target.restore(targetSnapshot)
            let replayTokens = concatenated([
                firstBonus.reshaped([1, 1]),
                drafts[0..., 0 ..< walk.accepted]
            ], axis: 1).asType(.int32)
            let replay = try target.forward(inputIDs: replayTokens)
            eval(replay.preMixerHidden)
            hiddenForCommit = replay.preMixerHidden
            print("rollback cible : oui · rejoué \(replayTokens.dim(1)) tokens")
        } else {
            hiddenForCommit = verification.preMixerHidden
            print("rollback cible : non · bloc entièrement accepté")
        }
        try engine.commit(
            targetHidden: hiddenForCommit,
            draftTokens: drafts,
            acceptedCount: walk.accepted,
            finalToken: finalToken,
            state: state)

        print("drafts : \(draftIDs)")
        print("cible  : \(targetIDs)")
        print("acceptation : \(walk.accepted)/\(draftIDs.count) · correction \(finalToken.item(Int32.self))")
        print("état drafter : offset \(state.nextPosition) · prochaine proposition préparée")
        print("durée round borné : \(String(format: "%.3fs", durationSeconds(ContinuousClock.now - targetStarted)))")
    }

    private func greedyToken(from logits: MLXArray) -> MLXArray {
        let token = ArgMaxSampler().sample(logits: logits)
        eval(token)
        return token
    }
}

struct FlashQSAParity: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flash-qsa-parity",
        abstract: "Comparer le chemin QSA Swift avec un fixture Python MLX")

    @Argument(help: "Fixture safetensors produit par scripts/qwen4-exp-qsa-reference.py")
    var fixture: String

    @Option(name: .long, help: "Budget de blocs QSA")
    var budget: Int = 8

    @Option(name: .long, help: "Ratio de compression des blocs")
    var compressRatio: Int = 4

    func run() throws {
        let json = """
        {
          "hidden_size": 8, "num_hidden_layers": 4,
          "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
          "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
          "full_attention_interval": 4,
          "linear_num_key_heads": 2, "linear_num_value_heads": 4,
          "linear_key_head_dim": 4, "linear_value_head_dim": 4, "linear_conv_kernel_dim": 4,
          "num_experts": 4, "num_experts_per_tok": 1,
          "moe_intermediate_size": 4, "shared_expert_intermediate_size": 4,
          "indexer_budget": 8, "indexer_compress_ratio": 4,
          "indexer_head_dim": 128, "indexer_kv_heads": 1, "indexer_n_heads": 4,
          "hc_count": 4, "hc_lowrank": 2,
          "ngram_size": 3, "ngram_vocab_size_base": 32,
          "split_ngram_parts": 128, "ple_layer_ids": [2], "ple_conv_kernel_size": 4,
          "vocab_size": 32, "max_position_embeddings": 128
        }
        """
        let configuration = try JSONDecoder().decode(
            Qwen4ExpTextConfiguration.self, from: Data(json.utf8))
        let report = try Qwen4ExpQSAParity.compareFixture(
            at: URL(fileURLWithPath: fixture),
            indexer: Qwen4ExpQSAIndexer(configuration: configuration),
            budget: budget,
            compressRatio: compressRatio)
        print("fixture: \(fixture)")
        for name in report.maxAbsoluteError.keys.sorted() {
            let maximum = report.maxAbsoluteError[name] ?? 0
            let mean = report.meanAbsoluteError[name] ?? 0
            print("\(name): max \(String(format: "%.6g", maximum)) · mean \(String(format: "%.6g", mean))")
        }
        print("QSA parité: OK · pire max \(String(format: "%.6g", report.worstMaxAbsoluteError))")
    }
}

struct FlashMRoPEParity: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flash-mrope-parity",
        abstract: "Comparer les tables MRoPE multimodales avec un fixture Python MLX")

    @Argument(help: "Fixture safetensors produit par scripts/qwen4-exp-mrope-reference.py")
    var fixture: String

    func run() throws {
        let report = try Qwen4ExpMRoPEParity.compareFixture(
            at: URL(fileURLWithPath: fixture))
        print("fixture: \(fixture)")
        for name in report.maxAbsoluteError.keys.sorted() {
            let maximum = report.maxAbsoluteError[name] ?? 0
            let mean = report.meanAbsoluteError[name] ?? 0
            print("\(name): max \(String(format: "%.6g", maximum)) · mean \(String(format: "%.6g", mean))")
        }
        print("MRoPE parité: OK · pire max \(String(format: "%.6g", report.worstMaxAbsoluteError))")
    }
}

struct FlashVisionParity: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flash-vision-parity",
        abstract: "Comparer la tour vision Flash-Next avec un fixture Python MLX")

    @Argument(help: "Répertoire du checkpoint Flash-Next")
    var modelPath: String

    @Argument(help: "Fixture safetensors produit par scripts/qwen4-exp-vision-reference.py")
    var fixture: String

    func run() throws {
        let report = try Qwen4ExpVisionParity.compareFixture(
            modelDirectory: URL(fileURLWithPath: modelPath, isDirectory: true),
            fixtureURL: URL(fileURLWithPath: fixture))
        print("fixture: \(fixture)")
        print("sortie: \(report.outputShape)")
        print("vision: max \(String(format: "%.6g", report.maxAbsoluteError)) · mean \(String(format: "%.6g", report.meanAbsoluteError))")
        for name in report.stageMaxAbsoluteError.keys.sorted() {
            let value = report.stageMaxAbsoluteError[name] ?? 0
            print("\(name): max \(String(format: "%.6g", value))")
        }
        print("Vision parité: OK")
    }
}

struct FlashLanguageParity: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flash-language-parity",
        abstract: "Comparer la couche langage 0 Flash-Next avec un fixture Python MLX")

    @Argument(help: "Répertoire du checkpoint Flash-Next")
    var modelPath: String

    @Argument(help: "Fixture safetensors produit par scripts/qwen4-exp-language-reference.py")
    var fixture: String

    func run() throws {
        let report = try Qwen4ExpLanguageParity.compareFixture(
            modelDirectory: URL(fileURLWithPath: modelPath, isDirectory: true),
            fixtureURL: URL(fileURLWithPath: fixture))
        print("fixture: \(fixture)")
        print("sortie: \(report.outputShape)")
        print("layer 0: max \(String(format: "%.6g", report.outputMaxAbsoluteError)) · mean \(String(format: "%.6g", report.outputMeanAbsoluteError))")
        for name in report.stageMaxAbsoluteError.keys.sorted() {
            print("\(name): max \(String(format: "%.6g", report.stageMaxAbsoluteError[name] ?? 0))")
        }
        for name in report.cacheMaxAbsoluteError.keys.sorted() {
            print("\(name): max \(String(format: "%.6g", report.cacheMaxAbsoluteError[name] ?? 0))")
        }
        print("Parité bornée langage couche 0: OK · tolérance de sortie 1.0")
    }
}

struct FlashGlobalParity: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flash-global-parity",
        abstract: "Comparer embedding, mixer final et lm-head avec un fixture Python MLX")

    @Argument(help: "Répertoire du checkpoint Flash-Next")
    var modelPath: String

    @Argument(help: "Fixture safetensors produit par scripts/qwen4-exp-global-reference.py")
    var fixture: String

    func run() throws {
        let report = try Qwen4ExpGlobalParity.compareFixture(
            modelDirectory: URL(fileURLWithPath: modelPath, isDirectory: true),
            fixtureURL: URL(fileURLWithPath: fixture))
        print("fixture: \(fixture)")
        print("embedding: max \(String(format: "%.6g", report.embeddedMaxAbsoluteError)) · mean \(String(format: "%.6g", report.embeddedMeanAbsoluteError))")
        print("réduction hyper: max \(String(format: "%.6g", report.reducedMaxAbsoluteError)) · mean \(String(format: "%.6g", report.reducedMeanAbsoluteError))")
        print("lm-head/logits: max \(String(format: "%.6g", report.logitsMaxAbsoluteError)) · mean \(String(format: "%.6g", report.logitsMeanAbsoluteError))")
        print("Parité globaux Flash-Next: OK · erreurs nulles sur ce fixture")
    }
}

struct FlashSingleLayerParity: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flash-single-layer-parity",
        abstract: "Comparer le chemin embedding → couche 0 → lm-head avec Python MLX")

    @Argument(help: "Répertoire du checkpoint Flash-Next")
    var modelPath: String

    @Argument(help: "Fixture safetensors produit par scripts/qwen4-exp-single-layer-reference.py")
    var fixture: String

    func run() throws {
        let report = try Qwen4ExpSingleLayerParity.compareFixture(
            modelDirectory: URL(fileURLWithPath: modelPath, isDirectory: true),
            fixtureURL: URL(fileURLWithPath: fixture))
        print("fixture: \(fixture)")
        print("embedding: max \(String(format: "%.6g", report.embeddedMaxAbsoluteError)) · mean \(String(format: "%.6g", report.embeddedMeanAbsoluteError))")
        print("couche 0: max \(String(format: "%.6g", report.layerMaxAbsoluteError)) · mean \(String(format: "%.6g", report.layerMeanAbsoluteError))")
        for name in report.stageMaxAbsoluteError.keys.sorted() {
            print("\(name): max \(String(format: "%.6g", report.stageMaxAbsoluteError[name] ?? 0))")
        }
        print("réduction hyper: max \(String(format: "%.6g", report.reducedMaxAbsoluteError)) · mean \(String(format: "%.6g", report.reducedMeanAbsoluteError))")
        print("lm-head/logits: max \(String(format: "%.6g", report.logitsMaxAbsoluteError)) · mean \(String(format: "%.6g", report.logitsMeanAbsoluteError))")
        print("Parité bornée single-layer Flash-Next: OK · diagnostic, pas qualification logits")
    }
}

struct FlashPublicLayerParity: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flash-public-layer-parity",
        abstract: "Comparer un appel public de couche Flash-Next avec Python MLX")

    @Argument(help: "Répertoire du checkpoint Flash-Next")
    var modelPath: String

    @Argument(help: "Fixture produit par qwen4-exp-public-layer-reference.py")
    var fixture: String

    @Flag(name: .long, help: "Déquantifier les poids de la couche (E3, sans PLE)")
    var dequantized = false

    func run() throws {
        let report = try Qwen4ExpPublicLayerParity.compareFixture(
            modelDirectory: URL(fileURLWithPath: modelPath, isDirectory: true),
            fixtureURL: URL(fileURLWithPath: fixture), dequantized: dequantized)
        print("fixture: \(fixture)")
        print("couche \(report.layerIndex): sortie max \(String(format: "%.6g", report.outputMaxAbsoluteError)) · mean \(String(format: "%.6g", report.outputMeanAbsoluteError))")
        print("embedding: max \(String(format: "%.6g", report.embeddedMaxAbsoluteError)) · mean \(String(format: "%.6g", report.embeddedMeanAbsoluteError))")
        for name in report.stageMaxAbsoluteError.keys.sorted() {
            let metric = report.stageMetrics[name]
            print("\(name): max \(String(format: "%.6g", report.stageMaxAbsoluteError[name] ?? 0))" +
                  " · rel \(String(format: "%.6g", metric?.relativeRMSError ?? 0))" +
                  " · cos \(String(format: "%.6g", metric?.cosineSimilarity ?? 0))")
        }
        if let routing = report.moeRoutingMetrics {
            print("routage MoE: \(routing.positionsWithDifferentMembership)/\(routing.positions) positions différentes" +
                  " · \(routing.differingExpertAssignments) affectations · top-\(routing.topK)")
        }
        print("Parité appel public Flash-Next: OK · couche \(report.layerIndex)" +
              (dequantized ? " · E3 déquantifiée" : " · 4-bit"))
    }
}

struct FlashSelectedLayersParity: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flash-selected-layers-parity",
        abstract: "Comparer une chaîne de couches Flash-Next avec Python MLX")

    @Argument(help: "Répertoire du checkpoint Flash-Next")
    var modelPath: String

    @Argument(help: "Fixture produit par qwen4-exp-selected-layers-reference.py")
    var fixture: String

    func run() throws {
        let report = try Qwen4ExpSelectedLayersParity.compareFixture(
            modelDirectory: URL(fileURLWithPath: modelPath, isDirectory: true),
            fixtureURL: URL(fileURLWithPath: fixture))
        print("fixture: \(fixture)")
        for layer in report.layerMaxAbsoluteError.keys.sorted() {
            let chained = report.chainedLayerMetrics[layer]
            let rebased = report.reanchoredLayerMetrics[layer]
            print("couche \(layer): chaîne max \(String(format: "%.6g", report.layerMaxAbsoluteError[layer] ?? 0))" +
                  " · rel \(String(format: "%.6g", chained?.relativeRMSError ?? 0))" +
                  " · E2 rebasée max \(String(format: "%.6g", rebased?.maxAbsoluteError ?? 0))" +
                  " · rel \(String(format: "%.6g", rebased?.relativeRMSError ?? 0))")
        }
        print("E1 mixer Swift sur état Python: max \(String(format: "%.6g", report.mixerFromPythonMetrics.maxAbsoluteError))" +
              " · rel \(String(format: "%.6g", report.mixerFromPythonMetrics.relativeRMSError))" +
              " · cos \(String(format: "%.6g", report.mixerFromPythonMetrics.cosineSimilarity))")
        print("réduction chaîne: max \(String(format: "%.6g", report.reducedMaxAbsoluteError))")
        let logits = report.logitsFromPythonMetrics
        print("E1 logits Swift sur état Python: max \(String(format: "%.6g", logits.values.maxAbsoluteError))" +
              " · rel \(String(format: "%.6g", logits.values.relativeRMSError))" +
              " · cos \(String(format: "%.6g", logits.values.cosineSimilarity))" +
              " · marge Python \(String(format: "%.6g", logits.pythonTop1Margin))")
        print("logits chaîne: max \(String(format: "%.6g", report.logitsMaxAbsoluteError))")
        print("token chaîne Swift/Python: \(report.swiftNextToken)/\(report.pythonNextToken) · \(report.tokensMatch ? "IDENTIQUE" : "DIFFÉRENT")")
        print("E1 token Swift/Python: \(logits.swiftTopToken)/\(logits.pythonTopToken) · rang Swift dans Python \(logits.swiftTokenRankInPython) · rang Python dans Swift \(logits.pythonTokenRankInSwift)")
    }
}

struct Serve: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Démarrer l'API d'inférence Qwen3.8 sur le LAN")

    @Option(name: .long, help: "Répertoire local du modèle")
    var modelPath: String

    @Option(name: .long, help: "Port HTTP (défaut : 8848)")
    var port: Int = Qwen38InferenceServer.defaultPort

    @Option(name: .long, help: "Clé Bearer optionnelle")
    var apiKey: String?

    func run() async throws {
        let runtime = Qwen38Runtime()
        print("Chargement du modèle…")
        try await runtime.load(from: URL(fileURLWithPath: modelPath, isDirectory: true))
        let server = Qwen38InferenceServer(runtime: runtime)
        try await server.start(
            port: port,
            apiKey: apiKey,
            modelsDirectory: URL(fileURLWithPath: modelPath, isDirectory: true)
                .deletingLastPathComponent())
        print("Qwen3.8 écoute sur http://0.0.0.0:\(port)")
        print("POST /v1/chat/completions · GET /v1/models · GET /metrics")
        defer { Task { await server.stop() } }
        while !Task.isCancelled {
            try await Task.sleep(for: .seconds(3_600))
        }
    }
}

struct MTPParity: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mtp-parity",
        abstract: "Comparer les IDs M2 local et M1 upstream")

    @Option(name: .long, help: "Répertoire local du modèle")
    var modelPath: String

    @Option(name: .long, help: "Prompt utilisateur")
    var prompt: String

    @Option(name: .long, help: "Chemin d'une image à joindre au prompt")
    var image: String?

    @Option(name: .long, help: "Nombre maximum de tokens")
    var maxTokens: Int = 32

    @Option(name: .long, help: "Largeur totale verify = bonus + drafts")
    var blockSize: Int = 2

    func run() async throws {
        let runtime = Qwen38Runtime()
        try await runtime.load(
            from: URL(fileURLWithPath: modelPath, isDirectory: true), preloadMTP: true)
        let imageURLs = image.map { [URL(fileURLWithPath: $0, isDirectory: false)] } ?? []
        let result = try await runtime.compareLocalMTPWithUpstream(
            prompt: prompt,
            imageURLs: imageURLs,
            options: Qwen38GenerationOptions(
                maxTokens: maxTokens,
                temperature: 0,
                enableThinking: true,
                reasoningEffort: "low",
                kvBits: nil,
                mtp: .init(enabled: true, draftDepth: .fixed(max(blockSize - 1, 1)))),
            blockSize: blockSize)

        print("M2 local : \(result.localTokenIDs.count) tokens")
        print("M1 upstream : \(result.upstreamTokenIDs.count) tokens")
        let parityLabel = result.isIdentical ? "IDENTIQUE" : "DIFFÉRENTE"
        print("parité IDs : \(parityLabel)")
        if let firstDifference = result.firstDifference {
            print("première différence à l'index : \(firstDifference)")
        }
        print("M2 : \(await runtime.decode(tokenIDs: result.localTokenIDs))")
        print("M1 : \(await runtime.decode(tokenIDs: result.upstreamTokenIDs))")
    }
}

struct MTPConversationProbe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mtp-conversation-probe",
        abstract: "Vérifier la réutilisation M2 sur deux tours")

    @Option(name: .long, help: "Répertoire local du modèle")
    var modelPath: String

    @Option(name: .long, help: "Prompt du premier tour")
    var prompt: String

    @Option(name: .long, help: "Prompt du second tour")
    var secondPrompt: String = "Il a fait quoi pour arriver là ?"

    @Option(name: .long, help: "Prompt optionnel du troisième tour")
    var thirdPrompt: String?

    @Option(name: .long, help: "Chemin d'une image pour le premier tour")
    var image: String?

    @Option(name: .long, help: "Nombre maximum de tokens par tour")
    var maxTokens: Int = 16

    @Option(name: .long, help: "Largeur totale verify = bonus + drafts")
    var blockSize: Int = 3

    func run() async throws {
        let runtime = Qwen38Runtime()
        try await runtime.load(
            from: URL(fileURLWithPath: modelPath, isDirectory: true), preloadMTP: true)
        let imageURLs = image.map { [URL(fileURLWithPath: $0, isDirectory: false)] } ?? []
        let results = try await runtime.runLocalMTPConversation(
            prompt: prompt,
            secondPrompt: secondPrompt,
            thirdPrompt: thirdPrompt,
            imageURLs: imageURLs,
            options: Qwen38GenerationOptions(
                maxTokens: maxTokens,
                temperature: 0,
                enableThinking: true,
                reasoningEffort: "low",
                kvBits: nil,
                mtp: .init(enabled: true, draftDepth: .fixed(max(blockSize - 1, 1)))),
            blockSize: blockSize)
        for (index, result) in results.enumerated() {
            print("Tour \(index + 1) : \(await runtime.decode(tokenIDs: result.tokenIDs))")
            print(
                "  rounds=\(result.stats.rounds) proposés=\(result.stats.proposedTokens) "
                    + "acceptés=\(result.stats.acceptedTokens)")
        }
    }
}

struct MTPProbe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mtp-probe",
        abstract: "Exécuter la première boucle MTP locale M2 sur un checkpoint")

    @Option(name: .long, help: "Répertoire local du modèle")
    var modelPath: String

    @Option(name: .long, help: "Prompt utilisateur")
    var prompt: String

    @Option(name: .long, help: "Chemin d'une image à joindre au prompt")
    var image: String?

    @Option(name: .long, help: "Nombre maximum de tokens")
    var maxTokens: Int = 32

    @Option(name: .long, help: "Largeur totale verify = bonus + drafts")
    var blockSize: Int = 3

    @Flag(name: .long, help: "Comparer aussi au chemin cible standard avec les mêmes paramètres")
    var compareStandard = false

    func run() async throws {
        let runtime = Qwen38Runtime()
        try await runtime.load(
            from: URL(fileURLWithPath: modelPath, isDirectory: true), preloadMTP: true)
        let imageURLs = image.map { [URL(fileURLWithPath: $0, isDirectory: false)] } ?? []
        let result = try await runtime.runLocalMTP(
            prompt: prompt,
            imageURLs: imageURLs,
            options: Qwen38GenerationOptions(
                maxTokens: maxTokens,
                temperature: 0,
                enableThinking: true,
                reasoningEffort: "low",
                kvBits: nil,
                mtp: Qwen38MTPOptions(enabled: true, draftDepth: .fixed(blockSize - 1))),
            blockSize: blockSize)
        print(await runtime.decode(tokenIDs: result.tokenIDs))
        if compareStandard {
            let baselineOptions = Qwen38GenerationOptions(
                maxTokens: maxTokens,
                temperature: 0,
                enableThinking: true,
                reasoningEffort: "low",
                kvBits: nil,
                mtp: .init(enabled: false))
            let baselineStream = try await runtime.generate(
                prompt: prompt,
                imageURLs: imageURLs,
                options: baselineOptions,
                forceConversationReplay: true)
            var baselineText = ""
            for try await event in baselineStream {
                if case .chunk(let chunk) = event {
                    baselineText += chunk
                }
            }
            let localText = await runtime.decode(tokenIDs: result.tokenIDs)
            print("\n--- Comparaison cible standard ---")
            print(baselineText)
            let parityLabel = baselineText == localText ? "IDENTIQUE" : "DIFFÉRENTE"
            print("parité texte : \(parityLabel)")
        }
        print("\n--- M2 ---")
        print("token IDs: \(result.tokenIDs.map(String.init).joined(separator: ","))")
        print("tokens: \(result.tokenIDs.count)")
        print("rounds: \(result.stats.rounds)")
        print("proposés: \(result.stats.proposedTokens)")
        print("acceptés: \(result.stats.acceptedTokens)")
        print("restaurations GDN: \(result.stats.gdnRestores)")
        if let rate = result.stats.acceptanceRate {
            print(String(format: "accept rate: %.2f%%", rate * 100))
        }
    }
}

struct Info: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Inspecter un modèle local")

    @Argument(help: "Répertoire local du modèle")
    var modelPath: String

    func run() async throws {
        let directory = URL(fileURLWithPath: modelPath, isDirectory: true)
        let configURL = directory.appendingPathComponent("config.json")
        if let data = try? Data(contentsOf: configURL),
           let raw = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           raw["model_type"] as? String == "qwen4_exp"
        {
            let config = try Qwen4ExpConfiguration.load(from: directory)
            let preflight = try Qwen4ExpCheckpointPreflight.validate(directory)
            let size = Qwen38ModelCache.diskSize(of: directory)
            let text = config.textConfiguration
            let mtpLayers = ((raw["text_config"] as? [String: Any])?["mtp"] as? [String: Any])?["num_hidden_layers"] as? Int ?? 0
            print("model_type: \(config.modelType)")
            print("architecture: \(config.architectures.first ?? "unknown")")
            print("hidden_size: \(text.hiddenSize)")
            print("layers: \(text.numHiddenLayers)")
            print("layer_types: \(text.layerTypes.map(\.rawValue).joined(separator: ","))")
            print("mtp_layers: \(mtpLayers)")
            print("tensors: \(preflight.tensorCount)")
            print("shards: \(preflight.shardCount)")
            print("quantized_weights: \(preflight.quantizedWeightCount)")
            print("weight_bytes: \(ByteCountFormatter.string(fromByteCount: preflight.weightBytes, countStyle: .file))")
            print("disk: \(ByteCountFormatter.string(fromByteCount: size, countStyle: .file))")
            print("status: configuration/index valides, runtime qwen4_exp expérimental")
            print("norms: correction automatique Vontra (+1 -> zéro-centré) au chargement")
            return
        }
        let info = try Qwen38ModelValidator.validate(directory)
        let size = Qwen38ModelCache.diskSize(of: directory)
        print("model_type: \(info.modelType)")
        print("architecture: \(info.architecture ?? "unknown")")
        print("hidden_size: \(info.hiddenSize.map(String.init) ?? "unknown")")
        print("layers: \(info.numHiddenLayers.map(String.init) ?? "unknown")")
        print("disk: \(ByteCountFormatter.string(fromByteCount: size, countStyle: .file))")
    }
}

struct Generate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Générer une réponse en streaming")

    @Option(name: .long, help: "Répertoire local du modèle")
    var modelPath: String

    @Option(name: .long, help: "Prompt utilisateur")
    var prompt: String

    @Option(name: .long, help: "Chemin d'une image à joindre au prompt")
    var image: String?

    @Option(name: .long, help: "Nombre maximum de tokens")
    var maxTokens: Int = 256

    @Option(name: .long, help: "Température, 0 = greedy")
    var temperature: Float = 0

    @Flag(name: .long, help: "Activer le drafter MTP (greedy uniquement)")
    var mtp = false

    @Option(name: .long, help: "Moteur MTP : upstream ou local")
    var mtpEngine: String = "upstream"

    @Option(name: .long, help: "Nombre de tokens draftés par round (M1 plafonné à 1, M2 jusqu'à 8)")
    var mtpDraftTokens: Int = 1

    @Option(name: .long, help: "Chemin de sortie Chrome Trace")
    var trace: String?

    func run() async throws {
        let runtime = Qwen38Runtime()
        let directory = URL(fileURLWithPath: modelPath, isDirectory: true)
        print("Chargement de \(directory.path)…", terminator: "\n")
        try await runtime.load(from: directory, preloadMTP: mtp)
        switch await runtime.mtpState {
        case .active:
            print("Drafter MTP local : prêt")
        case .fallback(let reason):
            print("Drafter MTP local : fallback — " + reason)
        case .unavailable:
            print("Drafter MTP local : absent")
        }

        let options = Qwen38GenerationOptions(
            maxTokens: maxTokens,
            temperature: temperature,
            enableThinking: true,
            reasoningEffort: "low",
            mtp: Qwen38MTPOptions(
                enabled: mtp,
                draftDepth: .fixed(mtpDraftTokens),
                engine: try Self.mtpEngineValue(mtpEngine)
            )
        )
        let imageURLs = image.map { [URL(fileURLWithPath: $0, isDirectory: false)] } ?? []
        let stream = try await runtime.generate(
            prompt: prompt,
            imageURLs: imageURLs,
            options: options
        )
        for try await event in stream {
            switch event {
            case .chunk(let text):
                print(text, terminator: "")
                fflush(stdout)
            case .metrics(let run):
                print("\n\n--- Metrics ---")
                print(run.metrics.compactSummary)
                if let ttft = run.timeToFirstToken {
                    print("TTFT réel: \(String(format: "%.1f", ttft * 1000)) ms")
                }
                switch run.mtpStatus.availability {
                case .active:
                    print("MTP: actif · proposés " + String(run.mtpStatus.proposedTokens)
                        + " · acceptés " + String(run.mtpStatus.acceptedTokens))
                case .fallback(let reason):
                    print("MTP: fallback — " + reason)
                case .unavailable:
                    break
                }
                print("Prefill modèle: \(String(format: "%.1f", run.metrics.prefillTime * 1000)) ms")
                print("Stop: \(run.stopReason)")
                if let trace {
                    try run.chromeTrace.write(to: URL(fileURLWithPath: trace), options: .atomic)
                    print("Trace: \(trace)")
                }
            }
        }
    }

    private static func mtpEngineValue(_ rawValue: String) throws -> Qwen38MTPEngine {
        guard let engine = Qwen38MTPEngine(rawValue: rawValue.lowercased()) else {
            throw ValidationError("--mtp-engine doit valoir upstream ou local")
        }
        return engine
    }
}

struct ConversationBenchmark: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "conversation-benchmark",
        abstract: "Mesurer trois tours consécutifs avec le même contexte"
    )

    @Option(name: .long, help: "Répertoire local du modèle")
    var modelPath: String

    @Option(name: .long, help: "Image jointe au premier tour")
    var image: String

    @Flag(name: .long, help: "Activer le drafter MTP (greedy uniquement)")
    var mtp = false

    @Option(name: .long, help: "Moteur MTP : upstream ou local")
    var mtpEngine: String = "upstream"

    @Option(name: .long, help: "Nombre de tokens draftés par round (1 à 8)")
    var mtpDraftTokens: Int = 1

    @Option(name: .long, help: "Nombre maximum de tokens par tour")
    var maxTokens: Int = 2048

    @Option(name: .long, help: "Chemin d'un rapport TSV")
    var report: String?

    private static let prompts = [
        "c'est qui sur cette photo ?",
        "Il a fait quoi pour arriver là ?",
        "Il sera à nouveau président à la prochaine élection dans son pays ?"
    ]

    func run() async throws {
        let runtime = Qwen38Runtime()
        let modelURL = URL(fileURLWithPath: modelPath, isDirectory: true)
        let imageURL = URL(fileURLWithPath: image, isDirectory: false)
        guard FileManager.default.fileExists(atPath: imageURL.path) else {
            throw ValidationError("Image introuvable : \(imageURL.path)")
        }

        print("Modèle : \(modelURL.lastPathComponent)")
        guard (1...8).contains(mtpDraftTokens) else {
            throw ValidationError("--mtp-draft-tokens doit être compris entre 1 et 8")
        }
        let selectedMTPEngine = try Self.mtpEngineValue(mtpEngine)
        print(
            "Thinking : low · max tokens : \(maxTokens) · MTP : "
                + "\(mtp ? "on · \(selectedMTPEngine.rawValue) · draft \(mtpDraftTokens)" : "off")"
        )
        try await runtime.load(from: modelURL, preloadMTP: mtp)
        switch await runtime.mtpState {
        case .active: print("Drafter MTP : prêt")
        case .fallback(let reason): print("Drafter MTP : fallback — \(reason)")
        case .unavailable: print("Drafter MTP : absent")
        }

        var rows = [
            "model\tmtp_requested\tmtp_engine\tmtp_draft_tokens\tturn\tinput\tprompt_tokens\tgenerated_tokens\tttft_ms\tprefill_tok_s\tdecode_tok_s\tactive_memory_mb\tpeak_memory_mb\tcache_reused\tconversation_replayed\tmtp_availability\tmtp_block_size\tmtp_proposed\tmtp_accepted\tmtp_rounds\tmtp_accept_rate\tresponse_file\ttrace_file"
        ]
        for (index, prompt) in Self.prompts.enumerated() {
            let options = Qwen38GenerationOptions(
                maxTokens: maxTokens,
                temperature: 0,
                enableThinking: true,
                reasoningEffort: "low",
                kvBits: nil,
                mtp: Qwen38MTPOptions(
                    enabled: mtp,
                    draftDepth: .fixed(mtpDraftTokens),
                    engine: selectedMTPEngine
                )
            )
            let stream = try await runtime.generate(
                prompt: prompt,
                imageURLs: index == 0 ? [imageURL] : [],
                options: options
            )
            var run: Qwen38RunMetrics?
            var visibleParser = Qwen38ThinkingStreamParser()
            var responseText = ""
            for try await event in stream {
                switch event {
                case .chunk(let text):
                    responseText += visibleParser.append(text).content
                case .metrics(let metrics):
                    run = metrics
                }
            }
            responseText += visibleParser.finish().content
            guard let run else { throw Qwen38RuntimeError.missingCompletionInfo }

            let responseFile: String
            let traceFile: String
            if let report {
                let reportURL = URL(fileURLWithPath: report)
                let baseURL = reportURL.deletingPathExtension()
                let responseURL = baseURL.appendingPathExtension("turn\(index + 1).txt")
                let traceURL = baseURL.appendingPathExtension("turn\(index + 1).trace.json")
                try FileManager.default.createDirectory(
                    at: reportURL.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )
                try Data(responseText.utf8).write(to: responseURL, options: .atomic)
                try run.chromeTrace.write(to: traceURL, options: .atomic)
                responseFile = responseURL.path
                traceFile = traceURL.path
            } else {
                responseFile = "-"
                traceFile = "-"
            }

            let ttft = run.timeToFirstToken.map { String(format: "%.1f", $0 * 1000) } ?? "-"
            let prefill = run.metrics.promptTokens > 0
                ? String(format: "%.2f", Double(run.metrics.promptTokens) / run.metrics.prefillTime)
                : "-"
            let decode = run.metrics.generatedTokens > 0
                ? String(format: "%.2f", Double(run.metrics.generatedTokens) / run.metrics.generationTime)
                : "-"
            let activeMemory = String(format: "%.1f", Double(run.activeMemoryBytes) / 1_048_576)
            let peakMemory = String(format: "%.1f", Double(run.peakMemoryBytes) / 1_048_576)
            let mtpAvailability: String
            switch run.mtpStatus.availability {
            case .active: mtpAvailability = "active"
            case .fallback: mtpAvailability = "fallback"
            case .unavailable: mtpAvailability = "unavailable"
            }
            let mtpRate = run.mtpStatus.proposedTokens > 0
                ? String(format: "%.4f", run.mtpStatus.acceptanceRate ?? 0)
                : "-"
            print(
                "Tour \(index + 1) · \(run.inputDescription) · "
                    + "prompt \(run.metrics.promptTokens) · générés \(run.metrics.generatedTokens) · "
                    + "TTFT \(ttft) ms · prefill \(prefill) tok/s · decode \(decode) tok/s · "
                    + "pic MLX \(peakMemory) MiB · "
                    + "cache \(run.cacheReused ? "réutilisé" : (run.conversationReplayed ? "historique rejoué" : "initialisé")) · "
                    + "MTP \(mtpAvailability) \(run.mtpStatus.proposedTokens)/\(run.mtpStatus.acceptedTokens)"
            )
            rows.append([
                modelURL.lastPathComponent,
                mtp ? "1" : "0",
                mtp ? selectedMTPEngine.rawValue : "none",
                mtp ? String(mtpDraftTokens) : "0",
                String(index + 1),
                run.inputDescription,
                String(run.metrics.promptTokens),
                String(run.metrics.generatedTokens),
                ttft,
                prefill,
                decode,
                activeMemory,
                peakMemory,
                run.cacheReused ? "1" : "0",
                run.conversationReplayed ? "1" : "0",
                mtpAvailability,
                String(run.mtpStatus.blockSize ?? 0),
                String(run.mtpStatus.proposedTokens),
                String(run.mtpStatus.acceptedTokens),
                String(run.mtpStatus.rounds),
                mtpRate,
                responseFile,
                traceFile
            ].joined(separator: "\t"))
        }
        if let report {
            try Data(rows.joined(separator: "\n").appending("\n").utf8)
                .write(to: URL(fileURLWithPath: report), options: .atomic)
            print("Rapport : \(report)")
        }
    }

    private static func mtpEngineValue(_ rawValue: String) throws -> Qwen38MTPEngine {
        guard let engine = Qwen38MTPEngine(rawValue: rawValue.lowercased()) else {
            throw ValidationError("--mtp-engine doit valoir upstream ou local")
        }
        return engine
    }
}

struct ConversationParity: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "conversation-parity",
        abstract: "Comparer les sorties greedy standard et MTP sur trois tours"
    )

    @Option(name: .long, help: "Répertoire local du modèle")
    var modelPath: String

    @Option(name: .long, help: "Image jointe au premier tour")
    var image: String

    @Option(name: .long, help: "Nombre maximum de tokens par tour")
    var maxTokens: Int = 512

    @Flag(name: .long, help: "Retourner une erreur si une sortie diffère")
    var strict = false

    private struct TurnResult {
        let text: String
        let metrics: Qwen38RunMetrics
    }

    private static let prompts = [
        "c'est qui sur cette photo ?",
        "Il a fait quoi pour arriver là ?",
        "Il sera à nouveau président à la prochaine élection dans son pays ?"
    ]

    func run() async throws {
        let modelURL = URL(fileURLWithPath: modelPath, isDirectory: true)
        let imageURL = URL(fileURLWithPath: image, isDirectory: false)
        guard FileManager.default.fileExists(atPath: imageURL.path) else {
            throw ValidationError("Image introuvable : \(imageURL.path)")
        }

        print("Parité greedy · \(modelURL.lastPathComponent) · max tokens : \(maxTokens)")
        let standard = try await Self.runConversation(
            modelURL: modelURL, imageURL: imageURL, maxTokens: maxTokens, mtp: false)
        let speculative = try await Self.runConversation(
            modelURL: modelURL, imageURL: imageURL, maxTokens: maxTokens, mtp: true)

        var allEqual = true
        for index in Self.prompts.indices {
            let lhs = standard[index].text
            let rhs = speculative[index].text
            let equal = lhs == rhs
            allEqual = allEqual && equal
            let firstDifference = equal ? "—" : Self.firstDifference(lhs, rhs)
            let mtp = speculative[index].metrics.mtpStatus
            let mtpLabel: String
            switch mtp.availability {
            case .active:
                mtpLabel = "actif"
            case .unavailable:
                mtpLabel = "indisponible"
            case .fallback(let reason):
                mtpLabel = "fallback: \(reason)"
            }
            print(
                "Tour \(index + 1) · \(equal ? "IDENTIQUE" : "DIFFÉRENTE") · "
                    + "standard \(lhs.utf8.count) octets · MTP \(rhs.utf8.count) octets · "
                    + "première différence \(firstDifference) · "
                    + "MTP \(mtpLabel) · accept \(mtp.acceptedTokens)/\(mtp.proposedTokens)"
            )
        }

        if strict && !allEqual {
            throw ValidationError("La parité greedy standard/MTP a échoué.")
        }
        print(allEqual ? "Résultat : parité validée" : "Résultat : divergence à investiguer")
    }

    private static func runConversation(
        modelURL: URL,
        imageURL: URL,
        maxTokens: Int,
        mtp: Bool
    ) async throws -> [TurnResult] {
        let runtime = Qwen38Runtime()
        try await runtime.load(from: modelURL, preloadMTP: mtp)
        var results: [TurnResult] = []
        for (index, prompt) in prompts.enumerated() {
            let options = Qwen38GenerationOptions(
                maxTokens: maxTokens,
                temperature: 0,
                enableThinking: true,
                reasoningEffort: "low",
                kvBits: nil,
                mtp: .init(enabled: mtp, draftDepth: .fixed(1))
            )
            let stream = try await runtime.generate(
                prompt: prompt,
                imageURLs: index == 0 ? [imageURL] : [],
                options: options,
                forceConversationReplay: true
            )
            var text = ""
            var metrics: Qwen38RunMetrics?
            for try await event in stream {
                switch event {
                case .chunk(let chunk): text += chunk
                case .metrics(let value): metrics = value
                }
            }
            guard let metrics else { throw Qwen38RuntimeError.missingCompletionInfo }
            results.append(TurnResult(text: text, metrics: metrics))
        }
        await runtime.unload()
        return results
    }

    private static func firstDifference(_ lhs: String, _ rhs: String) -> String {
        let left = Array(lhs.utf8)
        let right = Array(rhs.utf8)
        let limit = min(left.count, right.count)
        for index in 0..<limit where left[index] != right[index] {
            return "offset \(index)"
        }
        return "offset \(limit)"
    }
}

struct Download: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Télécharger un modèle Hugging Face")

    @Argument(help: "Identifiant Hugging Face, par exemple mlx-community/Qwen3.8-27B-4bit")
    var modelID: String

    @Option(name: .long, help: "Répertoire racine local (Lexar par défaut)")
    var modelsDir: String?

    @Option(name: .long, help: "Token Hugging Face ; HF_TOKEN est utilisé par défaut")
    var token: String?

    func run() async throws {
        if let modelsDir {
            Qwen38ModelCache.customModelsDirectory = URL(fileURLWithPath: modelsDir, isDirectory: true)
        }
        let manager = Qwen38DownloadManager()
        let effectiveToken = token ?? ProcessInfo.processInfo.environment["HF_TOKEN"]
        let destination = Qwen38ModelCache.modelsDirectory
        print("Destination : \(destination.path)")
        let directory = try await manager.download(
            modelID: modelID,
            to: destination,
            token: effectiveToken
        ) { progress in
            let fraction = progress.fractionCompleted.map { String(format: "%.1f%%", $0 * 100) } ?? "?"
            let speed = ByteCountFormatter.string(
                fromByteCount: Int64(progress.bytesPerSecond), countStyle: .file
            )
            print("[\(fraction)] \(progress.file.path) — \(speed)/s", terminator: "\r")
            fflush(stdout)
        }
        print("\nModèle prêt : \(directory.path)")
    }
}
