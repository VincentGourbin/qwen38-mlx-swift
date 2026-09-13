import Foundation

/// P11.2/P11.4a : décodage à séquence forcée sur le checkpoint réel.
///
/// P11.2 (`docs/knowledge/log.md`, 2026-09-13, "l'ablation par soustraction
/// ne mesure pas ce qu'on croit") a montré que l'attribution du coût par
/// sous-bloc par ablation-et-soustraction sur une génération *libre* est
/// invalide : ablater un sous-bloc change les tokens produits, donc les
/// lectures n-gram et le routage MoE — on ne mesure plus le sous-bloc, on
/// mesure la dégénérescence de sortie qu'il provoque. Le remède est de
/// décoder une séquence de tokens **forcée**, identique pour toutes les
/// variantes, de sorte que seule la branche ablatée diffère.
///
/// P11.4a (`docs/knowledge/log.md`, 2026-09-12 nuit, "le plafond du
/// décodage spéculatif") a besoin, séparément, du coût d'un forward selon
/// le nombre de tokens qu'il traite — un chiffre que le TTFT du serveur ne
/// peut pas isoler (il porte ~250 ms de coût fixe de requête).
///
/// `flash-decode-bench` (Qwen38CLI) répond aux deux besoins avec un seul
/// instrument. Ce fichier isole toute la logique qui ne dépend pas d'un
/// forward MLX réel — découpage de la séquence forcée en pas de
/// `--tokens-per-step`, lecture du CSV `--ablate-sweep`, ordre d'alternance
/// des tours, statistiques descriptives — pour qu'elle reste testable sans
/// checkpoint (`Qwen38Tests`, cible qui ne dépend que de `Qwen38Core`/
/// `Qwen38Server`, jamais de l'exécutable CLI).
public enum Qwen4ExpDecodeBenchError: LocalizedError, Equatable {
    case invalidTokensPerStep(Int)
    case invalidStepCount(Int)
    case insufficientForcedIDs(required: Int, provided: Int)
    case emptyAblationSweep
    case emptyTokensPerStepSweep
    case invalidTokensPerStepSweepEntry(String)

    public var errorDescription: String? {
        switch self {
        case .invalidTokensPerStep(let value):
            return "--tokens-per-step doit être positif ; reçu \(value)."
        case .invalidStepCount(let value):
            return
                "le nombre total de pas (warmup + steps) doit être positif ; reçu \(value)."
        case .insufficientForcedIDs(let required, let provided):
            return
                "la séquence forcée fournit \(provided) identifiant(s) ; il en faut au moins "
                + "\(required) (warmup + steps, multiplié par tokens-per-step)."
        case .emptyAblationSweep:
            return "--ablate-sweep ne peut pas être vide ou ne contenir que des espaces."
        case .emptyTokensPerStepSweep:
            return "--tokens-per-step-sweep ne peut pas être vide ou ne contenir que des espaces."
        case .invalidTokensPerStepSweepEntry(let raw):
            return
                "--tokens-per-step-sweep : « \(raw) » n'est pas un entier positif "
                + "(valeurs attendues : 1,2,4,8…)."
        }
    }
}

/// Découpe la séquence forcée en `stepCount` pas de `tokensPerStep` jetons
/// chacun, dans l'ordre d'origine : le pas `i` (0-indexé) consomme
/// `forcedIDs[i*tokensPerStep ..< (i+1)*tokensPerStep]`. C'est la même
/// séquence de jetons pour tous les pas — seul le nombre qu'on en prend à
/// la fois change avec `tokensPerStep` (point (B) de P11.4a : le coût d'un
/// forward selon N).
///
/// Des jetons au-delà de `stepCount * tokensPerStep` (un `--forced-ids` plus
/// long que nécessaire) sont silencieusement ignorés — le caller peut
/// réutiliser une même séquence forcée pour plusieurs tailles de banc. Une
/// séquence trop courte, en revanche, est une erreur explicite : tronquer
/// silencieusement changerait `stepCount` sans le dire, ce qui est
/// exactement le genre de biais que cet instrument existe pour éliminer.
public func qwen4ExpSplitForcedDecodeSteps(
    forcedIDs: [Int32], tokensPerStep: Int, stepCount: Int
) throws -> [[Int32]] {
    guard tokensPerStep > 0 else {
        throw Qwen4ExpDecodeBenchError.invalidTokensPerStep(tokensPerStep)
    }
    guard stepCount > 0 else {
        throw Qwen4ExpDecodeBenchError.invalidStepCount(stepCount)
    }
    let required = stepCount * tokensPerStep
    guard forcedIDs.count >= required else {
        throw Qwen4ExpDecodeBenchError.insufficientForcedIDs(
            required: required, provided: forcedIDs.count)
    }
    return (0..<stepCount).map { step in
        let start = step * tokensPerStep
        return Array(forcedIDs[start..<(start + tokensPerStep)])
    }
}

/// Résout `--ablate-sweep` (une liste d'ablations séparées par des
/// virgules, même vocabulaire que `--ablate` — voir
/// `qwen4ExpResolveAblation`) en une liste non vide de variantes, dans
/// l'ordre où elles ont été données. Cet ordre fixe l'alternance des tours :
/// `qwen4ExpDecodeBenchSchedule` visite les variantes dans exactement cet
/// ordre à chaque tour. Les espaces autour de chaque élément sont ignorés ;
/// un élément vide (ex. virgule finale) est une erreur explicite, comme
/// `qwen4ExpResolveAblation("")`. Les doublons sont acceptés tels quels —
/// répéter une variante dans la liste répète juste sa visite à chaque tour,
/// ce qui reste une alternance valide, seulement redondante.
/// P11.4a : découpe `"1,2,4,8"` en valeurs de `--tokens-per-step`. Rejette
/// les entrées vides, non entières ou nulles. Pendant de
/// `qwen4ExpParseAblationSweep` pour l'autre dimension balayée : l'ordre est
/// préservé, c'est lui qui fixe l'alternance des tours.
public func qwen4ExpParseTokensPerStepSweep(_ raw: String) throws -> [Int] {
    let parts = raw.split(separator: ",").map {
        $0.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    // Une chaîne vide ou faite d'espaces donne `[]` ou `[""]` selon qu'elle
    // contient une virgule : les deux valent « balayage vide », pas « entrée
    // invalide ».
    guard !parts.isEmpty, parts.contains(where: { !$0.isEmpty }) else {
        throw Qwen4ExpDecodeBenchError.emptyTokensPerStepSweep
    }
    var out: [Int] = []
    for part in parts {
        guard let value = Int(part), value > 0 else {
            throw Qwen4ExpDecodeBenchError.invalidTokensPerStepSweepEntry(part)
        }
        out.append(value)
    }
    return out
}

public func qwen4ExpParseAblationSweep(_ raw: String) throws -> [Qwen4ExpLayerBenchAblation] {
    let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else {
        throw Qwen4ExpDecodeBenchError.emptyAblationSweep
    }
    return try trimmed.split(separator: ",", omittingEmptySubsequences: false).map { part in
        try qwen4ExpResolveAblation(rawValue: part.trimmingCharacters(in: .whitespacesAndNewlines))
    }
}

/// Une visite du calendrier round-major : au tour `round`, décoder le pas
/// forcé `round` sous l'ablation `variants[variantIndex]` (le caller indexe
/// sa propre liste de variantes avec `variantIndex`).
public struct Qwen4ExpDecodeBenchVisit: Equatable, Sendable {
    public let round: Int
    public let variantIndex: Int

    public init(round: Int, variantIndex: Int) {
        self.round = round
        self.variantIndex = variantIndex
    }
}

/// Énumère les visites d'un balayage `--ablate-sweep` **round-major** :
/// tour 0 visite chaque variante une fois (dans l'ordre de
/// `qwen4ExpParseAblationSweep`), puis tour 1 revisite chaque variante, etc.
/// C'est la méthodologie qui a fonctionné en P11.1 (alterner à chaque tour
/// plutôt que dérouler une variante en entier avant de passer à la
/// suivante) : elle répartit la dérive thermique/d'ordonnancement du run
/// également entre les variantes au lieu de la concentrer sur celle mesurée
/// en dernier — voir `docs/knowledge/log.md`, "P11.1" (écart-type 0,05 tok/s
/// avec ce protocole contre plusieurs tok/s en process neuf par variante).
///
/// `variantCount` ou `roundCount` non positif renvoie un calendrier vide :
/// il n'y a rien à alterner sans au moins une variante et un tour, et les
/// deux callers de cette fonction (`qwen4ExpParseAblationSweep`,
/// `qwen4ExpSplitForcedDecodeSteps`) ont déjà rejeté ces cas plus tôt dans
/// leur propre validation — pas la peine de dupliquer une deuxième erreur
/// ici pour une situation que l'appelant ne peut plus atteindre.
public func qwen4ExpDecodeBenchSchedule(variantCount: Int, roundCount: Int)
    -> [Qwen4ExpDecodeBenchVisit]
{
    guard variantCount > 0, roundCount > 0 else { return [] }
    var visits: [Qwen4ExpDecodeBenchVisit] = []
    visits.reserveCapacity(variantCount * roundCount)
    for round in 0..<roundCount {
        for variantIndex in 0..<variantCount {
            visits.append(Qwen4ExpDecodeBenchVisit(round: round, variantIndex: variantIndex))
        }
    }
    return visits
}

/// Statistiques descriptives d'une série de millisecondes par pas de
/// décodage — médiane, moyenne, écart-type (population, pas échantillon :
/// on décrit la série mesurée elle-même, on n'extrapole pas à une
/// population plus large), min et max, plus le nombre de pas retenus.
public struct Qwen4ExpDecodeBenchStats: Equatable {
    public let count: Int
    public let medianMs: Double
    public let meanMs: Double
    public let stddevMs: Double
    public let minMs: Double
    public let maxMs: Double

    public init(count: Int, medianMs: Double, meanMs: Double, stddevMs: Double, minMs: Double, maxMs: Double) {
        self.count = count
        self.medianMs = medianMs
        self.meanMs = meanMs
        self.stddevMs = stddevMs
        self.minMs = minMs
        self.maxMs = maxMs
    }
}

/// Calcule `Qwen4ExpDecodeBenchStats` sur une série de millisecondes par
/// pas. Une série vide renvoie des `.nan` partout avec `count == 0` plutôt
/// que de crasher — le caller (`flash-decode-bench`) l'utilise pour
/// détecter et signaler un `--warmup` qui a consommé tous les pas.
public func qwen4ExpDecodeBenchStats(millisecondsPerStep values: [Double]) -> Qwen4ExpDecodeBenchStats {
    guard !values.isEmpty else {
        return Qwen4ExpDecodeBenchStats(
            count: 0, medianMs: .nan, meanMs: .nan, stddevMs: .nan, minMs: .nan, maxMs: .nan)
    }
    let sorted = values.sorted()
    let mid = sorted.count / 2
    let median = sorted.count % 2 == 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid]
    let mean = values.reduce(0, +) / Double(values.count)
    let variance = values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(values.count)
    return Qwen4ExpDecodeBenchStats(
        count: values.count, medianMs: median, meanMs: mean, stddevMs: variance.squareRoot(),
        minMs: sorted.first!, maxMs: sorted.last!)
}
