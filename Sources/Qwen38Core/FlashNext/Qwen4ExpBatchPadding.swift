import Foundation
import MLX

/// P12.2 (voir `PLAN.md` « P12 » et `docs/knowledge/log.md` 2026-09-13,
/// « P12.1 : le lot passe la porte ») : lots à longueurs de prompt inégales.
///
/// P12.1 a validé le décodage par lots à longueur égale sans aucune
/// modification de production. Trois blocages restaient pour des longueurs
/// mélangées : l'horloge de position scalaire pour tout le lot, l'absence de
/// masque de remplissage (GDN comme QSA), et l'outillage vendor
/// (`ArraysCache`/`MambaCache`) jamais branché. Ce fichier porte la logique
/// pure — testable sans checkpoint — qui résout les deux premiers ; le
/// troisième s'est avéré ne rien apporter ici : `ArraysCache.leftPadding`
/// sert au *découpage/fusion* de lots en cours de route (`filter`/`extend`),
/// hors du périmètre de cette étape (« ne fais pas d'ordonnanceur »). Le nom
/// est repris tel quel ci-dessous parce qu'il désigne exactement la même
/// convention de remplissage.
///
/// **Remplissage à gauche, choisi plutôt qu'à droite.** Avec un remplissage
/// à gauche, la dernière colonne du lot est, pour CHAQUE ligne, son dernier
/// jeton réel : le pas de décodage relit `logits[:, -1, :]` exactement comme
/// dans le cas à longueur égale de P12.1, sans qu'aucun appelant n'ait à
/// suivre, ligne par ligne, l'index où lire son prochain jeton. Le
/// remplissage à droite aurait exigé un rassemblement (`takeAlong`) par ligne
/// à chaque pas pour la même information, pour un gain nul : les
/// `positionIDs` explicites ci-dessous corrigent déjà le décalage de
/// position que le remplissage à gauche est censé introduire, donc son seul
/// inconvénient théorique disparaît. C'est aussi la convention déjà nommée
/// côté vendor : `ArraysCache.leftPadding` / `MambaCache` et
/// `ArraysCache.makeMask` (`Vendor/mlx-swift-lm/Libraries/MLXLMCommon/
/// KVCache.swift`) sont construits pour ce sens de remplissage, pas l'autre.
///
/// **Jeton de remplissage : l'EOS du checkpoint, pas un jeton arbitraire.**
/// `Qwen4ExpNGramEmbedding.callAsFunction` initialise l'historique de
/// n-grammes en l'absence de tout contexte avec `contextLength` jetons EOS
/// (`Qwen4ExpPLE.swift`), et `shiftRightIgnoringEOS` traite déjà tout EOS
/// comme une frontière de segment dure : au-delà d'un EOS, l'historique lu
/// est le jeton EOS lui-même, jamais un jeton antérieur. Remplir à gauche
/// avec EOS place donc chaque ligne dans EXACTEMENT la même situation que si
/// elle démarrait seule, quelle que soit la longueur de son remplissage —
/// aucun masque supplémentaire n'est nécessaire pour la construction des
/// identifiants de n-gramme elle-même. Un autre jeton de remplissage aurait
/// fait lire aux premiers jetons réels un historique de n-gramme différent
/// de leur exécution seule, et cassé la parité pour toute ligne dont le
/// remplissage est plus court que la fenêtre de contexte.
public enum Qwen4ExpBatchPaddingError: LocalizedError, Equatable {
    case emptyBatch

    public var errorDescription: String? {
        switch self {
        case .emptyBatch:
            return "Le lot ne contient aucune séquence."
        }
    }
}

/// Longueur commune (après remplissage) et décalage à gauche de chaque
/// ligne du lot. Ne construit aucun `MLXArray` — pure logique de
/// dimensionnement, testable sans device Metal.
public struct Qwen4ExpBatchPaddingLayout: Equatable, Sendable {
    public let maxLength: Int
    public let leftPadding: [Int]

    public init(maxLength: Int, leftPadding: [Int]) {
        precondition(maxLength >= 0 && leftPadding.allSatisfy { $0 >= 0 && $0 <= maxLength })
        self.maxLength = maxLength
        self.leftPadding = leftPadding
    }

    public var batchSize: Int { leftPadding.count }
}

/// Calcule le remplissage à gauche nécessaire pour aligner `tokenCounts`
/// (une longueur de prompt réelle par ligne, avant tout remplissage) sur la
/// plus longue.
public func qwen4ExpComputeBatchPaddingLayout(
    tokenCounts: [Int]
) throws -> Qwen4ExpBatchPaddingLayout {
    guard let maxLength = tokenCounts.max() else {
        throw Qwen4ExpBatchPaddingError.emptyBatch
    }
    precondition(
        tokenCounts.allSatisfy { $0 > 0 },
        "chaque séquence du lot doit contenir au moins un jeton")
    return Qwen4ExpBatchPaddingLayout(
        maxLength: maxLength, leftPadding: tokenCounts.map { maxLength - $0 })
}

/// Remplit chaque ligne à gauche avec `padTokenID` (l'EOS du checkpoint —
/// voir le commentaire de fichier) jusqu'à `layout.maxLength`.
public func qwen4ExpLeftPadTokenIDs(
    _ sequences: [[Int32]], layout: Qwen4ExpBatchPaddingLayout, padTokenID: Int32
) -> [[Int32]] {
    precondition(sequences.count == layout.batchSize)
    return zip(sequences, layout.leftPadding).map { tokens, padding in
        precondition(padding + tokens.count == layout.maxLength)
        return Array(repeating: padTokenID, count: padding) + tokens
    }
}

/// `positionIDs` logiques du bloc rempli à gauche : chaque ligne recommence
/// sa propre horloge à 0 sur sa première colonne réelle (colonne
/// `leftPadding[row]`) — exactement les positions qu'elle porterait seule.
/// Les colonnes de remplissage reçoivent la position 0 ; sans conséquence,
/// puisqu'aucune d'elles n'est jamais lue par l'attention (masquée par
/// validité de clé, `Qwen4ExpQSAAttention.causalMask(..., leftPadding:)`) ni
/// par la récurrence GDN/PLE (masquée par `qwen4ExpBatchPaddingValidityMask`).
public func qwen4ExpLeftPaddedPositionIDs(
    layout: Qwen4ExpBatchPaddingLayout, offset: Int = 0
) -> [[Int32]] {
    layout.leftPadding.map { padding in
        (0 ..< layout.maxLength).map { column in
            column < padding ? Int32(0) : Int32(offset + column - padding)
        }
    }
}

/// `positionIDs` d'un pas de décodage à un jeton par ligne, après le
/// préremplissage : la ligne `row` en est à son jeton logique
/// `tokenCounts[row] + step` (`step` 0-indexé, 0 pour le premier jeton décodé
/// après le préremplissage). Remplissage à gauche oblige, toutes les lignes
/// avancent d'exactement un jeton réel par pas — jamais de nouveau
/// remplissage à décoder.
public func qwen4ExpLeftPaddedDecodePositionIDs(
    tokenCounts: [Int], step: Int
) -> [Int32] {
    precondition(step >= 0)
    return tokenCounts.map { Int32($0 + step) }
}

/// Masque de validité `[B,S]` (`true` = jeton réel, `false` = remplissage)
/// pour la portion de lot actuellement soumise au décodeur — le contrat
/// `[B,T]` déjà accepté par `Qwen4ExpGatedDeltaNet.callAsFunction(mask:)` et
/// `Qwen4ExpPLELayer.callAsFunction(mask:)`, documenté mais jamais alimenté
/// avant P12.2 (`Qwen4ExpStreamingDecoder.forward`, commentaire « GDN's
/// recurrence is causal by construction... reserved for padded/ragged
/// batches »). `columnOffset` est la position absolue, dans l'historique
/// complet du lot, de la première colonne de cet appel.
public func qwen4ExpBatchPaddingValidityMask(
    layout: Qwen4ExpBatchPaddingLayout, columnOffset: Int, columnCount: Int
) -> [[Bool]] {
    layout.leftPadding.map { padding in
        (0 ..< columnCount).map { columnOffset + $0 >= padding }
    }
}

// MARK: - Constructions MLXArray (nécessitent un device, pas un checkpoint)

/// Version `MLXArray` de `qwen4ExpLeftPaddedPositionIDs`, au format `[3,B,S]`
/// attendu par `Qwen4ExpMRoPE` — les trois axes MRoPE portent la même valeur,
/// comme `Qwen4ExpMRoPE.textPositionIDs` : cette sonde ne porte aucun
/// contenu visuel.
public func qwen4ExpLeftPaddedPositionIDsArray(
    layout: Qwen4ExpBatchPaddingLayout, offset: Int = 0
) -> MLXArray {
    let rows = qwen4ExpLeftPaddedPositionIDs(layout: layout, offset: offset)
    let temporal = MLXArray(rows.flatMap { $0 }).reshaped([layout.batchSize, layout.maxLength])
    return stacked([temporal, temporal, temporal])
}

/// Version `MLXArray` de `qwen4ExpLeftPaddedDecodePositionIDs`, au format
/// `[3,B,1]`.
public func qwen4ExpLeftPaddedDecodePositionIDsArray(
    tokenCounts: [Int], step: Int
) -> MLXArray {
    let values = qwen4ExpLeftPaddedDecodePositionIDs(tokenCounts: tokenCounts, step: step)
    let temporal = MLXArray(values).reshaped([tokenCounts.count, 1])
    return stacked([temporal, temporal, temporal])
}

/// Version `MLXArray` de `qwen4ExpBatchPaddingValidityMask`, construite par
/// comparaison plutôt que par imbrication de tableaux Swift — même style que
/// `Qwen4ExpQSAAttention.causalMask`.
public func qwen4ExpBatchPaddingValidityMaskArray(
    leftPadding: [Int], columnOffset: Int, columnCount: Int
) -> MLXArray {
    let columns = MLXArray(Int32(columnOffset) ..< Int32(columnOffset + columnCount))
        .reshaped([1, columnCount])
    let padding = MLXArray(leftPadding.map(Int32.init)).reshaped([leftPadding.count, 1])
    return columns .>= padding
}

// MARK: - Vérification de non-régression : le prompt de référence P12.1/P12.2

/// Le prompt de référence utilisé depuis P12.1 pour la non-contamination
/// inter-séquences. Sa suite greedy attendue ne dépend ni de sa position
/// dans le lot ni des longueurs des autres prompts — c'est le critère de
/// justesse non négociable de P12.2.
public let qwen4ExpBatchReferencePrompt =
    "Explique en français qui est le président de la Chine et quel est son rôle."

public let qwen4ExpBatchReferenceTokenIDs: [Int32] = [
    2229, 85648, 401, 1147, 183085, 1725, 41016, 90171,
]

/// Résultat de la comparaison d'une ligne à la référence : `expected` et
/// `actual` sont toujours de même longueur (`actual` est tronqué à la
/// longueur de `expected`), pour qu'un désaccord soit lisible jeton à jeton.
public struct Qwen4ExpBatchReferenceCheck: Equatable, Sendable {
    public let expected: [Int32]
    public let actual: [Int32]

    public init(expected: [Int32], actual: [Int32]) {
        self.expected = expected
        self.actual = actual
    }

    public var matches: Bool { actual == expected }

    /// Rang (0-indexé) du premier jeton qui diverge, ou `nil` si `matches`.
    public var firstMismatchIndex: Int? {
        zip(expected, actual).enumerated().first(where: { $0.element.0 != $0.element.1 })?.offset
    }
}

/// Compare les jetons générés par une ligne à la référence canonique, si son
/// prompt (débarrassé de ses espaces de bord) est bien le prompt de
/// référence. Renvoie `nil` pour toute autre ligne — jamais une vérification
/// silencieusement réussie sur un prompt qui n'est pas la référence.
public func qwen4ExpCheckBatchReference(
    prompt: String, generated: [Int32]
) -> Qwen4ExpBatchReferenceCheck? {
    guard prompt.trimmingCharacters(in: .whitespacesAndNewlines) == qwen4ExpBatchReferencePrompt
    else {
        return nil
    }
    let expected = qwen4ExpBatchReferenceTokenIDs
    return Qwen4ExpBatchReferenceCheck(expected: expected, actual: Array(generated.prefix(expected.count)))
}
