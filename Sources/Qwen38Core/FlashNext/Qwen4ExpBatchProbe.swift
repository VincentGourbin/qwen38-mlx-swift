import Foundation

/// P11 (dernier chantier, `docs/knowledge/log.md` 2026-09-13 « Les 33 %
/// d'inactivité GPU ne viennent pas de la soumission : ils sont
/// structurels ») : logique pure de `flash-batch-probe`, testable sans
/// checkpoint.
///
/// La trace Metal montre que le GPU est inactif un tiers du pas de décodage
/// à lot de taille 1 — pas assez de travail indépendant pour remplir 40
/// cœurs. La courbe « jetons par forward » (P11.4a) chiffre la place
/// disponible, mais avec des jetons du **même** flux, qui partagent tout
/// leur état. La question ouverte est de savoir si le même gain tient avec
/// des **séquences indépendantes**, chacune avec son propre état récurrent
/// GDN et son propre cache d'attention. Ce fichier isole le découpage des
/// prompts, la validation de longueur et la vérification de non-
/// contamination inter-séquences — tout ce qui ne dépend pas d'un forward
/// MLX réel — pour que `Qwen38Tests` puisse les exercer sans charger de
/// checkpoint.
public enum Qwen4ExpBatchProbeError: LocalizedError, Equatable {
    case noPrompts
    case emptyPromptAtIndex(Int)
    case unequalPromptLengths(tokenCounts: [Int])
    case invalidMaxNewTokens(Int)

    public var errorDescription: String? {
        switch self {
        case .noPrompts:
            return "--prompts ne peut pas être vide ou ne contenir que des espaces."
        case .emptyPromptAtIndex(let index):
            return
                "--prompts : le prompt \(index) (0-indexé) est vide une fois découpé sur \"|\" "
                + "et débarrassé de ses espaces de bord."
        case .unequalPromptLengths(let tokenCounts):
            let detail = tokenCounts.enumerated()
                .map { "prompt \($0.offset) : \($0.element) jeton(s)" }
                .joined(separator: " · ")
            return
                "Les prompts doivent produire exactement le même nombre de jetons après rendu "
                + "du gabarit ChatML (pas de remplissage à cette étape) ; obtenu — \(detail)."
        case .invalidMaxNewTokens(let value):
            return "--max-new-tokens doit être positif ; reçu \(value)."
        }
    }
}

/// Découpe `--prompts "<a>|<b>|<c>"` en une liste ordonnée de prompts, un par
/// séquence du lot. Chaque élément est débarrassé de ses espaces de bord ;
/// un élément vide (chaîne d'entrée vide, barre verticale finale ou
/// consécutive) est une erreur explicite qui nomme sa position, jamais une
/// séquence silencieusement absente du lot.
public func qwen4ExpSplitBatchPrompts(_ raw: String) throws -> [String] {
    let trimmedRaw = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmedRaw.isEmpty else {
        throw Qwen4ExpBatchProbeError.noPrompts
    }
    let parts = raw.split(separator: "|", omittingEmptySubsequences: false).map {
        $0.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    for (index, part) in parts.enumerated() where part.isEmpty {
        throw Qwen4ExpBatchProbeError.emptyPromptAtIndex(index)
    }
    return parts
}

/// Vérifie que chaque prompt du lot produit le même nombre de jetons après
/// rendu ChatML — la contrainte assumée par cette étape de la sonde (pas de
/// remplissage, pas de masque de remplissage). Renvoie la longueur commune ;
/// lève une erreur qui imprime la longueur obtenue pour chaque prompt sinon,
/// pour que l'appelant puisse ajuster ses prompts.
@discardableResult
public func qwen4ExpValidateEqualPromptTokenCounts(_ tokenCounts: [Int]) throws -> Int {
    guard let first = tokenCounts.first, tokenCounts.allSatisfy({ $0 == first }) else {
        throw Qwen4ExpBatchProbeError.unequalPromptLengths(tokenCounts: tokenCounts)
    }
    return first
}

/// Résultat de la vérification de non-contamination inter-séquences : les B
/// séquences décodées côte à côte dans un même forward doivent produire des
/// identifiants strictement indépendants les uns des autres. Comparer
/// chaque séquence à la première (au lieu d'une comparaison par paires)
/// suffit à détecter toute contamination et nomme directement le premier
/// rang fautif.
public struct Qwen4ExpBatchParityResult: Equatable, Sendable {
    public let allEqual: Bool
    /// Rang (0-indexé, toujours > 0) de la première séquence qui diffère de
    /// la séquence 0, ou `nil` si toutes concordent.
    public let firstMismatchIndex: Int?

    public init(allEqual: Bool, firstMismatchIndex: Int?) {
        self.allEqual = allEqual
        self.firstMismatchIndex = firstMismatchIndex
    }
}

/// Compare les identifiants produits par chaque séquence du lot à ceux de la
/// séquence 0. Un lot vide ou à une seule séquence concorde trivialement —
/// il n'y a personne avec qui se contaminer.
public func qwen4ExpBatchParityCheck(_ sequences: [[Int32]]) -> Qwen4ExpBatchParityResult {
    guard let reference = sequences.first else {
        return Qwen4ExpBatchParityResult(allEqual: true, firstMismatchIndex: nil)
    }
    for (index, sequence) in sequences.enumerated() where index > 0 {
        if sequence != reference {
            return Qwen4ExpBatchParityResult(allEqual: false, firstMismatchIndex: index)
        }
    }
    return Qwen4ExpBatchParityResult(allEqual: true, firstMismatchIndex: nil)
}
