import Foundation

/// Constantes de troncature reprises telles quelles de `Scripts/agent-loop.py`
/// (P13.2) : un pas d'agent coûte cher en préfill (P13.2/P13.3), il ne faut
/// jamais renvoyer un fichier ou un `grep` entiers dans l'historique.
public enum AgentTruncation {
    /// `read_file`/`grep` : au plus 200 lignes lues, résultat coupé à 6000
    /// caractères (Python : `[:6000]`).
    public static let toolOutputCharLimit = 6000
    /// Contenu du message `role: tool` réinjecté dans la conversation :
    /// second plafond, plus serré (Python : `run_tool(...)[:4000]`).
    public static let messageCharLimit = 4000
    /// `read_file` : au plus 200 lignes par appel.
    public static let readFileLineLimit = 200
    /// `grep` : au plus 60 lignes de correspondance.
    public static let grepLineLimit = 60
    /// `list_files` : au plus 80 entrées.
    public static let listFilesEntryLimit = 80

    /// Coupe `text` à `limit` caractères (Unicode scalaires groupés en
    /// graphèmes — comportement suffisant pour du texte de code/documentation).
    public static func truncate(_ text: String, limit: Int) -> String {
        guard limit > 0, text.count > limit else { return text }
        return String(text.prefix(limit))
    }
}
