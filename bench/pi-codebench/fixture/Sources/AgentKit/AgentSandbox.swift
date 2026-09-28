import Foundation

/// Un chemin demandé par le modèle qui sort du dossier racine choisi par
/// l'utilisateur. Jamais suivi — voir `AgentSandbox.resolve`.
public struct AgentPathError: Error, LocalizedError, Equatable, Sendable {
    public let requested: String
    public init(requested: String) { self.requested = requested }
    public var errorDescription: String? {
        "chemin hors du dossier racine choisi : \"\(requested)\""
    }
}

/// Résout tout chemin d'outil sous une racine fixe, et refuse ce qui en
/// sort. Reprend le principe de `safe()` dans `Scripts/agent-loop.py` :
/// résolution absolue puis vérification stricte du préfixe (avec une
/// frontière de `/`, pour ne jamais confondre `.../projet-evil` avec
/// `.../projet`).
///
/// Ne résout pas les liens symboliques (ni côté racine, ni côté chemin
/// demandé) — même comportement que la version Python, qui ne fait que de
/// l'arithmétique de chemins sans toucher au système de fichiers.
public struct AgentSandbox: Sendable, Equatable {
    public let root: URL

    public init(root: URL) {
        self.root = URL(fileURLWithPath: root.path).standardizedFileURL
    }

    /// Résout `path` sous `root`. `nil` ou vide veut dire "la racine
    /// elle-même", comme `safe(".")` côté Python.
    public func resolve(_ path: String?) throws -> URL {
        let trimmed = path?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let requested = trimmed.isEmpty ? "." : trimmed

        let candidate: URL
        if requested.hasPrefix("/") {
            // Un chemin absolu remplace la racine, exactement comme
            // `os.path.join(ROOT, p)` en Python quand `p` est absolu — pour
            // que le contrôle de préfixe qui suit le rejette bien.
            candidate = URL(fileURLWithPath: requested)
        } else {
            candidate = URL(fileURLWithPath: requested, relativeTo: root)
        }
        let resolved = candidate.standardizedFileURL

        let rootPath = root.path
        let resolvedPath = resolved.path
        guard resolvedPath == rootPath || resolvedPath.hasPrefix(rootPath + "/") else {
            throw AgentPathError(requested: trimmed)
        }
        return resolved
    }
}
