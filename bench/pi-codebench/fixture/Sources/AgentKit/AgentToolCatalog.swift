import Foundation

/// Les quatre outils portés depuis `Scripts/agent-loop.py`. Trois en lecture
/// seule (`list_files`, `read_file`, `grep`), un pour conclure
/// (`final_answer`). Pas de `run_command` : aucune exécution de commande
/// shell côté GUI (voir le rapport de portage).
public enum AgentToolCatalog {
    /// Nom de chaque paramètre requis par outil — sert à calculer la
    /// métrique « appel valide » (outil connu + paramètres requis présents),
    /// la même que celle qui a jugé tout le chantier §P13.
    public static let requiredArguments: [String: [String]] = [
        "list_files": ["path"],
        "read_file": ["path"],
        "grep": ["pattern"],
        "final_answer": ["answer"],
    ]

    public static let toolNames: Set<String> = Set(requiredArguments.keys)

    /// Schéma JSON envoyé au serveur dans `tools` (format OpenAI), construit
    /// à la demande (jamais stocké en état global) pour rester `Sendable`
    /// sans y consacrer un type JSON dédié — voir `AgentWireFormat`.
    public static func schema() -> [[String: Any]] {
        [
            [
                "type": "function",
                "function": [
                    "name": "list_files",
                    "description": "Liste les fichiers d'un dossier sous la racine choisie.",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "path": ["type": "string", "description": "chemin relatif à la racine"]
                        ],
                        "required": ["path"],
                    ],
                ],
            ],
            [
                "type": "function",
                "function": [
                    "name": "read_file",
                    "description": "Lit un fichier sous la racine choisie. Renvoie au plus 200 lignes.",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "path": ["type": "string"],
                            "start_line": ["type": "integer"],
                        ],
                        "required": ["path"],
                    ],
                ],
            ],
            [
                "type": "function",
                "function": [
                    "name": "grep",
                    "description": "Cherche un motif sous la racine choisie et renvoie les lignes correspondantes.",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "pattern": ["type": "string"],
                            "path": ["type": "string"],
                        ],
                        "required": ["pattern"],
                    ],
                ],
            ],
            [
                "type": "function",
                "function": [
                    "name": "final_answer",
                    "description": "Donne la réponse finale à l'utilisateur et termine.",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "answer": ["type": "string"]
                        ],
                        "required": ["answer"],
                    ],
                ],
            ],
        ]
    }
}
