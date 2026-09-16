/// Empêche deux exécutions de la boucle d'agent de tourner en même temps.
///
/// Extrait de `AgentPanelViewModel` (`Qwen38BenchUI`) pour rester testable
/// sans réseau ni checkpoint — même raison d'être que le reste de ce module
/// (voir le commentaire sur la cible `Qwen38Agent` dans `Package.swift`).
/// Contexte du bug corrigé : avant la remontée de l'état du panneau Agent
/// vers `BenchViewModel`, changer d'onglet détruisait `AgentPanelViewModel`
/// (et sa tâche en cours avec lui) ; une fois l'état hissé au niveau du
/// modèle qui survit aux onglets, il devient possible de cliquer « Lancer »
/// deux fois de suite (par exemple par un double clic, ou pendant qu'une
/// tâche tourne encore en arrière-plan) — cette garde est ce qui empêche
/// deux boucles de vivre en même temps dans ce cas.
public struct AgentRunGate: Sendable {
    public private(set) var isRunning = false

    public init() {}

    /// Le seul point d'entrée pour démarrer une boucle : renvoie `true` et
    /// marque la garde comme occupée si aucune boucle ne tourne déjà, sinon
    /// renvoie `false` sans rien changer — l'appelant ne doit alors rien
    /// faire d'autre (pas de nouvelle tâche, pas de réinitialisation de
    /// l'état affiché).
    public mutating func tryStart() -> Bool {
        guard !isRunning else { return false }
        isRunning = true
        return true
    }

    /// Libère la garde, que la boucle se soit terminée normalement (réponse
    /// finale, budget épuisé, erreur) ou qu'elle ait été arrêtée par
    /// l'utilisateur. Idempotent : appeler `finish()` alors que la garde est
    /// déjà libre ne fait rien.
    public mutating func finish() {
        isRunning = false
    }
}
