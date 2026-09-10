import Foundation
import MLXProfiler

/// Long-lived profiling session shared across requests (2026-09-10).
///
/// By default every generation creates its own `ProfilingSession` and
/// disables the profiler when the stream ends, which is what the GUI's
/// per-run report and "Exporter trace" need. A process that wants ONE trace
/// covering a whole serving session (e.g. `qwen38 serve --trace`) installs a
/// shared session here: generation paths then wrap each request in a
/// `Requête N` phase on that session, never replace `activeSession`, never
/// disable the profiler, and skip the per-request report/Chrome export.
public enum Qwen38Profiling {
    nonisolated(unsafe) public static var sharedSession: ProfilingSession?
    private static let counterLock = NSLock()
    nonisolated(unsafe) private static var requestCounter = 0

    /// Returns the session to record into and whether the caller owns it.
    /// A caller that owns its session disables the profiler at the end; a
    /// caller that borrows the shared one only closes its request phase.
    public static func beginRequestSession(
        title: String, metadata: [String: String], phase: String
    ) -> (session: ProfilingSession, ownsSession: Bool, phase: String) {
        let profiler = MLXProfiler.shared
        if let shared = sharedSession {
            // The per-turn index restarts at 1 for every stateless request;
            // number requests globally so the trace reads "Requête 1…N".
            counterLock.lock(); requestCounter += 1; let n = requestCounter; counterLock.unlock()
            let globalPhase = "Requête \(n)"
            profiler.activeSession = shared
            profiler.enable()
            profiler.start(globalPhase)
            return (shared, false, globalPhase)
        }
        let session = ProfilingSession(config: .singleRun, subsystem: "com.qwen38mlx")
        session.title = title
        for (key, value) in metadata { session.metadata[key] = value }
        profiler.activeSession = session
        profiler.enable()
        return (session, true, phase)
    }

    public static func endRequestSession(ownsSession: Bool, phase: String) {
        let profiler = MLXProfiler.shared
        if ownsSession { profiler.disable() } else { profiler.end(phase) }
    }
}
