import Foundation
import Testing
@testable import Qwen38Core
@testable import Qwen38Server

// Issue #2 : sécurité et fiabilité du serveur. Un vrai serveur Hummingbird sur
// un port aléatoire, un catalogue factice (aucun poids) : ces tests ne
// chargent jamais de modèle, ils vérifient ce qui est refusé avant.

private func securityCatalog() throws -> URL {
    let root = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
        .appendingPathComponent("qwen38-security-\(UUID().uuidString)", isDirectory: true)
    let flashNext = root.appendingPathComponent("Qwen3.8-Flash-Next-4bit", isDirectory: true)
    try FileManager.default.createDirectory(at: flashNext, withIntermediateDirectories: true)
    try JSONSerialization.data(withJSONObject: qwen4ExpFixtureConfig())
        .write(to: flashNext.appendingPathComponent("config.json"))
    return root
}

private func request(
    _ port: Int, _ path: String, method: String = "GET", key: String? = nil, body: Data? = nil
) async throws -> (status: Int, body: Data) {
    var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/\(path)")!)
    request.httpMethod = method
    if let key { request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization") }
    if let body {
        request.httpBody = body
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    }
    let (data, response) = try await URLSession.shared.data(for: request)
    return ((response as? HTTPURLResponse)?.statusCode ?? -1, data)
}

private func chatBody(content: Any) throws -> Data {
    try JSONSerialization.data(withJSONObject: [
        "model": "Qwen3.8-Flash-Next-4bit",
        "messages": [["role": "user", "content": content]],
    ])
}

@Test("Issue #2 : hors boucle locale, le serveur refuse de démarrer sans clé d'API")
func serverRefusesNonLoopbackWithoutKey() async throws {
    let root = try securityCatalog()
    defer { try? FileManager.default.removeItem(at: root) }
    let server = Qwen38InferenceServer(runtime: Qwen38Runtime())
    await #expect(throws: Qwen38ServerError.remoteAccessNeedsAPIKey("0.0.0.0")) {
        try await server.start(port: Int.random(in: 20_000 ..< 40_000), host: "0.0.0.0", modelsDirectory: root)
    }
    #expect(Qwen38InferenceServer.isLoopback("127.0.0.1"))
    #expect(Qwen38InferenceServer.isLoopback("localhost"))
    #expect(!Qwen38InferenceServer.isLoopback("0.0.0.0"))
    #expect(!Qwen38InferenceServer.isLoopback("192.168.1.20"))
}

@Test("Issue #2 : avec une clé, /v1/models, /metrics et le chat exigent la clé ; /healthz ne dit que « vivant »")
func serverRequiresKeyAndRedactsHealth() async throws {
    let root = try securityCatalog()
    defer { try? FileManager.default.removeItem(at: root) }
    let port = Int.random(in: 20_000 ..< 40_000)
    let server = Qwen38InferenceServer(runtime: Qwen38Runtime())
    try await server.start(port: port, apiKey: "secret-de-test", modelsDirectory: root)
    defer { Task { await server.stop() } }

    #expect(try await request(port, "v1/models").status == 401)
    #expect(try await request(port, "v1/models", key: "mauvaise").status == 401)
    #expect(try await request(port, "v1/models", key: "secret-de-test").status == 200)
    #expect(try await request(port, "metrics").status == 401)
    #expect(try await request(port, "metrics", key: "secret-de-test").status == 200)
    #expect(try await request(port, "v1/chat/completions", method: "POST",
                              body: try chatBody(content: "salut")).status == 401)

    let anonymous = try await request(port, "healthz")
    #expect(anonymous.status == 200)
    let anonymousKeys = Set(((try JSONSerialization.jsonObject(with: anonymous.body)) as? [String: Any] ?? [:]).keys)
    #expect(anonymousKeys == ["status", "model_loaded"])
    let authorized = try await request(port, "healthz", key: "secret-de-test")
    let authorizedKeys = Set(((try JSONSerialization.jsonObject(with: authorized.body)) as? [String: Any] ?? [:]).keys)
    #expect(authorizedKeys.contains("batch_size_configured"))
}

@Test("Issue #2 : corps trop gros → 413 ; file:// → 400 ; plus de 4 images → 413")
func serverEnforcesBodyAndMediaLimits() async throws {
    let root = try securityCatalog()
    defer { try? FileManager.default.removeItem(at: root) }
    let port = Int.random(in: 20_000 ..< 40_000)
    let server = Qwen38InferenceServer(runtime: Qwen38Runtime())
    try await server.start(port: port, modelsDirectory: root)
    defer { Task { await server.stop() } }

    let oversized = try chatBody(content: String(repeating: "a", count: Qwen38InferenceServer.maxRequestBodyBytes + 1024))
    #expect(try await request(port, "v1/chat/completions", method: "POST", body: oversized).status == 413)

    let fileImage: [[String: Any]] = [
        ["type": "text", "text": "décris"],
        ["type": "image_url", "image_url": ["url": "file:///etc/hosts"]],
    ]
    #expect(try await request(port, "v1/chat/completions", method: "POST",
                              body: try chatBody(content: fileImage)).status == 400)

    let tinyPNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    let fiveImages: [[String: Any]] = [["type": "text", "text": "compare"]]
        + Array(repeating: ["type": "image_url", "image_url": ["url": tinyPNG]], count: 5)
    #expect(try await request(port, "v1/chat/completions", method: "POST",
                              body: try chatBody(content: fiveImages)).status == 413)
}

@Test("Issue #2 : comparaison de clé à temps constant, correcte sur égalité, différence et longueurs")
func serverConstantTimeKeyComparison() {
    #expect(Qwen38InferenceServer.constantTimeEquals("Bearer abc", "Bearer abc"))
    #expect(!Qwen38InferenceServer.constantTimeEquals("Bearer abd", "Bearer abc"))
    #expect(!Qwen38InferenceServer.constantTimeEquals("Bearer ab", "Bearer abc"))
    #expect(!Qwen38InferenceServer.constantTimeEquals("Bearer abcd", "Bearer abc"))
    #expect(!Qwen38InferenceServer.constantTimeEquals("", "Bearer abc"))
}

@Test("Issue #2 : la file n'est libérée qu'à la vraie fin du flux, ou quand le client l'abandonne")
func serverQueueReleasedAtStreamEnd() async throws {
    actor Flag { var value = false; func set() { value = true } }

    // 1. Flux consommé jusqu'au bout : rien de libéré avant le dernier événement.
    let released = Flag()
    let release = Qwen38ReleaseOnce { await released.set() }
    let (source, feed) = AsyncThrowingStream<Qwen38GenerationEvent, Error>.makeStream()
    let wrapped = qwen38ReleasingWhenFinished(source, release: release)
    #expect(release.handedOff)
    feed.yield(.chunk("un"))
    var iterator = wrapped.makeAsyncIterator()
    _ = try await iterator.next()
    #expect(await released.value == false)
    feed.finish()
    let end = try await iterator.next()
    #expect(end.map { _ in true } == nil)
    #expect(await released.value == true)

    // 2. Client qui abandonne le flux en cours de route : libéré quand même.
    let abandoned = Flag()
    let abandonedRelease = Qwen38ReleaseOnce { await abandoned.set() }
    let (endless, _) = AsyncThrowingStream<Qwen38GenerationEvent, Error>.makeStream()
    let consumer = Task {
        for try await _ in qwen38ReleasingWhenFinished(endless, release: abandonedRelease) {}
    }
    try await Task.sleep(for: .milliseconds(50))
    #expect(await abandoned.value == false)
    consumer.cancel()
    for _ in 0 ..< 100 where await abandoned.value == false { try await Task.sleep(for: .milliseconds(10)) }
    #expect(await abandoned.value == true)
}
