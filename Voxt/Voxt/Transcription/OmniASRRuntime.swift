import Foundation

/// Where the Voxt-owned backend lives and how to start it.
nonisolated struct OmniBackendConfiguration: Sendable, Equatable {
    var pythonExecutable: URL
    var backendDirectory: URL
    var derivedRoot: URL
    var ffmpegLibraryDirectory: URL?
    var startupTimeoutSeconds: Double = 180
}

nonisolated struct OmniServerEndpoint: Sendable, Equatable {
    let host: String
    let port: Int
    let modelName: String
    let serverProcessIdentifier: Int32

    var baseURL: URL { URL(string: "http://\(host):\(port)")! }
}

nonisolated enum OmniASRRuntimeError: LocalizedError, Equatable {
    case launchFailed(String)
    case serverExited(Int?)
    case retired

    var errorDescription: String? {
        switch self {
        case .launchFailed(let reason):
            return "The local Omni server did not start: \(reason)"
        case .serverExited(let code):
            return "The local Omni server stopped unexpectedly (code \(code.map(String.init) ?? "unknown"))."
        case .retired:
            return "The local Omni server was released before the request finished."
        }
    }
}

/// One owned SGLang-Omni server for one model: prepare once, share, retire once.
///
/// `ready → retiring → stopped` is a real barrier: `retire()` returns only after
/// the supervisor has stopped the server and every process it started.
actor OmniASRRuntime {
    enum State: Equatable {
        case idle
        case starting
        case ready(OmniServerEndpoint)
        case retiring
        case stopped
        case failed(String)
    }

    nonisolated let kind: OmniASRModelKind
    nonisolated let modelDirectory: URL
    nonisolated let configuration: OmniBackendConfiguration

    private(set) var state: State = .idle
    private var supervisor: Process?
    private var controlPipe: Pipe?
    private var preparation: Task<OmniServerEndpoint, Error>?
    private var retirement: Task<Void, Never>?
    private var supervisorEvents: AsyncThrowingStream<[String: Any], Error>.Iterator?
    private let session: URLSession

    init(kind: OmniASRModelKind, modelDirectory: URL, configuration: OmniBackendConfiguration) {
        self.kind = kind
        self.modelDirectory = modelDirectory
        self.configuration = configuration
        let sessionConfiguration = URLSessionConfiguration.ephemeral
        // Loopback requests must never be routed through a system proxy.
        sessionConfiguration.connectionProxyDictionary = [:]
        sessionConfiguration.timeoutIntervalForRequest = 600
        sessionConfiguration.timeoutIntervalForResource = 3600
        self.session = URLSession(configuration: sessionConfiguration)
    }

    /// Starts the server once; concurrent callers share the same launch.
    func prepare() async throws -> OmniServerEndpoint {
        switch state {
        case .ready(let endpoint):
            return endpoint
        case .retiring, .stopped:
            throw OmniASRRuntimeError.retired
        case .failed(let reason):
            throw OmniASRRuntimeError.launchFailed(reason)
        case .idle, .starting:
            break
        }
        if let preparation {
            return try await preparation.value
        }
        state = .starting
        let task = Task { try await self.launch() }
        preparation = task
        do {
            let endpoint = try await task.value
            if case .starting = state {
                state = .ready(endpoint)
            }
            guard case .ready = state else { throw OmniASRRuntimeError.retired }
            return endpoint
        } catch {
            if case .starting = state {
                state = .failed(error.localizedDescription)
            }
            await stopSupervisor()
            throw error
        }
    }

    /// Idempotent and awaitable: every caller returns after the server is gone.
    func retire() async {
        if let retirement {
            await retirement.value
            return
        }
        state = .retiring
        let task = Task { await self.stopSupervisor() }
        retirement = task
        await task.value
        state = .stopped
    }

    func transcribe(
        _ request: OmniTranscriptionRequest,
        onDelta: (@Sendable (String) -> Void)? = nil
    ) async throws -> OmniTranscriptionResult {
        guard case .ready(let endpoint) = state else {
            throw OmniASRRuntimeError.retired
        }
        let boundary = "voxt-\(UUID().uuidString)"
        var urlRequest = URLRequest(url: endpoint.baseURL.appendingPathComponent("v1/audio/transcriptions"))
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        let body = OmniMultipartBody.transcription(request, modelName: endpoint.modelName, boundary: boundary)
        let (bytes, response) = try await session.streamingUpload(urlRequest, body: body)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        if status != 200 {
            var detail = ""
            for try await line in bytes.lines {
                detail += line
                if detail.count > 2000 { break }
            }
            throw OmniTranscriptionError.httpStatus(status, detail)
        }
        var parser = OmniTranscriptionStreamParser()
        for try await line in bytes.lines {
            try Task.checkCancellation()
            if let delta = try parser.consume(line: line), !delta.isEmpty {
                onDelta?(delta)
            }
        }
        return OmniTranscriptionResult(text: try parser.finish())
    }

    private func launch() async throws -> OmniServerEndpoint {
        let process = Process()
        process.executableURL = configuration.pythonExecutable
        var arguments = [
            "-m", "voxt_omni_backend.supervisor",
            "--model-kind", kind.rawValue,
            "--model-directory", modelDirectory.path,
            "--derived-root", configuration.derivedRoot.path,
            "--startup-timeout-s", String(configuration.startupTimeoutSeconds),
        ]
        if let ffmpegLibraryDirectory = configuration.ffmpegLibraryDirectory {
            arguments += ["--ffmpeg-library-directory", ffmpegLibraryDirectory.path]
        }
        process.arguments = arguments
        process.currentDirectoryURL = configuration.backendDirectory
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONPATH"] = configuration.backendDirectory.path
        environment["PYTHONUNBUFFERED"] = "1"
        process.environment = environment
        let control = Pipe()
        let events = Pipe()
        process.standardInput = control
        process.standardOutput = events
        process.standardError = FileHandle.nullDevice
        do {
            try process.run()
        } catch {
            throw OmniASRRuntimeError.launchFailed(error.localizedDescription)
        }
        supervisor = process
        controlPipe = control
        var iterator = Self.eventStream(events.fileHandleForReading).makeAsyncIterator()
        guard let first = try await iterator.next() else {
            throw OmniASRRuntimeError.launchFailed("supervisor exited before reporting")
        }
        supervisorEvents = iterator
        guard first["event"] as? String == "ready",
              let host = first["host"] as? String,
              let port = first["port"] as? Int,
              let modelName = first["model_name"] as? String,
              let serverPID = first["server_pid"] as? Int
        else {
            throw OmniASRRuntimeError.launchFailed(first["reason"] as? String ?? "\(first)")
        }
        return OmniServerEndpoint(
            host: host,
            port: port,
            modelName: modelName,
            serverProcessIdentifier: Int32(serverPID)
        )
    }

    private func stopSupervisor() async {
        preparation?.cancel()
        guard let process = supervisor else { return }
        supervisor = nil
        if process.isRunning, let controlPipe {
            let shutdown = Data("{\"command\": \"shutdown\"}\n".utf8)
            try? controlPipe.fileHandleForWriting.write(contentsOf: shutdown)
            try? controlPipe.fileHandleForWriting.close()
        }
        controlPipe = nil
        await Self.waitForExit(process)
        supervisorEvents = nil
    }

    private static func waitForExit(_ process: Process) async {
        while process.isRunning {
            try? await Task.sleep(for: .milliseconds(20))
        }
    }

    private nonisolated static func eventStream(
        _ handle: FileHandle
    ) -> AsyncThrowingStream<[String: Any], Error> {
        AsyncThrowingStream { continuation in
            let reader = Task.detached {
                do {
                    for try await line in handle.bytes.lines {
                        guard let data = line.data(using: .utf8),
                              let event = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                        else { continue }
                        continuation.yield(event)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in reader.cancel() }
        }
    }
}

nonisolated extension URLSession {
    /// Streams an upload's response so cancellation reaches the server mid-request.
    func streamingUpload(
        _ request: URLRequest,
        body: Data
    ) async throws -> (URLSession.AsyncBytes, URLResponse) {
        var uploadRequest = request
        uploadRequest.httpBody = body
        return try await bytes(for: uploadRequest)
    }
}
