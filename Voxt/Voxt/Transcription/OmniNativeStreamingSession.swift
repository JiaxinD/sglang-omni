import Foundation
import MLXAudioSTT

/// Presents an Omni live session through the native streaming session seam.
///
/// The adapter leases the runtime for the session's lifetime, so retiring the
/// runtime waits for the session to end instead of cutting it off.
nonisolated final class OmniNativeStreamingSession: MLXNativeStreamingSession, @unchecked Sendable {
    private enum Backend {
        case qwen(OmniRealtimeTranscriptionSession)
        case moss(OmniMossLiveSession)
    }

    let events: AsyncStream<TranscriptionEvent>
    private let backend: Backend
    private let forwarding: Task<Void, Never>

    /// Qwen3-ASR preview over the realtime transcription socket.
    static func qwen(runtime: OmniASRRuntime, language: String?) async throws -> OmniNativeStreamingSession {
        let endpoint = try await runtime.beginUse()
        let session = OmniRealtimeTranscriptionSession(endpoint: endpoint, language: language)
        return OmniNativeStreamingSession(backend: .qwen(session), source: session.events, runtime: runtime)
    }

    /// MOSS preview: the original window schedule, each window decoded by the server.
    static func moss(runtime: OmniASRRuntime, prompt: String?, maxTokensPerPass: Int) async throws -> OmniNativeStreamingSession {
        let endpoint = try await runtime.beginUse()
        let session = OmniMossLiveSession(
            runtime: runtime,
            endpoint: endpoint,
            configuration: .init(maxTokensPerPass: maxTokensPerPass, prompt: prompt)
        )
        return OmniNativeStreamingSession(backend: .moss(session), source: session.events, runtime: runtime)
    }

    private init(backend: Backend, source: AsyncStream<OmniLiveEvent>, runtime: OmniASRRuntime) {
        self.backend = backend
        let isMoss: Bool
        if case .moss = backend { isMoss = true } else { isMoss = false }
        let (events, continuation) = AsyncStream.makeStream(of: TranscriptionEvent.self)
        self.events = events
        forwarding = Task.detached {
            for await event in source {
                switch event {
                case .display(let confirmedText, let provisionalText):
                    continuation.yield(.displayUpdate(confirmedText: confirmedText, provisionalText: provisionalText))
                case .ended(let text, let segments):
                    continuation.yield(.ended(Self.endedOutput(text: text, segments: segments, isMoss: isMoss)))
                case .failed(let message):
                    continuation.yield(.failed(StreamingFailure(message: message)))
                }
            }
            continuation.finish()
            await runtime.endUse()
        }
    }

    private static func endedOutput(text: String, segments: [OmniTranscriptSegment], isMoss: Bool) -> STTOutput {
        guard isMoss else { return STTOutput(text: text) }
        let segments = segments
            .map {
                STTTranscriptSegment(
                    text: $0.text,
                    startTime: $0.startSeconds,
                    endTime: $0.endSeconds,
                    speakerID: $0.speakerID
                )
            }
        return STTOutput(text: text, segments: segments.isEmpty ? nil : segments)
    }

    func feedAudio(samples: [Float]) {
        switch backend {
        case .qwen(let session): session.feedAudio(samples: samples)
        case .moss(let session): session.feedAudio(samples: samples)
        }
    }

    func stop() {
        switch backend {
        case .qwen(let session): session.stop()
        case .moss(let session): session.stop()
        }
    }

    func cancel() {
        switch backend {
        case .qwen(let session): session.cancel()
        case .moss(let session): session.cancel()
        }
    }
}
