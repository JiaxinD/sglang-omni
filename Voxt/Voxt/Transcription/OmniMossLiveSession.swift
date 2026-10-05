import Foundation
import os

/// MOSS live preview: the window schedule MLXAudio used, decoded by the local server.
///
/// Fixed 4 s windows are decoded once each and become confirmed text; between
/// them, partial passes over the newest ≤2.5 s (at least 1.25 s, at most once
/// per second) supply the provisional text. One decode runs at a time and
/// windows share no speaker state, matching the original session.
nonisolated final class OmniMossLiveSession: @unchecked Sendable {
    nonisolated struct Configuration: Sendable {
        var sampleRate = 16_000
        var windowSeconds = 4.0
        var partialWindowSeconds = 2.5
        var minimumPartialSeconds = 1.25
        var decodeIntervalSeconds = 1.0
        var maxTokensPerPass = 1024
        var prompt: String?
    }

    private enum PassKind { case partial, finalWindow }

    private struct State {
        var pendingSamples: [Float] = []
        var pendingStartSample = 0
        var isDecoding = false
        var lastDecodeAt: Date?
        var completedText = ""
        var provisionalText = ""
        var isActive = true
        var decodeTask: Task<Void, Never>?
        var stopTask: Task<Void, Never>?
        var windowSegments: [OmniTranscriptSegment] = []
    }

    let events: AsyncStream<OmniLiveEvent>
    private let continuation: AsyncStream<OmniLiveEvent>.Continuation
    private let runtime: OmniASRRuntime
    private let endpoint: OmniServerEndpoint
    private let configuration: Configuration
    private let state = OSAllocatedUnfairLock(initialState: State())

    /// `endpoint` is a lease the caller holds for the session's lifetime.
    init(runtime: OmniASRRuntime, endpoint: OmniServerEndpoint, configuration: Configuration) {
        self.runtime = runtime
        self.endpoint = endpoint
        self.configuration = configuration
        (events, continuation) = AsyncStream.makeStream(of: OmniLiveEvent.self)
    }

    private var windowSamples: Int { Int((configuration.windowSeconds * Double(configuration.sampleRate)).rounded()) }
    private var minimumPartialSamples: Int {
        max(configuration.sampleRate, Int((configuration.minimumPartialSeconds * Double(configuration.sampleRate)).rounded()))
    }
    private var partialWindowSamples: Int {
        max(minimumPartialSamples, Int((min(configuration.windowSeconds, configuration.partialWindowSeconds) * Double(configuration.sampleRate)).rounded()))
    }

    func feedAudio(samples: [Float]) {
        let now = Date()
        let windowSamples = windowSamples
        let minimumPartialSamples = minimumPartialSamples
        let partialWindowSamples = partialWindowSamples
        let interval = max(1.0, configuration.decodeIntervalSeconds)
        state.withLock { state in
            guard let launch = nextLaunch(
                &state, appending: samples, now: now, windowSamples: windowSamples,
                minimumPartialSamples: minimumPartialSamples, partialWindowSamples: partialWindowSamples,
                interval: interval
            ) else { return }
            state.decodeTask = Task.detached { [self] in
                defer { self.state.withLock { $0.isDecoding = false } }
                await decode(launch.samples, kind: launch.kind, offsetSeconds: launch.offsetSeconds)
            }
        }
    }

    private func nextLaunch(
        _ state: inout State,
        appending samples: [Float],
        now: Date,
        windowSamples: Int,
        minimumPartialSamples: Int,
        partialWindowSamples: Int,
        interval: Double
    ) -> (samples: [Float], kind: PassKind, offsetSeconds: Double)? {
            guard state.isActive else { return nil }
            state.pendingSamples.append(contentsOf: samples)
            guard !state.isDecoding else { return nil }
            if state.pendingSamples.count >= windowSamples {
                let window = Array(state.pendingSamples.prefix(windowSamples))
                let offset = Double(state.pendingStartSample) / Double(configuration.sampleRate)
                state.pendingSamples.removeFirst(windowSamples)
                state.pendingStartSample += windowSamples
                state.isDecoding = true
                state.lastDecodeAt = now
                return (window, .finalWindow, offset)
            }
            guard state.pendingSamples.count >= minimumPartialSamples else { return nil }
            if let last = state.lastDecodeAt, now.timeIntervalSince(last) < interval { return nil }
            state.isDecoding = true
            state.lastDecodeAt = now
            let partialCount = min(state.pendingSamples.count, partialWindowSamples)
            let partialStart = state.pendingSamples.count - partialCount
            return (
                Array(state.pendingSamples[partialStart...]),
                .partial,
                Double(state.pendingStartSample + partialStart) / Double(configuration.sampleRate)
            )
    }

    /// Decodes the remaining tail as a final window, then emits `.ended`.
    func stop() {
        let snapshot = state.withLock { state -> ([Float], Int)? in
            guard state.isActive else { return nil }
            state.isActive = false
            return (state.pendingSamples, state.pendingStartSample)
        }
        guard let (pending, pendingStart) = snapshot else { return }
        let stopTask = Task.detached { [self] in
            while state.withLock({ $0.isDecoding }) {
                try? await Task.sleep(for: .milliseconds(10))
            }
            guard !Task.isCancelled else { return }
            if !pending.isEmpty {
                await decode(pending, kind: .finalWindow, offsetSeconds: Double(pendingStart) / Double(configuration.sampleRate))
            }
            guard !Task.isCancelled else { return }
            let (text, segments) = state.withLock {
                ($0.completedText.isEmpty ? $0.provisionalText : $0.completedText, $0.windowSegments)
            }
            continuation.yield(.ended(text: text, segments: segments))
            continuation.finish()
        }
        state.withLock { $0.stopTask = stopTask }
    }

    /// Abandons in-flight work; the server aborts the request when its connection closes.
    func cancel() {
        let tasks = state.withLock { state -> [Task<Void, Never>] in
            state.isActive = false
            state.pendingSamples = []
            return [state.decodeTask, state.stopTask].compactMap { $0 }
        }
        tasks.forEach { $0.cancel() }
        continuation.finish()
    }

    private func decode(_ samples: [Float], kind: PassKind, offsetSeconds: Double) async {
        let seconds = Double(samples.count) / Double(configuration.sampleRate)
        let maxTokens: Int
        switch kind {
        case .partial:
            maxTokens = min(configuration.maxTokensPerPass, max(48, Int((seconds * 16).rounded(.up))))
        case .finalWindow:
            maxTokens = min(max(1, configuration.maxTokensPerPass), max(96, Int((seconds * 32).rounded(.up))))
        }
        let request = OmniTranscriptionRequest(
            samples: samples,
            sampleRate: configuration.sampleRate,
            language: nil,
            prompt: configuration.prompt,
            maxNewTokens: max(1, maxTokens),
            stopAtEndOfText: true,
            stopOnTokenLoop: true
        )
        let streamed = OSAllocatedUnfairLock(initialState: "")
        let onDelta: @Sendable (String) -> Void = { [self] delta in
            guard !Task.isCancelled else { return }
            let provisional = streamed.withLock { text -> String in
                text += delta
                return OmniTranscriptionPlanning.offsetMossTimestamps(text, by: offsetSeconds)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
            guard !provisional.isEmpty else { return }
            let completed = state.withLock { state -> String in
                state.provisionalText = provisional
                return state.completedText
            }
            continuation.yield(Self.display(completed: completed, provisional: provisional))
        }
        do {
            let result = try await runtime.transcribe(request, holding: endpoint, onDelta: onDelta)
            guard !Task.isCancelled else { return }
            let text = OmniTranscriptionPlanning.offsetMossTimestampsInFinishedText(
                result.text.trimmingCharacters(in: .whitespacesAndNewlines),
                by: offsetSeconds
            )
            switch kind {
            case .partial:
                let completed = state.withLock { state -> String in
                    state.provisionalText = text
                    return state.completedText
                }
                continuation.yield(Self.display(completed: completed, provisional: text))
            case .finalWindow:
                // Each window's segments are parsed alone, so markup never spans windows.
                let windowSegments = OmniMossSegments.parse(text, fallbackEndSeconds: seconds)
                    .filter { $0.speakerID != nil }
                let completed = state.withLock { state -> String in
                    if !text.isEmpty {
                        state.completedText = state.completedText.isEmpty ? text : state.completedText + "\n" + text
                    }
                    state.windowSegments += windowSegments
                    state.provisionalText = ""
                    return state.completedText
                }
                continuation.yield(.display(confirmedText: completed, provisionalText: ""))
            }
        } catch {
            guard !Task.isCancelled, !(error is CancellationError) else { return }
            state.withLock { $0.isActive = false }
            continuation.yield(.failed(message: error.localizedDescription))
            continuation.finish()
        }
    }

    private static func display(completed: String, provisional: String) -> OmniLiveEvent {
        let confirmed = !completed.isEmpty && !provisional.isEmpty && !provisional.hasPrefix("\n")
            ? completed + "\n"
            : completed
        return .display(confirmedText: confirmed, provisionalText: provisional)
    }
}
