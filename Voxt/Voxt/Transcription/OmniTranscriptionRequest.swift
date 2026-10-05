import Foundation

/// Wire formats and per-model request planning for the local SGLang-Omni server.
nonisolated enum OmniASRModelKind: String, Sendable, CaseIterable {
    case qwen3ASR = "qwen3_asr"
    case mossTranscribeDiarize = "moss_transcribe_diarize"
    case whisper = "whisper"
}

nonisolated struct OmniTranscriptionRequest: Sendable, Equatable {
    var samples: [Float]
    var sampleRate: Int
    var language: String?
    var prompt: String?
    var maxNewTokens: Int?
    var stopAtEndOfText: Bool
    var stopOnTokenLoop: Bool
}

nonisolated struct OmniTranscriptionResult: Sendable, Equatable {
    let text: String
}

nonisolated enum OmniTranscriptionError: LocalizedError, Equatable {
    case httpStatus(Int, String)
    case streamError(String)
    case streamEndedWithoutDone
    case malformedEvent(String)

    var errorDescription: String? {
        switch self {
        case .httpStatus(let status, let detail):
            return "Local Omni server returned HTTP \(status): \(detail)"
        case .streamError(let message):
            return "Local Omni transcription failed: \(message)"
        case .streamEndedWithoutDone:
            return "Local Omni transcription ended before its final result."
        case .malformedEvent(let line):
            return "Local Omni transcription sent an unreadable event: \(line.prefix(120))"
        }
    }
}

nonisolated enum OmniWAVEncoding {
    /// IEEE float32 mono WAV, so the server decodes the exact capture samples.
    static func float32WAV(samples: [Float], sampleRate: Int) -> Data {
        let bytesPerSample = 4
        let dataByteCount = samples.count * bytesPerSample
        var data = Data(capacity: 44 + dataByteCount)
        func append<T: FixedWidthInteger>(_ value: T) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        data.append(contentsOf: Array("RIFF".utf8))
        append(UInt32(36 + dataByteCount))
        data.append(contentsOf: Array("WAVEfmt ".utf8))
        append(UInt32(16))
        append(UInt16(3))
        append(UInt16(1))
        append(UInt32(sampleRate))
        append(UInt32(sampleRate * bytesPerSample))
        append(UInt16(bytesPerSample))
        append(UInt16(32))
        data.append(contentsOf: Array("data".utf8))
        append(UInt32(dataByteCount))
        samples.withUnsafeBufferPointer { buffer in
            for sample in buffer {
                append(sample.bitPattern)
            }
        }
        return data
    }

    /// Little-endian PCM16 for the realtime socket, clamped like AVAudioConverter.
    static func pcm16(samples: ArraySlice<Float>) -> Data {
        var data = Data(capacity: samples.count * 2)
        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let value = Int16((clamped * Float(Int16.max)).rounded())
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        return data
    }
}

nonisolated enum OmniMultipartBody {
    static func transcription(
        _ request: OmniTranscriptionRequest,
        modelName: String,
        boundary: String
    ) -> Data {
        var fields: [(String, String)] = [
            ("model", modelName),
            ("stream", "true"),
            ("response_format", "json"),
        ]
        if let language = request.language {
            fields.append(("language", language))
        }
        if let prompt = request.prompt {
            fields.append(("prompt", prompt))
        }
        if let maxNewTokens = request.maxNewTokens {
            fields.append(("max_new_tokens", String(maxNewTokens)))
        }
        if request.stopAtEndOfText {
            fields.append(("stop_at_end_of_text", "true"))
        }
        if request.stopOnTokenLoop {
            fields.append(("stop_on_token_loop", "true"))
        }
        var body = Data()
        for (name, value) in fields {
            body.append(Data("--\(boundary)\r\n".utf8))
            body.append(Data("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".utf8))
            body.append(Data(value.utf8))
            body.append(Data("\r\n".utf8))
        }
        body.append(Data("--\(boundary)\r\n".utf8))
        body.append(Data("Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n".utf8))
        body.append(Data("Content-Type: audio/wav\r\n\r\n".utf8))
        body.append(OmniWAVEncoding.float32WAV(samples: request.samples, sampleRate: request.sampleRate))
        body.append(Data("\r\n--\(boundary)--\r\n".utf8))
        return body
    }
}

/// Server-sent transcription events. Only transcript.text.done is a success.
nonisolated struct OmniTranscriptionStreamParser {
    enum Outcome: Equatable {
        case pending
        case done(String)
    }

    private(set) var outcome: Outcome = .pending

    /// Returns the text delta the line carried, if any.
    @discardableResult
    mutating func consume(line: String) throws -> String? {
        guard line.hasPrefix("data:") else { return nil }
        let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
        if payload == "[DONE]" {
            guard case .done = outcome else { throw OmniTranscriptionError.streamEndedWithoutDone }
            return nil
        }
        guard let data = payload.data(using: .utf8),
              let event = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = event["type"] as? String
        else {
            throw OmniTranscriptionError.malformedEvent(payload)
        }
        switch type {
        case "transcript.text.delta":
            return event["delta"] as? String
        case "transcript.text.done":
            outcome = .done(event["text"] as? String ?? "")
            return nil
        case "error":
            let error = event["error"] as? [String: Any]
            throw OmniTranscriptionError.streamError(error?["message"] as? String ?? payload)
        default:
            return nil
        }
    }

    func finish() throws -> String {
        guard case .done(let text) = outcome else { throw OmniTranscriptionError.streamEndedWithoutDone }
        return text
    }
}

/// Splits one recording the way each original Swift model decoded it.
nonisolated enum OmniTranscriptionPlanning {
    struct Window: Equatable {
        let sampleRange: Range<Int>
        let offsetSeconds: Double
    }

    /// Whisper decodes fixed, non-overlapping 30 s windows independently.
    static let whisperWindowSeconds = 30.0
    /// Whisper's decoder holds 448 positions; one slot stays free after the prompt.
    static let whisperDecoderPositions = 448

    static func fixedWindows(sampleCount: Int, sampleRate: Int, windowSeconds: Double) -> [Window] {
        let windowSamples = Int(windowSeconds * Double(sampleRate))
        guard sampleCount > windowSamples else {
            return [Window(sampleRange: 0..<sampleCount, offsetSeconds: 0)]
        }
        return stride(from: 0, to: sampleCount, by: windowSamples).map { start in
            Window(
                sampleRange: start..<min(start + windowSamples, sampleCount),
                offsetSeconds: Double(start) / Double(sampleRate)
            )
        }
    }

    struct PaddedChunk: Equatable {
        let samples: [Float]
        let offsetSeconds: Double
    }

    /// The MLXAudio splitter Qwen3-ASR and MOSS used, on plain arrays: cut near
    /// the quietest 100 ms within ±5 s of each chunk end and zero-pad any chunk
    /// shorter than the minimum duration.
    static func energySplitChunks(
        _ samples: [Float],
        sampleRate: Int,
        chunkDurationSeconds: Float,
        minChunkDurationSeconds: Float,
        searchExpandSeconds: Float = 5.0,
        minWindowMilliseconds: Float = 100.0
    ) -> [PaddedChunk] {
        let totalSamples = samples.count
        let minSamples = Int(minChunkDurationSeconds * Float(sampleRate))
        func padded(_ range: Range<Int>) -> [Float] {
            var chunk = Array(samples[range])
            if chunk.count < minSamples {
                chunk.append(contentsOf: repeatElement(0, count: minSamples - chunk.count))
            }
            return chunk
        }
        if Float(totalSamples) / Float(sampleRate) <= chunkDurationSeconds {
            return [PaddedChunk(samples: padded(0..<totalSamples), offsetSeconds: 0)]
        }
        let maxChunkSamples = Int(chunkDurationSeconds * Float(sampleRate))
        let searchSamples = Int(searchExpandSeconds * Float(sampleRate))
        let minWindowSamples = Int(minWindowMilliseconds * Float(sampleRate) / 1000.0)
        var chunks: [PaddedChunk] = []
        var startSample = 0
        while startSample < totalSamples {
            let endSample = min(startSample + maxChunkSamples, totalSamples)
            let offsetSeconds = Double(Float(startSample) / Float(sampleRate))
            if endSample >= totalSamples {
                chunks.append(PaddedChunk(samples: padded(startSample..<totalSamples), offsetSeconds: offsetSeconds))
                break
            }
            let searchStart = max(startSample, endSample - searchSamples)
            let searchEnd = min(totalSamples, endSample + searchSamples)
            var cutSample = endSample
            if searchEnd - searchStart > minWindowSamples {
                let energyCount = searchEnd - searchStart - minWindowSamples + 1
                var windowSum: Float = 0
                for index in searchStart..<(searchStart + minWindowSamples) {
                    windowSum += samples[index] * samples[index]
                }
                let inverseWindow = 1.0 / Float(minWindowSamples)
                var minimumEnergy = windowSum * inverseWindow
                var minimumIndex = 0
                for offset in 1..<energyCount {
                    let leaving = samples[searchStart + offset - 1]
                    let entering = samples[searchStart + offset + minWindowSamples - 1]
                    windowSum += entering * entering - leaving * leaving
                    let energy = windowSum * inverseWindow
                    if energy < minimumEnergy {
                        minimumEnergy = energy
                        minimumIndex = offset
                    }
                }
                cutSample = searchStart + minimumIndex + minWindowSamples / 2
            }
            cutSample = max(cutSample, startSample + sampleRate)
            let actualEnd = min(cutSample, totalSamples)
            chunks.append(PaddedChunk(samples: padded(startSample..<actualEnd), offsetSeconds: offsetSeconds))
            startSample = cutSample
        }
        return chunks
    }

    static func whisperMaxNewTokens(stageMaxTokens: Int, languageHint: String?) -> Int {
        let promptTokenCount = languageHint == nil ? 3 : 4
        return max(1, min(stageMaxTokens, whisperDecoderPositions - promptTokenCount - 1))
    }

    /// MOSS timestamps restart at each chunk; shift them onto the recording's
    /// timeline with MLXAudio's tag rule: a bracketed number of at most 24
    /// characters, comma decimals allowed.
    static func offsetMossTimestamps(_ text: String, by offsetSeconds: Double) -> String {
        guard offsetSeconds != 0 else { return text }
        func offsetTag(_ tag: String) -> String {
            guard let value = Double(tag.dropFirst().dropLast().replacingOccurrences(of: ",", with: ".")) else {
                return tag
            }
            return String(format: "[%.2f]", locale: Locale(identifier: "en_US_POSIX"), value + offsetSeconds)
        }
        var output = ""
        var bufferedTag = ""
        var isBufferingTag = false
        for character in text {
            if isBufferingTag {
                bufferedTag.append(character)
                if character == "]" {
                    output += offsetTag(bufferedTag)
                    bufferedTag = ""
                    isBufferingTag = false
                } else if bufferedTag.count > 24 {
                    output += bufferedTag
                    bufferedTag = ""
                    isBufferingTag = false
                }
            } else if character == "[" {
                bufferedTag = "["
                isBufferingTag = true
            } else {
                output.append(character)
            }
        }
        return output + bufferedTag
    }
}

nonisolated struct OmniTranscriptSegment: Sendable, Equatable {
    let text: String
    let startSeconds: Double
    let endSeconds: Double
    let speakerID: String?
}

/// MOSS's `[start][Sxx]text[end]` markup, parsed with the rule MLXAudio used.
nonisolated enum OmniMossSegments {
    static func parse(_ text: String, fallbackEndSeconds: Double) -> [OmniTranscriptSegment] {
        let pattern = try! NSRegularExpression(
            pattern: #"\[(\d+(?:[\.,]\d+)?)\]\[(S\d+)\](.*?)\[(\d+(?:[\.,]\d+)?)\]"#,
            options: [.dotMatchesLineSeparators]
        )
        let source = text as NSString
        func seconds(_ range: NSRange) -> Double? {
            Double(source.substring(with: range).replacingOccurrences(of: ",", with: "."))
        }
        var segments: [OmniTranscriptSegment] = []
        for match in pattern.matches(in: text, range: NSRange(location: 0, length: source.length)) {
            guard let start = seconds(match.range(at: 1)),
                  let end = seconds(match.range(at: 4)),
                  end >= start
            else { continue }
            let segmentText = source.substring(with: match.range(at: 3))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !segmentText.isEmpty else { continue }
            segments.append(OmniTranscriptSegment(
                text: segmentText,
                startSeconds: start,
                endSeconds: end,
                speakerID: source.substring(with: match.range(at: 2))
            ))
        }
        if segments.isEmpty {
            return [OmniTranscriptSegment(text: text, startSeconds: 0, endSeconds: max(fallbackEndSeconds, 0), speakerID: nil)]
        }
        return segments
    }
}
