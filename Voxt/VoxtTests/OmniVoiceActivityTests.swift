// OmniVoiceActivityTests.swift
// Covers the wire formats of Silero VAD on the native runtime.

import XCTest
@testable import Voxt

final class OmniVoiceActivityTests: XCTestCase {
    func testStreamSendsFloat32LittleEndianSamples() {
        let data = OmniVoiceActivityStream.float32LittleEndian([0, 1, -0.5])

        XCTAssertEqual([UInt8](data), [
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x80, 0x3F,
            0x00, 0x00, 0x00, 0xBF,
        ])
    }

    func testStreamReplyCarriesTheLastChunkProbabilityOrNull() throws {
        XCTAssertEqual(try OmniVoiceActivityStream.probability(fromReply: #"{"probability":0.875}"#), 0.875)
        XCTAssertNil(try OmniVoiceActivityStream.probability(fromReply: #"{"probability":null}"#))
        XCTAssertThrowsError(try OmniVoiceActivityStream.probability(fromReply: #"{"error":"x"}"#))
        XCTAssertThrowsError(try OmniVoiceActivityStream.probability(fromReply: #"{"probability":"high"}"#))
    }

    func testSpeechTimestampsAreSampleRanges() throws {
        let response = Data(#"{"sample_rate":16000,"timestamps":[{"start":0,"end":512},{"start":1024,"end":4096}]}"#.utf8)

        XCTAssertEqual(try OmniVoiceActivityRequests.ranges(fromResponse: response), [0 ..< 512, 1024 ..< 4096])
        XCTAssertThrowsError(try OmniVoiceActivityRequests.ranges(fromResponse: Data(#"{"timestamps":[{"start":9,"end":3}]}"#.utf8)))
        XCTAssertThrowsError(try OmniVoiceActivityRequests.ranges(fromResponse: Data(#"{"detail":"x"}"#.utf8)))
    }
}
