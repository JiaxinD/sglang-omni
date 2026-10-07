// OmniASRRuntimeLaunchTests.swift
// Covers how the Omni runtime starts its native server process.

import XCTest
@testable import Voxt

final class OmniASRRuntimeLaunchTests: XCTestCase {
    func testRuntimeEnvironmentDropsInjectedDynamicLoaderVariables() {
        let inherited = [
            "PATH": "/usr/bin",
            "HOME": "/Users/someone",
            "DYLD_INSERT_LIBRARIES": "/Xcode/libXCTestBundleInject.dylib",
            "DYLD_LIBRARY_PATH": "/Xcode/usr/lib",
            "DYLD_FRAMEWORK_PATH": "/Xcode/Frameworks",
            "__XPC_DYLD_LIBRARY_PATH": "/Xcode/usr/lib",
        ]

        let environment = OmniASRRuntime.runtimeEnvironment(inheriting: inherited)

        XCTAssertEqual(environment, ["PATH": "/usr/bin", "HOME": "/Users/someone"])
    }

    func testLaunchSettingsNeedTheOmniBackendAndARuntimePath() {
        XCTAssertNil(OmniASRBackend.LaunchSettings(environment: ["VOXT_OMNI_RUNTIME": "/opt/qwen3_asr_server"]))
        XCTAssertNil(OmniASRBackend.LaunchSettings(environment: ["VOXT_ASR_BACKEND": "omni"]))
        XCTAssertNil(OmniASRBackend.LaunchSettings(environment: ["VOXT_ASR_BACKEND": "omni", "VOXT_OMNI_RUNTIME": ""]))
        let settings = OmniASRBackend.LaunchSettings(environment: [
            "VOXT_ASR_BACKEND": "omni",
            "VOXT_OMNI_RUNTIME": "/opt/qwen3_asr_server",
        ])
        XCTAssertEqual(settings?.runtimeExecutable, URL(fileURLWithPath: "/opt/qwen3_asr_server"))
    }

    /// The runtime binary is started directly in supervised mode, not through Python.
    func testLaunchRunsTheRuntimeInSupervisedMode() async throws {
        let scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("voxt-omni-launch-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        let runtimeExecutable = scratch.appendingPathComponent("qwen3_asr_server")
        let arguments = scratch.appendingPathComponent("arguments")
        let environment = scratch.appendingPathComponent("environment")
        // Records how it was started, then reports a failed start.
        let script = """
        #!/bin/sh
        printf '%s\\n' "$0" "$@" > '\(arguments.path)'
        env > '\(environment.path)'
        echo '{"event": "failed", "reason": "recorded"}'
        """
        try script.write(to: runtimeExecutable, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: runtimeExecutable.path)
        let modelDirectory = scratch.appendingPathComponent("model", isDirectory: true)
        let derivedRoot = scratch.appendingPathComponent("derived", isDirectory: true)
        let configuration = OmniBackendConfiguration(
            runtimeExecutable: runtimeExecutable,
            derivedRoot: derivedRoot,
            startupTimeoutSeconds: 42
        )
        let runtime = OmniASRRuntime(kind: .qwen3ASR, modelDirectory: modelDirectory, configuration: configuration)

        do {
            _ = try await runtime.prepare()
            XCTFail("the recording runtime never reports ready")
        } catch {
            XCTAssertEqual(error as? OmniASRRuntimeError, .launchFailed("recorded"))
        }
        await runtime.retire()

        let commandLine = try String(contentsOf: arguments, encoding: .utf8)
            .split(separator: "\n", omittingEmptySubsequences: false)
            .dropLast()
            .map(String.init)
        XCTAssertEqual(commandLine, [
            runtimeExecutable.path,
            "--supervised",
            "--model-kind", "qwen3_asr",
            "--model-directory", modelDirectory.path,
            "--derived-root", derivedRoot.path,
            "--startup-timeout-s", "42.0",
        ])
        let variables = try String(contentsOf: environment, encoding: .utf8)
            .split(separator: "\n")
            .compactMap { $0.split(separator: "=", maxSplits: 1).first.map(String.init) }
        XCTAssertFalse(variables.contains("PYTHONPATH"))
        XCTAssertFalse(variables.contains("PYTHONUNBUFFERED"))
        XCTAssertEqual(variables.filter { $0.hasPrefix("DYLD_") || $0.hasPrefix("__XPC_DYLD_") }, [])
    }

    func testDiagnosticTailKeepsOnlyTheEndOfLongOutput() {
        let tail = OmniDiagnosticTail(limit: 8)
        tail.append(Data("0123456789".utf8))
        tail.append(Data("ab".utf8))

        XCTAssertEqual(tail.text, "456789ab")
    }
}

@MainActor
final class OmniModelManagerRecoveryTests: XCTestCase {
    private actor LoadCounter {
        private(set) var value = 0

        func increment() {
            value += 1
        }
    }

    /// A server that crashed or was killed must not keep failing every request
    /// until the idle unload: the next load replaces the runtime.
    func testALoadedOmniRuntimeThatStoppedServingIsReplacedOnTheNextLoad() async throws {
        let scratch = FileManager.default.temporaryDirectory
        let configuration = OmniBackendConfiguration(
            runtimeExecutable: URL(fileURLWithPath: "/usr/bin/false"),
            derivedRoot: scratch
        )
        let loads = LoadCounter()
        let manager = MLXModelManager(modelRepo: "mlx-community/Qwen3-ASR-0.6B-4bit") { _ in
            await loads.increment()
            // Never launched, so never serving: the same answer a dead server gives.
            let runtime = OmniASRRuntime(kind: .qwen3ASR, modelDirectory: scratch, configuration: configuration)
            return MLXLoadedModelBox(loaded: .omni(runtime))
        }

        let first = try await manager.loadModel()
        let second = try await manager.loadModel()

        XCTAssertFalse(first.omniRuntime === second.omniRuntime)
        let loadCount = await loads.value
        XCTAssertEqual(loadCount, 2)
        await manager.shutdownForApplicationTermination()
    }
}
