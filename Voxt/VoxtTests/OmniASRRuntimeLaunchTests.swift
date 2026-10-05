// OmniASRRuntimeLaunchTests.swift
// Covers how the Omni runtime starts its supervisor process.

import XCTest
@testable import Voxt

final class OmniASRRuntimeLaunchTests: XCTestCase {
    func testSupervisorEnvironmentDropsInjectedDynamicLoaderVariables() {
        let inherited = [
            "PATH": "/usr/bin",
            "HOME": "/Users/someone",
            "DYLD_INSERT_LIBRARIES": "/Xcode/libXCTestBundleInject.dylib",
            "DYLD_LIBRARY_PATH": "/Xcode/usr/lib",
            "DYLD_FRAMEWORK_PATH": "/Xcode/Frameworks",
            "__XPC_DYLD_LIBRARY_PATH": "/Xcode/usr/lib",
            "PYTHONPATH": "/somewhere/else",
        ]

        let environment = OmniASRRuntime.supervisorEnvironment(
            inheriting: inherited,
            backendDirectory: URL(fileURLWithPath: "/backend", isDirectory: true)
        )

        XCTAssertEqual(environment["PATH"], "/usr/bin")
        XCTAssertEqual(environment["HOME"], "/Users/someone")
        XCTAssertEqual(environment["PYTHONPATH"], "/backend")
        XCTAssertEqual(environment["PYTHONUNBUFFERED"], "1")
        XCTAssertEqual(
            environment.keys.filter { $0.hasPrefix("DYLD_") || $0.hasPrefix("__XPC_DYLD_") },
            []
        )
    }

    func testDiagnosticTailKeepsOnlyTheEndOfLongOutput() {
        let tail = OmniDiagnosticTail(limit: 8)
        tail.append(Data("0123456789".utf8))
        tail.append(Data("ab".utf8))

        XCTAssertEqual(tail.text, "456789ab")
    }
}
