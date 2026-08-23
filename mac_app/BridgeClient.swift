import Foundation

struct VoiceEntry: Codable, Identifiable {
    let path: String
    let name: String
    let language: String?
    let sample_rate: Int?
    let valid: Bool
    let errors: [String]
    var id: String { path }
}

struct JobSnapshot: Codable {
    let status: String
    let command: String?
    let step: Int?
    let package: String?
}

struct BridgeSnapshot: Codable {
    let job: JobSnapshot
    let voices: [VoiceEntry]
}

enum BridgeClient {
    static func fetch(job: String, voices: String, workingDirectory: String) throws -> BridgeSnapshot {
        let process = Process()
        let python = ProcessInfo.processInfo.environment["PVS_PYTHON"] ?? "/usr/bin/env"
        process.executableURL = URL(fileURLWithPath: python)
        process.arguments = python == "/usr/bin/env"
            ? ["python3", "-m", "mac_voice", "bridge-status", "--job", job, "--voices", voices]
            : ["-m", "mac_voice", "bridge-status", "--job", job, "--voices", voices]
        process.currentDirectoryURL = URL(fileURLWithPath: workingDirectory)
        let pipe = Pipe()
        process.standardOutput = pipe
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { throw NSError(domain: "PersonalVoiceStudio", code: Int(process.terminationStatus)) }
        return try JSONDecoder().decode(BridgeSnapshot.self, from: pipe.fileHandleForReading.readDataToEndOfFile())
    }

    static func synthesize(voice: String, text: String, output: String, modelDirectory: String, workingDirectory: String) throws {
        let process = Process()
        let python = ProcessInfo.processInfo.environment["PVS_PYTHON"] ?? "/usr/bin/env"
        process.executableURL = URL(fileURLWithPath: python)
        let args = ["-m", "mac_voice", "synth", "--voice", voice, "--text", text, "--output", output, "--model-dir", modelDirectory]
        process.arguments = python == "/usr/bin/env" ? ["python3"] + args : args
        process.currentDirectoryURL = URL(fileURLWithPath: workingDirectory)
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { throw NSError(domain: "PersonalVoiceStudio", code: Int(process.terminationStatus)) }
    }
}
