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
    let error: String?
    let metrics: [String: Double]?
}

struct RunEntry: Codable, Identifiable {
    let name: String
    let path: String
    let job_status: String?
    let checkpoint: String?
    let metrics: String?
    let last_metrics: [String: Double]?
    var id: String { path }
}

struct MPSSnapshot: Codable {
    let status: String
    let torch: String?
    let macos: String
    let tensor_probe: Bool
    let action: String
}

struct BridgeSnapshot: Codable {
    let job: JobSnapshot
    let voices: [VoiceEntry]
    let runs: [RunEntry]
    let mps: MPSSnapshot
}

enum BridgeClient {
    private static func configure(_ process: Process, workingDirectory: String) {
        let environment = ProcessInfo.processInfo.environment
        let bundledPython = "/opt/homebrew/Caskroom/miniconda/base/envs/cosyvoice/bin/python"
        let configuredPython = environment["PVS_PYTHON"]
        let python = configuredPython ?? (FileManager.default.isExecutableFile(atPath: bundledPython) ? bundledPython : "/usr/bin/env")
        process.executableURL = URL(fileURLWithPath: python)
        process.currentDirectoryURL = URL(fileURLWithPath: workingDirectory)
        var childEnvironment = environment
        let existingPythonPath = environment["PYTHONPATH"]
        childEnvironment["PYTHONPATH"] = existingPythonPath.map { "\(workingDirectory):\($0)" } ?? workingDirectory
        process.environment = childEnvironment
    }

    private static func failure(_ process: Process, stderr: Pipe) -> NSError {
        let detail = String(data: stderr.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
        let lines = detail?.split(separator: "\n").map(String.init) ?? []
        let important = lines.filter { $0.contains("[ERROR]") || $0.contains("[BLOCKED]") || $0.contains("Traceback") || $0.contains("RuntimeError") || $0.contains("ValueError") }
        let selected = important.isEmpty ? Array(lines.suffix(12)) : Array(important.suffix(8))
        let message = selected.isEmpty ? "Python bridge exited with status \(process.terminationStatus)" : selected.joined(separator: "\n")
        return NSError(domain: "PersonalVoiceStudio", code: Int(process.terminationStatus), userInfo: [NSLocalizedDescriptionKey: message])
    }

    static func fetch(job: String, voices: String, workingDirectory: String) throws -> BridgeSnapshot {
        let process = Process()
        configure(process, workingDirectory: workingDirectory)
        let python = process.executableURL?.path ?? "/usr/bin/env"
        let args = ["-m", "mac_voice", "bridge-status", "--job", job, "--voices", voices]
        process.arguments = python == "/usr/bin/env" ? ["python3"] + args : args
        let pipe = Pipe()
        let stderr = Pipe()
        process.standardOutput = pipe
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { throw failure(process, stderr: stderr) }
        return try JSONDecoder().decode(BridgeSnapshot.self, from: pipe.fileHandleForReading.readDataToEndOfFile())
    }

    static func synthesize(voice: String, text: String, output: String, modelDirectory: String, workingDirectory: String, device: String = "cpu") throws {
        let process = Process()
        configure(process, workingDirectory: workingDirectory)
        let python = process.executableURL?.path ?? "/usr/bin/env"
        let command = device == "mps" ? "mps-synth" : "synth"
        let args = ["-m", "mac_voice", command, "--voice", voice, "--text", text, "--output", output, "--model-dir", modelDirectory]
        process.arguments = python == "/usr/bin/env" ? ["python3"] + args : args
        let stderr = Pipe()
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { throw failure(process, stderr: stderr) }
    }

    static func cancelJob(job: String, workingDirectory: String) throws {
        let process = Process()
        configure(process, workingDirectory: workingDirectory)
        let python = process.executableURL?.path ?? "/usr/bin/env"
        let args = ["-m", "mac_voice", "job-update", "--job", job, "--status", "cancelled"]
        process.arguments = python == "/usr/bin/env" ? ["python3"] + args : args
        let stderr = Pipe()
        process.standardError = stderr
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { throw failure(process, stderr: stderr) }
    }

    static func createVoiceModel(dataset: String, name: String, modelDirectory: String, workingDirectory: String) throws {
        let process = Process()
        configure(process, workingDirectory: workingDirectory)
        let python = process.executableURL?.path ?? "/usr/bin/env"
        let args = ["-m", "mac_voice", "create-voice", "--dataset", dataset, "--name", name, "--model-dir", modelDirectory]
        process.arguments = python == "/usr/bin/env" ? ["python3"] + args : args
        let output = Pipe()
        process.standardOutput = output
        process.standardError = output
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { throw failure(process, stderr: output) }
    }
}
