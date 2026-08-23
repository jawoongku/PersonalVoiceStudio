import SwiftUI

@main
struct PersonalVoiceStudioApp: App {
    private let projectDirectory: String = {
        if let configured = ProcessInfo.processInfo.environment["PVS_PROJECT_DIR"] {
            return configured
        }
        let bundle = Bundle.main.bundleURL
        let candidate = bundle.deletingLastPathComponent().deletingLastPathComponent()
        let marker = candidate.appendingPathComponent("mac_voice")
        return FileManager.default.fileExists(atPath: marker.path) ? candidate.path : FileManager.default.currentDirectoryPath
    }()
    private let jobPath = ProcessInfo.processInfo.environment["PVS_JOB_PATH"] ?? "artifacts/runs/package_job/job.json"
    private let voicesPath = ProcessInfo.processInfo.environment["PVS_VOICES_PATH"] ?? "artifacts/voices"

    var body: some Scene {
        WindowGroup {
            ContentView(
                projectDirectory: projectDirectory,
                jobPath: jobPath,
                voicesPath: voicesPath
            )
        }
    }
}
