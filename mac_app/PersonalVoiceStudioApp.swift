import SwiftUI

@main
struct PersonalVoiceStudioApp: App {
    private let projectDirectory = ProcessInfo.processInfo.environment["PVS_PROJECT_DIR"] ?? FileManager.default.currentDirectoryPath
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
