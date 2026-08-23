import SwiftUI

@main
struct PersonalVoiceStudioApp: App {
    private let projectDirectory = ProcessInfo.processInfo.environment["PVS_PROJECT_DIR"] ?? FileManager.default.currentDirectoryPath

    var body: some Scene {
        WindowGroup {
            ContentView(
                projectDirectory: projectDirectory,
                jobPath: "artifacts/runs/package_job/job.json",
                voicesPath: "artifacts/voices"
            )
        }
    }
}
