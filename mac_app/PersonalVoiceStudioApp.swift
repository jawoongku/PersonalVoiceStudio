import SwiftUI

@main
struct PersonalVoiceStudioApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView(
                projectDirectory: FileManager.default.currentDirectoryPath,
                jobPath: "artifacts/runs/package_job/job.json",
                voicesPath: "artifacts/voices"
            )
        }
    }
}
