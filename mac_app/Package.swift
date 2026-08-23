// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "PersonalVoiceStudio",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "PersonalVoiceStudio", targets: ["PersonalVoiceStudio"])],
    targets: [.executableTarget(name: "PersonalVoiceStudio", path: ".", exclude: ["Info.plist"])]
)
