import SwiftUI

struct ContentView: View {
    @State private var snapshot: BridgeSnapshot?
    @State private var errorMessage: String?
    @StateObject private var recorder = Recorder()
    let projectDirectory: String
    let jobPath: String
    let voicesPath: String

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Personal Voice Studio").font(.title)
            if let snapshot {
                Text("작업 상태: \(snapshot.job.status)")
                Text("Voice Package: \(snapshot.voices.filter(\.valid).count)개 사용 가능")
                List(snapshot.voices) { voice in
                    HStack { Text(voice.name); Spacer(); Text(voice.valid ? "사용 가능" : "오류") }
                }
            } else if let errorMessage {
                Text(errorMessage).foregroundStyle(.red)
            } else {
                ProgressView("상태를 불러오는 중")
            }
            Button("새로고침") { refresh() }
            Button(recorder.isRecording ? "녹음 중지" : "마이크 녹음 시작") {
                if recorder.isRecording { recorder.stop() }
                else {
                    let url = URL(fileURLWithPath: projectDirectory).appendingPathComponent("artifacts/ui_recording.wav")
                    recorder.requestMicrophonePermission { granted in
                        DispatchQueue.main.async {
                            guard granted else { errorMessage = "마이크 권한이 필요합니다."; return }
                            do { try recorder.start(to: url) } catch { errorMessage = "녹음 오류: \(error.localizedDescription)" }
                        }
                    }
                }
            }
            if let output = recorder.lastOutput { Text("녹음 파일: \(output.path)").font(.caption) }
        }
        .padding()
        .task { refresh() }
    }

    private func refresh() {
        do { snapshot = try BridgeClient.fetch(job: jobPath, voices: voicesPath, workingDirectory: projectDirectory); errorMessage = nil }
        catch { errorMessage = "Python bridge 오류: \(error.localizedDescription)" }
    }
}
