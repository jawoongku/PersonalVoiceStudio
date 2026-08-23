import SwiftUI

struct ContentView: View {
    private let refreshTimer = Timer.publish(every: 10, on: .main, in: .common).autoconnect()
    @State private var snapshot: BridgeSnapshot?
    @State private var errorMessage: String?
    @State private var ttsText = ""
    @State private var ttsOutput: URL?
    @State private var isSynthesizing = false
    @StateObject private var recorder = Recorder()
    @StateObject private var audioPlayer = AudioPlayer()
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
                Text("학습 runs: \(snapshot.runs.count)")
                List(snapshot.runs) { run in
                    VStack(alignment: .leading) {
                        HStack { Text(run.name); Spacer(); Text(run.job_status ?? "unknown") }
                        if let checkpoint = run.checkpoint { Text(checkpoint).font(.caption).foregroundStyle(.secondary) }
                        if let metrics = run.metrics { Text(metrics).font(.caption2).foregroundStyle(.secondary) }
                        if let values = run.last_metrics, let train = values["train_loss"] { Text("train loss: \(train)").font(.caption2) }
                        if let values = run.last_metrics, let val = values["val_loss"] { Text("val loss: \(val)").font(.caption2) }
                        if let values = run.last_metrics, let lr = values["learning_rate"] { Text("learning rate: \(lr)").font(.caption2) }
                    }
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
            if let output = recorder.lastOutput {
                Button(audioPlayer.isPlaying ? "재생 중지" : "녹음 재생") {
                    if audioPlayer.isPlaying { audioPlayer.stop() }
                    else { do { try audioPlayer.play(url: output) } catch { errorMessage = "재생 오류: \(error.localizedDescription)" } }
                }
            }
            TextField("TTS 텍스트", text: $ttsText)
            Button(isSynthesizing ? "TTS 생성 중..." : "Voice Package TTS 생성") {
                guard let voice = snapshot?.voices.first(where: { $0.valid }) else { errorMessage = "사용 가능한 Voice Package가 없습니다."; return }
                let output = URL(fileURLWithPath: projectDirectory).appendingPathComponent("artifacts/swiftui_tts.wav")
                isSynthesizing = true
                let modelDirectory = ProcessInfo.processInfo.environment["PVS_MODEL_DIR"] ?? "/Users/jawoongku/Models/Fun-CosyVoice3-0.5B"
                DispatchQueue.global(qos: .userInitiated).async {
                    do {
                        try BridgeClient.synthesize(voice: voice.path, text: ttsText, output: output.path, modelDirectory: modelDirectory, workingDirectory: projectDirectory)
                        DispatchQueue.main.async { ttsOutput = output; isSynthesizing = false }
                    } catch {
                        DispatchQueue.main.async { errorMessage = "TTS 오류: \(error.localizedDescription)"; isSynthesizing = false }
                    }
                }
            }
            .disabled(isSynthesizing)
            if let ttsOutput {
                Button("생성 음성 재생") { do { try audioPlayer.play(url: ttsOutput) } catch { errorMessage = "재생 오류: \(error.localizedDescription)" } }
            }
        }
        .padding()
        .task { refresh() }
        .onReceive(refreshTimer) { _ in refresh() }
    }

    private func refresh() {
        do { snapshot = try BridgeClient.fetch(job: jobPath, voices: voicesPath, workingDirectory: projectDirectory); errorMessage = nil }
        catch { errorMessage = "Python bridge 오류: \(error.localizedDescription)" }
    }
}
