import SwiftUI
import AppKit

struct ContentView: View {
    private let refreshTimer = Timer.publish(every: 10, on: .main, in: .common).autoconnect()
    @State private var snapshot: BridgeSnapshot?
    @State private var errorMessage: String?
    @State private var ttsText = ""
    @State private var ttsOutput: URL?
    @State private var isSynthesizing = false
    @State private var isCancelling = false
    @State private var selectedVoicePath: String?
    @State private var ttsDevice = "cpu"
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
                if let step = snapshot.job.step { Text("현재 step: \(step)").font(.caption) }
                if let metrics = snapshot.job.metrics {
                    if let train = metrics["train_loss"] { Text("train loss: \(train)").font(.caption) }
                    if let dev = metrics["val_loss"] { Text("dev loss: \(dev)").font(.caption) }
                }
                if let jobError = snapshot.job.error, !jobError.isEmpty {
                    Text(jobError).font(.caption).foregroundStyle(.red)
                }
                if snapshot.job.status == "running" {
                    Button(isCancelling ? "취소 요청 중..." : "학습 취소") {
                        isCancelling = true
                        DispatchQueue.global(qos: .utility).async {
                            do {
                                try BridgeClient.cancelJob(job: jobPath, workingDirectory: projectDirectory)
                                DispatchQueue.main.async { isCancelling = false; refresh() }
                            } catch {
                                DispatchQueue.main.async { errorMessage = "취소 오류: \(error.localizedDescription)"; isCancelling = false }
                            }
                        }
                    }
                    .disabled(isCancelling)
                }
                HStack(alignment: .top) {
                    Image(systemName: snapshot.mps.tensor_probe ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                        .foregroundStyle(snapshot.mps.tensor_probe ? .green : .orange)
                    VStack(alignment: .leading) {
                        Text("MPS: \(snapshot.mps.status)").font(.headline)
                        Text(snapshot.mps.action).font(.caption).foregroundStyle(.secondary)
                    }
                }
                Text("Voice Package: \(snapshot.voices.filter(\.valid).count)개 사용 가능")
                Picker("TTS Voice", selection: $selectedVoicePath) {
                    ForEach(snapshot.voices.filter(\.valid)) { voice in
                        Text(voice.name).tag(Optional(voice.path))
                    }
                }
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
                Picker("TTS 실행 장치", selection: $ttsDevice) {
                    Text("CPU").tag("cpu")
                    Text("MPS").tag("mps").disabled(!snapshot.mps.tensor_probe)
                }
                .pickerStyle(.segmented)
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
                Button("녹음 파일을 Finder에서 보기") { NSWorkspace.shared.activateFileViewerSelecting([output]) }
            }
            TextField("TTS 텍스트", text: $ttsText)
            Button(isSynthesizing ? "TTS 생성 중..." : "Voice Package TTS 생성") {
                guard !ttsText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { errorMessage = "TTS 텍스트를 입력해 주세요."; return }
                guard let voicePath = selectedVoicePath ?? snapshot?.voices.first(where: { $0.valid })?.path else { errorMessage = "사용 가능한 Voice Package가 없습니다."; return }
                let output = URL(fileURLWithPath: projectDirectory).appendingPathComponent("artifacts/swiftui_tts.wav")
                let device = ttsDevice
                isSynthesizing = true
                let modelDirectory = ProcessInfo.processInfo.environment["PVS_MODEL_DIR"] ?? "/Users/jawoongku/Models/Fun-CosyVoice3-0.5B"
                DispatchQueue.global(qos: .userInitiated).async {
                    do {
                        try BridgeClient.synthesize(voice: voicePath, text: ttsText, output: output.path, modelDirectory: modelDirectory, workingDirectory: projectDirectory, device: device)
                        DispatchQueue.main.async { ttsOutput = output; isSynthesizing = false }
                    } catch {
                        DispatchQueue.main.async { errorMessage = "TTS 오류: \(error.localizedDescription)"; isSynthesizing = false }
                    }
                }
            }
            .disabled(isSynthesizing)
            if let ttsOutput {
                Button("생성 음성 재생") { do { try audioPlayer.play(url: ttsOutput) } catch { errorMessage = "재생 오류: \(error.localizedDescription)" } }
                Button("생성 파일을 Finder에서 보기") { NSWorkspace.shared.activateFileViewerSelecting([ttsOutput]) }
            }
        }
        .padding()
        .task { refresh() }
        .onReceive(refreshTimer) { _ in refresh() }
    }

    private func refresh() {
        DispatchQueue.global(qos: .utility).async {
            do {
                let value = try BridgeClient.fetch(job: jobPath, voices: voicesPath, workingDirectory: projectDirectory)
                DispatchQueue.main.async {
                    snapshot = value
                    if let configured = ProcessInfo.processInfo.environment["PVS_TTS_DEVICE"], configured == "mps", value.mps.tensor_probe {
                        ttsDevice = configured
                    }
                    if selectedVoicePath == nil { selectedVoicePath = value.voices.first(where: { $0.valid })?.path }
                    errorMessage = nil
                }
            } catch {
                DispatchQueue.main.async { errorMessage = "Python bridge 오류: \(error.localizedDescription)" }
            }
        }
    }
}
