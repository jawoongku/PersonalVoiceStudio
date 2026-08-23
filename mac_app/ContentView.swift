import SwiftUI
import AppKit
import AVFoundation

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
    @State private var trainingSentence = "오늘 아침에는 평소보다 조금 일찍 일어났습니다."
    @State private var sentenceIndex = 0
    @State private var recordingValidation: String?
    @State private var selectedTab = 0
    @StateObject private var recorder = Recorder()
    @StateObject private var audioPlayer = AudioPlayer()
    let projectDirectory: String
    let jobPath: String
    let voicesPath: String

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Personal Voice Studio").font(.title)
            Picker("화면", selection: $selectedTab) {
                Text("모델 생성").tag(0)
                Text("TTS 생성").tag(1)
            }
            .pickerStyle(.segmented)
            if selectedTab == 0 {
            if let snapshot {
                Text("작업 상태: \(snapshot.job.status)")
                if let step = snapshot.job.step { Text("현재 step: \(step)").font(.caption) }
                if let metrics = snapshot.job.metrics {
                    if let train = metrics["train_loss"] { Text("train loss: \(train)").font(.caption) }
                    if let dev = metrics["val_loss"] { Text("dev loss: \(dev)").font(.caption) }
                    if let memory = metrics["mps_memory"] { Text(String(format: "MPS memory: %.1f MB", memory / 1_048_576.0)).font(.caption) }
                    if let driver = metrics["driver_memory"] { Text(String(format: "MPS driver: %.1f MB", driver / 1_048_576.0)).font(.caption) }
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
            Divider()
            Text("학습용 문장 녹음").font(.headline)
            Text("아래 문장을 그대로 읽고 녹음한 뒤, 검증을 통과하면 학습 데이터로 저장할 수 있습니다.")
                .font(.caption).foregroundStyle(.secondary)
            Text(trainingSentence).font(.title3).padding(.vertical, 4)
            HStack {
                Button("다음 문장") { nextTrainingSentence() }
                Button(recorder.isRecording ? "녹음 중지" : "이 문장 녹음") {
                    if recorder.isRecording { recorder.stop(); validateRecording() }
                    else {
                        let raw = URL(fileURLWithPath: projectDirectory).appendingPathComponent("data/my_voice/raw", isDirectory: true)
                        try? FileManager.default.createDirectory(at: raw, withIntermediateDirectories: true)
                        let url = raw.appendingPathComponent(String(format: "ui_%@.wav", UUID().uuidString))
                        recorder.requestMicrophonePermission { granted in
                            DispatchQueue.main.async {
                                guard granted else { errorMessage = "마이크 권한이 필요합니다."; return }
                                do { try recorder.start(to: url); recordingValidation = "녹음 중입니다..." }
                                catch { errorMessage = "녹음 오류: \(error.localizedDescription)" }
                            }
                        }
                    }
                }
            }
            WaveformView(samples: recorder.waveform, active: recorder.isRecording)
                .frame(height: 72)
                .background(Color.black.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
            if let validation = recordingValidation { Text(validation).font(.caption).foregroundStyle(.secondary) }
            if let output = recorder.lastOutput, !recorder.isRecording {
                HStack {
                    Button("검증") { validateRecording() }
                    Button("검증 통과 파일 저장") { saveTrainingRecording(output) }
                        .disabled(!(recordingValidation?.hasPrefix("사용 가능") ?? false))
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
            } else {
                Text("Voice Package TTS").font(.headline)
                Text("생성된 음성 모델을 선택하고 문장을 입력해 음성을 만듭니다.")
                    .font(.caption).foregroundStyle(.secondary)
                if let snapshot {
                    Picker("Voice Package", selection: $selectedVoicePath) {
                        ForEach(snapshot.voices.filter(\.valid)) { voice in
                            Text(voice.name).tag(Optional(voice.path))
                        }
                    }
                    Picker("실행 장치", selection: $ttsDevice) {
                        Text("CPU").tag("cpu")
                        Text("MPS").tag("mps").disabled(!snapshot.mps.tensor_probe)
                    }
                    .pickerStyle(.segmented)
                }
            }
            if selectedTab == 1 {
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

    private func nextTrainingSentence() {
        let sentences = [
            "오늘 아침에는 평소보다 조금 일찍 일어났습니다.",
            "창문을 열어 보니 바람이 생각보다 시원하게 불고 있습니다.",
            "아직 결정하지 못한 부분은 조금 더 고민해 볼 생각입니다.",
            "중요한 것은 얼마나 빨리 하느냐가 아니라 제대로 하는 것입니다.",
            "그런데 이 방법이 가장 좋은 방법일까요?",
            "그러면 다른 방법을 한번 찾아보는 게 어떨까요?"
        ]
        sentenceIndex = (sentenceIndex + 1) % sentences.count
        trainingSentence = sentences[sentenceIndex]
        recordingValidation = nil
    }

    private func validateRecording() {
        guard let output = recorder.lastOutput else { recordingValidation = "녹음 파일이 없습니다."; return }
        do {
            let audio = try AVAudioFile(forReading: output)
            let seconds = Double(audio.length) / audio.processingFormat.sampleRate
            let channels = audio.processingFormat.channelCount
            if seconds <= 0 || seconds > 30 { recordingValidation = "재녹음 필요: 길이는 0초 초과 30초 이하여야 합니다." }
            else if channels != 1 { recordingValidation = "재녹음 검토: mono 녹음을 권장합니다. (현재 \(channels)ch)" }
            else { recordingValidation = String(format: "사용 가능: %.2f초, %.0fHz, mono. 문장 일치 여부를 확인한 뒤 저장하세요.", seconds, audio.processingFormat.sampleRate) }
        } catch { recordingValidation = "검증 오류: \(error.localizedDescription)" }
    }

    private func saveTrainingRecording(_ output: URL) {
        let transcript = URL(fileURLWithPath: projectDirectory).appendingPathComponent("data/my_voice/transcripts.csv")
        do {
            try FileManager.default.createDirectory(at: transcript.deletingLastPathComponent(), withIntermediateDirectories: true)
            if !FileManager.default.fileExists(atPath: transcript.path) {
                try "filename,text\n".write(to: transcript, atomically: true, encoding: .utf8)
            }
            let filename = output.lastPathComponent
            let escaped = trainingSentence.replacingOccurrences(of: "\"", with: "\"\"")
            let line = "\"\(filename)\",\"\(escaped)\"\n"
            let handle = try FileHandle(forWritingTo: transcript)
            try handle.seekToEnd()
            handle.write(Data(line.utf8))
            try handle.close()
            recordingValidation = "저장 완료: \(filename)"
        } catch { errorMessage = "학습 데이터 저장 오류: \(error.localizedDescription)" }
    }
}

private struct WaveformView: View {
    let samples: [Float]
    let active: Bool

    var body: some View {
        GeometryReader { proxy in
            HStack(alignment: .center, spacing: 2) {
                ForEach(Array(samples.enumerated()), id: \.offset) { _, sample in
                    Capsule()
                        .fill(active ? Color.accentColor : Color.secondary.opacity(0.45))
                        .frame(width: max(1, (proxy.size.width - 94) / 48), height: max(3, CGFloat(sample) * proxy.size.height))
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .center)
            .padding(.horizontal, 8)
        }
    }
}
