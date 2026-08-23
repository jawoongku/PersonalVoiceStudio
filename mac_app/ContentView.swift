import SwiftUI
import AppKit
import AVFoundation

struct ContentView: View {
    private let refreshTimer = Timer.publish(every: 10, on: .main, in: .common).autoconnect()

    @State private var snapshot: BridgeSnapshot?
    @State private var errorMessage: String?
    @State private var selectedSection: WorkspaceSection? = .dashboard
    @State private var ttsText = ""
    @State private var ttsOutput: URL?
    @State private var isSynthesizing = false
    @State private var isCancelling = false
    @State private var selectedVoicePath: String?
    @State private var ttsDevice = "cpu"
    @State private var trainingSentence = "오늘 아침에는 평소보다 조금 일찍 일어났습니다."
    @State private var sentenceIndex = 0
    @State private var recordingValidation: String?
    @State private var recordings: [URL] = []
    @State private var recordingTranscripts: [String: String] = [:]
    @State private var selectedRecordings: Set<URL> = []
    @State private var modelName = "my_voice"
    @State private var models: [URL] = []
    @State private var renameModel: URL?
    @State private var renameValue = ""
    @State private var isCreatingDataset = false
    @State private var datasetProgress = 0.0
    @State private var datasetProgressText = ""
    @StateObject private var recorder = Recorder()
    @StateObject private var audioPlayer = AudioPlayer()

    let projectDirectory: String
    let jobPath: String
    let voicesPath: String

    var body: some View {
        NavigationSplitView {
            List(WorkspaceSection.allCases, selection: $selectedSection) { section in
                Label(section.title, systemImage: section.symbol)
                    .tag(section)
            }
            .navigationTitle("Voice Studio")
            .listStyle(.sidebar)
        } detail: {
            Group {
                switch selectedSection ?? .dashboard {
                case .dashboard: dashboard
                case .recording: recordingWorkspace
                case .training: trainingWorkspace
                case .tts: ttsWorkspace
                }
            }
            .navigationTitle((selectedSection ?? .dashboard).title)
            .toolbar { toolbarContent }
        }
        .task { refresh(); refreshRecordings(); refreshModels() }
        .onReceive(refreshTimer) { _ in refresh() }
        .onReceive(NotificationCenter.default.publisher(for: .personalVoiceRefresh)) { _ in refresh() }
        .alert("작업을 완료할 수 없습니다", isPresented: hasError) {
            Button("확인", role: .cancel) { errorMessage = nil }
        } message: {
            Text(errorMessage ?? "알 수 없는 오류가 발생했습니다.")
        }
        .alert("모델 이름 변경", isPresented: Binding(get: { renameModel != nil }, set: { if !$0 { renameModel = nil } })) {
            TextField("모델 이름", text: $renameValue)
            Button("취소", role: .cancel) { renameModel = nil }
            Button("변경") { commitRenameModel() }
        } message: {
            Text("새 모델 이름을 입력하세요.")
        }
    }

    @ToolbarContentBuilder
    private var toolbarContent: some ToolbarContent {
        ToolbarItem(placement: .primaryAction) {
            Button(action: refresh) {
                Label("상태 새로고침", systemImage: "arrow.clockwise")
            }
            .help("상태 새로고침")
            .keyboardShortcut("r", modifiers: [.command])
        }

        if snapshot?.job.status == "running" {
            ToolbarItem(placement: .primaryAction) {
                Button(role: .destructive, action: cancelTraining) {
                    Label(isCancelling ? "취소 요청 중" : "학습 취소", systemImage: "stop.fill")
                }
                .disabled(isCancelling)
                .help("실행 중인 학습 취소")
            }
        }
    }

    private var dashboard: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                if let snapshot {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("현재 프로젝트").font(.title2)
                        Text("녹음부터 Voice Package 생성, TTS까지의 상태를 확인합니다.")
                            .foregroundStyle(.secondary)
                    }

                    GroupBox("작업 상태") {
                        Grid(alignment: .leading, horizontalSpacing: 24, verticalSpacing: 12) {
                            GridRow {
                                statusLabel(title: "학습", value: snapshot.job.status, symbol: jobSymbol(snapshot.job.status))
                                statusLabel(title: "사용 가능한 Voice Package", value: "\(snapshot.voices.filter(\.valid).count)개", symbol: "waveform.badge.checkmark")
                            }
                            GridRow {
                                statusLabel(title: "실행 장치", value: snapshot.mps.status, symbol: snapshot.mps.tensor_probe ? "cpu" : "exclamationmark.triangle")
                                statusLabel(title: "학습 기록", value: "\(snapshot.runs.count)개", symbol: "clock.arrow.circlepath")
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.vertical, 4)
                    }

                    if let action = snapshot.mps.action.nonEmpty {
                        LabeledContent("실행 장치 안내") {
                            Text(action).foregroundStyle(.secondary).multilineTextAlignment(.trailing)
                        }
                    }

                    if let error = snapshot.job.error?.nonEmpty {
                        ContentUnavailableView {
                            Label("학습 상태를 확인하세요", systemImage: "exclamationmark.triangle")
                        } description: {
                            Text(error)
                        }
                    }

                    GroupBox("최근 학습") {
                        if snapshot.runs.isEmpty {
                            Text("아직 학습 기록이 없습니다. 녹음을 저장한 뒤 학습을 시작하세요.")
                                .foregroundStyle(.secondary)
                        } else {
                            VStack(alignment: .leading, spacing: 10) {
                                ForEach(snapshot.runs.prefix(5)) { run in
                                    VStack(alignment: .leading, spacing: 3) {
                                        HStack {
                                            Text(run.name)
                                            Spacer()
                                            Text(run.job_status ?? "상태 없음").foregroundStyle(.secondary)
                                        }
                                        if let checkpoint = run.checkpoint?.nonEmpty {
                                            Text(checkpoint).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                                        }
                                    }
                                    .contextMenu {
                                        if let checkpoint = run.checkpoint?.nonEmpty {
                                            Button("Finder에서 보기") {
                                                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: checkpoint)])
                                            }
                                        }
                                    }
                                    if run.id != snapshot.runs.prefix(5).last?.id { Divider() }
                                }
                            }
                        }
                    }
                } else {
                    loadingState
                }
            }
            .frame(maxWidth: 880, alignment: .leading)
            .padding()
        }
    }

    private var recordingWorkspace: some View {
        ScrollView {
            HStack(alignment: .top, spacing: 20) {
                VStack(alignment: .leading, spacing: 16) {
                    Text("녹음 파일").font(.title2)
                    GroupBox("저장된 녹음") {
                        if recordings.isEmpty {
                            Text("아직 저장된 녹음 파일이 없습니다.").foregroundStyle(.secondary)
                        } else {
                            ScrollView(.vertical) {
                                LazyVStack(alignment: .leading, spacing: 8) {
                                ForEach(recordings, id: \.self) { recording in
                                    HStack {
                                        Toggle("", isOn: Binding(get: { selectedRecordings.contains(recording) }, set: { checked in
                                            if checked { selectedRecordings.insert(recording) } else { selectedRecordings.remove(recording) }
                                        })).labelsHidden()
                                        VStack(alignment: .leading) {
                                            Text(recording.lastPathComponent)
                                            Text(recordingTranscripts[recording.lastPathComponent] ?? "transcript 미등록")
                                                .lineLimit(2)
                                            Text(recordingDuration(recording)).font(.caption).foregroundStyle(.secondary)
                                        }
                                        Spacer()
                                        Button("재생") { togglePlayback(recording) }
                                        Button("삭제", role: .destructive) { deleteRecording(recording) }
                                    }
                                    if recording != recordings.last { Divider() }
                                }
                                }
                            }
                            .frame(maxHeight: 320)
                        }
                        Text("선택 \(selectedRecordings.count)개").font(.caption).foregroundStyle(.secondary)
                        TextField("모델 이름", text: $modelName).textFieldStyle(.roundedBorder)
                        if isCreatingDataset {
                            ProgressView(value: datasetProgress) {
                                Text(datasetProgressText).font(.caption)
                            }
                            .progressViewStyle(.linear)
                        }
                        Button("선택 파일로 모델 데이터 만들기", action: createSelectedDataset)
                            .disabled(isCreatingDataset || selectedRecordings.isEmpty || sanitizedModelName.isEmpty)
                    }
                    GroupBox("생성된 모델") {
                        if models.isEmpty { Text("아직 생성된 모델이 없습니다.").foregroundStyle(.secondary) }
                        else {
                            ForEach(models, id: \.self) { model in
                                HStack {
                                    Image(systemName: "cube").foregroundStyle(Color.accentColor)
                                    Text(model.lastPathComponent)
                                    Spacer()
                                    Button("이름 변경") { renameModel = model; renameValue = model.lastPathComponent }
                                    Button("삭제", role: .destructive) { deleteModel(model) }
                                }
                            }
                        }
                    }
                }
                .frame(minWidth: 360, maxWidth: .infinity, alignment: .leading)

                VStack(alignment: .leading, spacing: 16) {
                    Text("녹음").font(.title2)
                    Text("문장을 읽고 녹음한 뒤 검증 결과를 확인하세요.").foregroundStyle(.secondary)
                    GroupBox("녹음할 문장") {
                        VStack(alignment: .leading, spacing: 14) {
                            Text(trainingSentence).font(.title3).textSelection(.enabled)
                            HStack {
                                Button("다음 문장", action: nextTrainingSentence)
                                Spacer()
                                Button(recorder.isRecording ? "녹음 중지" : "이 문장 녹음", action: toggleRecording)
                            }
                        }
                    }
                    GroupBox("입력 레벨") {
                        WaveformView(samples: recorder.waveform, active: recorder.isRecording).frame(height: 100)
                    }
                    if let validation = recordingValidation { Text(validation).foregroundStyle(.secondary) }
                    if let output = recorder.lastOutput, !recorder.isRecording {
                        GroupBox("최근 녹음") {
                            HStack {
                                Text(output.lastPathComponent)
                                Spacer()
                                Button("검증", action: validateRecording)
                                Button("학습 데이터로 저장") { saveTrainingRecording(output) }
                                    .disabled(!(recordingValidation?.hasPrefix("사용 가능") ?? false))
                                Button("재생") { togglePlayback(output) }
                            }
                        }
                    }
                }
                .frame(minWidth: 360, maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxWidth: 1100, alignment: .leading)
            .padding()
        }
    }

    private var trainingWorkspace: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("학습 상태").font(.title2)
                    Text("현재 학습 작업과 최근 지표를 확인합니다. 학습 시작과 설정 변경은 기존 실행 도구에서 관리됩니다.")
                        .foregroundStyle(.secondary)
                }

                if let snapshot {
                    GroupBox("현재 작업") {
                        Form {
                            LabeledContent("상태", value: snapshot.job.status)
                            if let step = snapshot.job.step { LabeledContent("현재 step", value: String(step)) }
                            if let command = snapshot.job.command?.nonEmpty {
                                LabeledContent("명령") { Text(command).lineLimit(2).textSelection(.enabled) }
                            }
                            if let metrics = snapshot.job.metrics {
                                if let value = metrics["train_loss"] { LabeledContent("Train loss", value: formattedMetric(value)) }
                                if let value = metrics["val_loss"] { LabeledContent("Validation loss", value: formattedMetric(value)) }
                                if let value = metrics["mps_memory"] { LabeledContent("MPS memory", value: formattedBytes(value)) }
                            }
                        }
                    }

                    if snapshot.job.status == "running" {
                        Button(role: .destructive, action: cancelTraining) {
                            Label(isCancelling ? "취소 요청 중…" : "학습 취소", systemImage: "stop.fill")
                        }
                        .disabled(isCancelling)
                    }

                    GroupBox("학습 기록") {
                        if snapshot.runs.isEmpty {
                            Text("표시할 학습 기록이 없습니다.").foregroundStyle(.secondary)
                        } else {
                            ForEach(snapshot.runs) { run in
                                VStack(alignment: .leading, spacing: 4) {
                                    HStack {
                                        Text(run.name)
                                        Spacer()
                                        Text(run.job_status ?? "상태 없음").foregroundStyle(.secondary)
                                    }
                                    if let values = run.last_metrics {
                                        HStack(spacing: 16) {
                                            if let train = values["train_loss"] { Text("Train \(formattedMetric(train))") }
                                            if let val = values["val_loss"] { Text("Validation \(formattedMetric(val))") }
                                            if let lr = values["learning_rate"] { Text("LR \(formattedMetric(lr))") }
                                        }
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                    }
                                }
                                .padding(.vertical, 5)
                            }
                        }
                    }
                } else {
                    loadingState
                }
            }
            .frame(maxWidth: 760, alignment: .leading)
            .padding()
        }
    }

    private var ttsWorkspace: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Voice Package TTS").font(.title2)
                    Text("Voice Package를 선택하고 텍스트를 입력해 음성을 생성합니다.")
                        .foregroundStyle(.secondary)
                }

                if let snapshot {
                    if snapshot.voices.filter(\.valid).isEmpty {
                        ContentUnavailableView {
                            Label("사용 가능한 Voice Package가 없습니다", systemImage: "waveform.badge.exclamationmark")
                        } description: {
                            Text("학습이 완료된 Voice Package를 준비한 후 다시 시도하세요.")
                        }
                    } else {
                        Form {
                            Picker("Voice Package", selection: $selectedVoicePath) {
                                ForEach(snapshot.voices.filter(\.valid)) { voice in
                                    Text(voice.name).tag(Optional(voice.path))
                                }
                            }
                            Picker("실행 장치", selection: $ttsDevice) {
                                Text("CPU").tag("cpu")
                                Text("MPS").tag("mps").disabled(!snapshot.mps.tensor_probe)
                            }
                            TextField("생성할 텍스트", text: $ttsText, axis: .vertical)
                                .lineLimit(3...8)
                        }

                        HStack {
                            Button(isSynthesizing ? "생성 중…" : "음성 생성", action: synthesize)
                                .disabled(isSynthesizing || ttsText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                            if isSynthesizing { ProgressView().controlSize(.small) }
                        }

                        if let output = ttsOutput {
                            GroupBox("생성된 음성") {
                                HStack {
                                    Text(output.lastPathComponent)
                                    Spacer()
                                    Button(audioPlayer.isPlaying ? "재생 중지" : "재생") { togglePlayback(output) }
                                    Button("Finder에서 보기") { NSWorkspace.shared.activateFileViewerSelecting([output]) }
                                }
                                .padding(.vertical, 4)
                            }
                        }
                    }
                } else {
                    loadingState
                }
            }
            .frame(maxWidth: 760, alignment: .leading)
            .padding()
        }
    }

    private var loadingState: some View {
        ContentUnavailableView {
            Label("상태를 불러오는 중", systemImage: "arrow.triangle.2.circlepath")
        } description: {
            Text("프로젝트와 Voice Package 상태를 확인하고 있습니다.")
        }
    }

    private var hasError: Binding<Bool> {
        Binding(get: { errorMessage != nil }, set: { if !$0 { errorMessage = nil } })
    }

    private func statusLabel(title: String, value: String, symbol: String) -> some View {
        LabeledContent {
            Text(value).foregroundStyle(.primary)
        } label: {
            Label(title, systemImage: symbol).foregroundStyle(.secondary)
        }
    }

    private func jobSymbol(_ status: String) -> String {
        switch status {
        case "running": return "progress.indicator"
        case "completed", "succeeded": return "checkmark.circle"
        case "failed", "cancelled": return "exclamationmark.triangle"
        default: return "clock"
        }
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
                    if selectedVoicePath == nil || !value.voices.contains(where: { $0.path == selectedVoicePath && $0.valid }) {
                        selectedVoicePath = value.voices.first(where: { $0.valid })?.path
                    }
                    errorMessage = nil
                }
            } catch {
                DispatchQueue.main.async { errorMessage = "프로젝트 상태를 불러오지 못했습니다. \(error.localizedDescription)" }
            }
        }
    }

    private func cancelTraining() {
        isCancelling = true
        DispatchQueue.global(qos: .utility).async {
            do {
                try BridgeClient.cancelJob(job: jobPath, workingDirectory: projectDirectory)
                DispatchQueue.main.async { isCancelling = false; refresh() }
            } catch {
                DispatchQueue.main.async { errorMessage = "학습 취소 요청을 보내지 못했습니다. \(error.localizedDescription)"; isCancelling = false }
            }
        }
    }

    private func toggleRecording() {
        if recorder.isRecording {
            recorder.stop()
            validateRecording()
            return
        }

        let raw = URL(fileURLWithPath: projectDirectory).appendingPathComponent("data/my_voice/raw", isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: raw, withIntermediateDirectories: true)
            let url = raw.appendingPathComponent("ui_\(UUID().uuidString).wav")
            recorder.requestMicrophonePermission { granted in
                DispatchQueue.main.async {
                    guard granted else {
                        errorMessage = "마이크 권한이 필요합니다. 시스템 설정에서 Personal Voice Studio의 마이크 접근을 허용하세요."
                        return
                    }
                    do {
                        try recorder.start(to: url)
                        recordingValidation = "녹음 중입니다. 완료되면 녹음 중지를 선택하세요."
                    } catch {
                        errorMessage = "녹음을 시작하지 못했습니다. \(error.localizedDescription)"
                    }
                }
            }
        } catch {
            errorMessage = "녹음 저장 위치를 준비하지 못했습니다. \(error.localizedDescription)"
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
        guard let output = recorder.lastOutput else {
            recordingValidation = "검증할 녹음 파일이 없습니다."
            return
        }
        do {
            let audio = try AVAudioFile(forReading: output)
            let seconds = Double(audio.length) / audio.processingFormat.sampleRate
            let channels = audio.processingFormat.channelCount
            if seconds <= 0 || seconds > 30 {
                recordingValidation = "재녹음 필요: 길이는 0초 초과 30초 이하여야 합니다."
            } else if channels != 1 {
                recordingValidation = "재녹음 검토: mono 녹음을 권장합니다. (현재 \(channels)ch)"
            } else {
                recordingValidation = String(format: "사용 가능: %.2f초, %.0fHz, mono. 문장 일치 여부를 확인한 뒤 저장하세요.", seconds, audio.processingFormat.sampleRate)
            }
        } catch {
            recordingValidation = "검증 오류: \(error.localizedDescription)"
        }
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
            refreshRecordings()
        } catch {
            errorMessage = "학습 데이터를 저장하지 못했습니다. \(error.localizedDescription)"
        }
    }

    private func refreshRecordings() {
        let root = URL(fileURLWithPath: projectDirectory).appendingPathComponent("data/my_voice/raw", isDirectory: true)
        let transcriptURL = URL(fileURLWithPath: projectDirectory).appendingPathComponent("data/my_voice/transcripts.csv")
        var transcripts: [String: String] = [:]
        if let data = try? String(contentsOf: transcriptURL, encoding: .utf8) {
            for line in data.split(separator: "\n").dropFirst() {
                let parts = line.split(separator: ",", maxSplits: 1, omittingEmptySubsequences: false)
                if parts.count == 2 {
                    let filename = parts[0].trimmingCharacters(in: CharacterSet(charactersIn: "\""))
                    let text = parts[1].trimmingCharacters(in: CharacterSet(charactersIn: "\"")).replacingOccurrences(of: "\"\"", with: "\"")
                    transcripts[filename] = text
                }
            }
        }
        recordingTranscripts = transcripts
        recordings = (try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil))?
            .filter { $0.pathExtension.lowercased() == "wav" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent } ?? []
        selectedRecordings = selectedRecordings.intersection(Set(recordings))
    }

    private func refreshModels() {
        let root = URL(fileURLWithPath: projectDirectory).appendingPathComponent("data/models", isDirectory: true)
        models = (try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: [.isDirectoryKey]))?
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
            .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending } ?? []
    }

    private func deleteModel(_ model: URL) {
        do {
            try FileManager.default.removeItem(at: model)
            refreshModels()
            recordingValidation = "모델 삭제 완료: \(model.lastPathComponent)"
        } catch {
            errorMessage = "모델을 삭제하지 못했습니다. \(error.localizedDescription)"
        }
    }

    private func commitRenameModel() {
        guard let source = renameModel else { return }
        let value = renameValue.trimmingCharacters(in: .whitespacesAndNewlines)
        let safe = value.map { $0.isLetter || $0.isNumber || $0 == "_" || $0 == "-" ? $0 : "_" }
        let name = String(safe).trimmingCharacters(in: CharacterSet(charactersIn: "._-"))
        guard !name.isEmpty else { errorMessage = "모델 이름을 입력하세요."; return }
        let destination = source.deletingLastPathComponent().appendingPathComponent(name, isDirectory: true)
        do {
            if FileManager.default.fileExists(atPath: destination.path) { throw NSError(domain: "PersonalVoiceStudio", code: 2, userInfo: [NSLocalizedDescriptionKey: "같은 이름의 모델이 이미 있습니다."]) }
            try FileManager.default.moveItem(at: source, to: destination)
            renameModel = nil
            modelName = name
            refreshModels()
            recordingValidation = "모델 이름 변경 완료: \(name)"
        } catch {
            errorMessage = "모델 이름을 변경하지 못했습니다. \(error.localizedDescription)"
        }
    }

    private func deleteRecording(_ recording: URL) {
        do {
            try FileManager.default.removeItem(at: recording)
            selectedRecordings.remove(recording)
            refreshRecordings()
            recordingValidation = "삭제 완료: \(recording.lastPathComponent)"
        } catch {
            errorMessage = "녹음 파일을 삭제하지 못했습니다. \(error.localizedDescription)"
        }
    }

    private func createSelectedDataset() {
        let sources = selectedRecordings.sorted { $0.lastPathComponent < $1.lastPathComponent }
        let safeName = sanitizedModelName
        let displayName = modelName.trimmingCharacters(in: .whitespacesAndNewlines)
        let project = projectDirectory
        isCreatingDataset = true
        datasetProgress = 0
        datasetProgressText = "선택 파일과 transcript 확인 중…"
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let root = URL(fileURLWithPath: project).appendingPathComponent("data/models/\(safeName)", isDirectory: true)
                let raw = root.appendingPathComponent("raw", isDirectory: true)
                let transcript = root.appendingPathComponent("transcripts.csv")
                let sourceTranscript = URL(fileURLWithPath: project).appendingPathComponent("data/my_voice/transcripts.csv")
                let sourceLines = (try String(contentsOf: sourceTranscript, encoding: .utf8).split(separator: "\n").map(String.init))
                var output = "filename,text\n"
                var missing: [String] = []
                for source in sources {
                    let prefix = "\"\(source.lastPathComponent)\"," 
                    guard let line = sourceLines.first(where: { $0.hasPrefix(prefix) }) else { missing.append(source.lastPathComponent); continue }
                    output += line + "\n"
                }
                guard missing.isEmpty else { throw NSError(domain: "PersonalVoiceStudio", code: 3, userInfo: [NSLocalizedDescriptionKey: "transcript가 없는 파일: \(missing.joined(separator: ", "))"]) }
                try FileManager.default.createDirectory(at: raw, withIntermediateDirectories: true)
                for (index, source) in sources.enumerated() {
                    let destination = raw.appendingPathComponent(source.lastPathComponent)
                    if FileManager.default.fileExists(atPath: destination.path) { try FileManager.default.removeItem(at: destination) }
                    try FileManager.default.copyItem(at: source, to: destination)
                    DispatchQueue.main.async {
                        datasetProgress = Double(index + 1) / Double(sources.count)
                        datasetProgressText = "파일 복사 중 \(index + 1)/\(sources.count)"
                    }
                }
                try output.write(to: transcript, atomically: true, encoding: .utf8)
                DispatchQueue.main.async {
                    isCreatingDataset = false
                    recordingValidation = "모델 \(displayName) 데이터 생성 완료: \(sources.count)개 · \(root.path)"
                    refreshModels()
                }
            } catch {
                DispatchQueue.main.async {
                    isCreatingDataset = false
                    errorMessage = "선택 파일 학습 데이터 생성에 실패했습니다. \(error.localizedDescription)"
                }
            }
        }
    }

    private var sanitizedModelName: String {
        let value = modelName.trimmingCharacters(in: .whitespacesAndNewlines)
        let safe = value.map { character in
            character.isLetter || character.isNumber || character == "_" || character == "-" ? character : "_"
        }
        return String(safe).trimmingCharacters(in: CharacterSet(charactersIn: "._-"))
    }

    private func recordingDuration(_ url: URL) -> String {
        guard let audio = try? AVAudioFile(forReading: url) else { return "WAV 읽기 실패" }
        let seconds = Double(audio.length) / audio.processingFormat.sampleRate
        return String(format: "%.2f초 · %.0fHz · %uch", seconds, audio.processingFormat.sampleRate, audio.processingFormat.channelCount)
    }

    private func synthesize() {
        guard !ttsText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            errorMessage = "생성할 텍스트를 입력하세요."
            return
        }
        guard let voicePath = selectedVoicePath ?? snapshot?.voices.first(where: { $0.valid })?.path else {
            errorMessage = "사용 가능한 Voice Package가 없습니다."
            return
        }
        let output = URL(fileURLWithPath: projectDirectory).appendingPathComponent("artifacts/swiftui_tts.wav")
        let device = ttsDevice
        let text = ttsText
        isSynthesizing = true
        let modelDirectory = ProcessInfo.processInfo.environment["PVS_MODEL_DIR"] ?? "/Users/jawoongku/Models/Fun-CosyVoice3-0.5B"
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try BridgeClient.synthesize(voice: voicePath, text: text, output: output.path, modelDirectory: modelDirectory, workingDirectory: projectDirectory, device: device)
                DispatchQueue.main.async { ttsOutput = output; isSynthesizing = false }
            } catch {
                DispatchQueue.main.async { errorMessage = "음성을 생성하지 못했습니다. \(error.localizedDescription)"; isSynthesizing = false }
            }
        }
    }

    private func togglePlayback(_ url: URL) {
        if audioPlayer.isPlaying {
            audioPlayer.stop()
        } else {
            do {
                try audioPlayer.play(url: url)
            } catch {
                errorMessage = "오디오를 재생하지 못했습니다. \(error.localizedDescription)"
            }
        }
    }

    private func formattedMetric(_ value: Double) -> String {
        String(format: "%.4f", value)
    }

    private func formattedBytes(_ value: Double) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(value), countStyle: .memory)
    }
}

private enum WorkspaceSection: String, CaseIterable, Identifiable {
    case dashboard
    case recording
    case training
    case tts

    var id: String { rawValue }

    var title: String {
        switch self {
        case .dashboard: return "대시보드"
        case .recording: return "녹음"
        case .training: return "학습"
        case .tts: return "TTS"
        }
    }

    var symbol: String {
        switch self {
        case .dashboard: return "rectangle.3.group"
        case .recording: return "mic"
        case .training: return "cpu"
        case .tts: return "waveform"
        }
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

private extension String {
    var nonEmpty: String? { isEmpty ? nil : self }
}

extension Notification.Name {
    static let personalVoiceRefresh = Notification.Name("PersonalVoiceStudio.refresh")
}
