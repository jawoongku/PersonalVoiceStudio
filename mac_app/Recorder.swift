import AVFoundation
import Foundation

final class Recorder: ObservableObject {
    @Published private(set) var isRecording = false
    @Published private(set) var lastOutput: URL?
    @Published private(set) var waveform: [Float] = Array(repeating: 0, count: 48)
    private let engine = AVAudioEngine()
    private var file: AVAudioFile?

    func requestMicrophonePermission(_ completion: @escaping (Bool) -> Void) {
        AVCaptureDevice.requestAccess(for: .audio, completionHandler: completion)
    }

    func start(to url: URL) throws {
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        file = try AVAudioFile(forWriting: url, settings: format.settings)
        waveform = Array(repeating: 0, count: 48)
        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, _ in
            do {
                try self?.file?.write(from: buffer)
                guard let channel = buffer.floatChannelData?[0] else { return }
                let count = Int(buffer.frameLength)
                let sum = (0..<count).reduce(Float.zero) { $0 + channel[$1] * channel[$1] }
                let rms = min(1, sqrt(sum / Float(max(1, count))) * 4)
                DispatchQueue.main.async { [weak self] in
                    guard let self, self.isRecording else { return }
                    self.waveform.append(rms)
                    if self.waveform.count > 48 { self.waveform.removeFirst(self.waveform.count - 48) }
                }
            } catch { self?.stop() }
        }
        engine.prepare()
        try engine.start()
        isRecording = true
        lastOutput = url
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        file = nil
        isRecording = false
    }
}
