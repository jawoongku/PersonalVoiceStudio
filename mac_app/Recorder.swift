import AVFoundation
import Foundation

final class Recorder: ObservableObject {
    @Published private(set) var isRecording = false
    @Published private(set) var lastOutput: URL?
    private let engine = AVAudioEngine()
    private var file: AVAudioFile?

    func start(to url: URL) throws {
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        file = try AVAudioFile(forWriting: url, settings: format.settings)
        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, _ in
            do { try self?.file?.write(from: buffer) } catch { self?.stop() }
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
