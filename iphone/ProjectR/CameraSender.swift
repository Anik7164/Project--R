import AVFoundation
import CoreImage
import ImageIO
import UIKit

/// This phone as a match camera: JPEG pictures to the PC's /ws/camera, the
/// same socket the web camera page uses. The PC lists it as
/// "Phone camera: <name>" for Camera 1, 2 or 3.
///
/// Started by the operator choosing this iPhone on the PC (the PC hands the
/// app a token for it, so a watching phone can film too), or by hand with
/// the operator password. Either way it goes with this phone's id, so the
/// PC keeps one entry for it.
///
/// No video stream: a picture is packed when one is due (FRAME_GAP) and only
/// while fewer than 2 are still on their way, so slow Wi-Fi never builds lag.
final class CameraSender: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate,
                          URLSessionWebSocketDelegate {
    @Published private(set) var status = "Camera off"
    @Published private(set) var running = false
    @Published private(set) var rate = 0

    let session = AVCaptureSession()

    static let frameGap: CFTimeInterval = 1.0 / 15      // the PC analyses up to 12 a second
    static let longSide: CGFloat = 1280
    static let quality: CGFloat = 0.7

    // Main queue:
    private var urlSession: URLSession?
    private var socket: URLSessionWebSocketTask?
    private var target: URL?
    private var wanted = false
    private var back = true
    private var generation = 0

    // Camera queue (`work`) only:
    private let work = DispatchQueue(label: "projectr.camera")
    private let context = CIContext()
    private var sending: URLSessionWebSocketTask?
    private var open = false
    private var inFlight = 0
    private var lastSent: CFTimeInterval = 0
    private var counted = 0
    private var countedAt: CFTimeInterval = 0
    private var orientation: CGImagePropertyOrientation = .up

    @MainActor
    func start(base: URL, key: String, name: String, back: Bool, token: String = "", app: String = "") {
        guard !running else { return }
        guard var parts = URLComponents(url: base, resolvingAgainstBaseURL: false) else { return }
        parts.scheme = base.scheme == "https" ? "wss" : "ws"
        parts.path = "/ws/camera"
        var query = [URLQueryItem(name: "name", value: name), URLQueryItem(name: "key", value: key)]
        if !token.isEmpty { query.append(URLQueryItem(name: "token", value: token)) }
        if !app.isEmpty { query.append(URLQueryItem(name: "app", value: app)) }
        parts.queryItems = query
        target = parts.url
        self.back = back
        running = true
        wanted = true
        status = "Opening the camera…"
        UIApplication.shared.isIdleTimerDisabled = true
        UIDevice.current.beginGeneratingDeviceOrientationNotifications()
        NotificationCenter.default.addObserver(self, selector: #selector(orientationChanged),
                                               name: UIDevice.orientationDidChangeNotification, object: nil)
        didGuess = false
        updateOrientation()
        AVCaptureDevice.requestAccess(for: .video) { granted in
            Task { @MainActor in
                guard self.wanted else { return }
                guard granted else {
                    self.stop()
                    self.status = "The camera is not allowed for Project R. Turn it on in Settings › Project R."
                    return
                }
                self.work.async {
                    self.configure()
                    self.session.startRunning()
                }
                self.connect()
            }
        }
    }

    @MainActor
    func stop() {
        wanted = false
        generation += 1
        socket?.cancel(with: .goingAway, reason: nil)
        socket = nil
        urlSession?.invalidateAndCancel()
        urlSession = nil
        work.async {
            self.sending = nil
            self.open = false
            if self.session.isRunning { self.session.stopRunning() }
        }
        if running {
            UIApplication.shared.isIdleTimerDisabled = false
            NotificationCenter.default.removeObserver(self, name: UIDevice.orientationDidChangeNotification,
                                                      object: nil)
        }
        running = false
        rate = 0
        status = "Camera off"
    }

    // MARK: camera

    private func configure() {
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720
        for input in session.inputs { session.removeInput(input) }
        let position: AVCaptureDevice.Position = back ? .back : .front
        if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position),
           let input = try? AVCaptureDeviceInput(device: device), session.canAddInput(input) {
            session.addInput(input)
        }
        if session.outputs.isEmpty {
            let output = AVCaptureVideoDataOutput()
            output.alwaysDiscardsLateVideoFrames = true
            output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            output.setSampleBufferDelegate(self, queue: work)
            if session.canAddOutput(output) { session.addOutput(output) }
        }
        session.commitConfiguration()
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard open, let task = sending, inFlight < 2 else { return }
        let now = CACurrentMediaTime()
        guard now - lastSent >= CameraSender.frameGap,
              let pixels = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        lastSent = now
        var image = CIImage(cvPixelBuffer: pixels).oriented(orientation)
        let long = max(image.extent.width, image.extent.height)
        if long > CameraSender.longSide {
            let scale = CameraSender.longSide / long
            image = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        }
        let quality = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let jpeg = context.jpegRepresentation(of: image, colorSpace: space,
                                                    options: [quality: CameraSender.quality]) else { return }
        inFlight += 1
        task.send(.data(jpeg)) { [weak self] error in
            guard let self else { return }
            self.work.async {
                self.inFlight = max(0, self.inFlight - 1)
                if error == nil { self.counted += 1 }
                let now = CACurrentMediaTime()
                if self.countedAt == 0 { self.countedAt = now }
                if now - self.countedAt >= 2 {
                    let rate = Int((Double(self.counted) / (now - self.countedAt)).rounded())
                    self.counted = 0
                    self.countedAt = now
                    Task { @MainActor in self.rate = rate }
                }
            }
        }
    }

    @MainActor
    @objc private func orientationChanged() {
        updateOrientation()
    }

    /// Pictures go to the PC upright, held portrait or landscape.
    @MainActor
    private func updateOrientation() {
        let value: CGImagePropertyOrientation?
        switch UIDevice.current.orientation {
        case .portrait: value = .right
        case .portraitUpsideDown: value = .left
        case .landscapeLeft: value = back ? .up : .down
        case .landscapeRight: value = back ? .down : .up
        default: value = initialOrientation()
        }
        if let value { work.async { self.orientation = value } }
    }

    @MainActor
    private func initialOrientation() -> CGImagePropertyOrientation? {
        guard running, !didGuess else { return nil }       // face up or down: keep the last one
        didGuess = true
        let scene = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.first
        switch scene?.interfaceOrientation {
        case .landscapeRight?: return back ? .up : .down
        case .landscapeLeft?: return back ? .down : .up
        default: return .right
        }
    }

    private var didGuess = false

    // MARK: socket

    @MainActor
    private func connect() {
        guard wanted, let target else { return }
        generation += 1
        let mine = generation
        status = "Connecting to the PC…"
        let session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
        let task = session.webSocketTask(with: target)
        task.maximumMessageSize = 32 * 1024 * 1024
        urlSession?.invalidateAndCancel()
        urlSession = session
        socket = task
        work.async {
            self.sending = task
            self.open = false
            self.inFlight = 0
        }
        task.resume()
        listen(task, mine)
    }

    private func listen(_ task: URLSessionWebSocketTask, _ mine: Int) {
        task.receive { [weak self] result in
            guard let self else { return }
            switch result {
            case .success(let message):
                if case let .string(text) = message,
                   let data = text.data(using: .utf8),
                   let told = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
                   let name = told["name"] as? String {
                    Task { @MainActor in
                        if mine == self.generation { self.status = "Sending to the PC as “\(name)”" }
                    }
                }
                self.listen(task, mine)
            case .failure:
                let reason = task.closeReason.flatMap { String(data: $0, encoding: .utf8) } ?? ""
                Task { @MainActor in self.lost(mine, reason: reason) }
            }
        }
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didOpenWithProtocol protocol: String?) {
        work.async {
            if self.sending === webSocketTask { self.open = true }
        }
    }

    @MainActor
    private func lost(_ mine: Int, reason: String) {
        guard mine == generation, wanted else { return }
        work.async { self.open = false }
        if reason.contains("operator key") {
            stop()
            status = "Only the operator can start a camera from here. On the PC, choose this iPhone under Camera 1, 2 or 3: it then starts filming by itself."
            return
        }
        status = "Lost the PC, connecting again…"
        Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            guard let self, self.wanted, mine == self.generation else { return }
            self.connect()
        }
    }
}
