import AVFoundation
import CoreImage
import CoreMotion
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
/// No video stream: a picture is packed when one is due (about 24 a second;
/// the PC shows up to 20 and analyses 12) and only while fewer than 3 are
/// still on their way, so slow Wi-Fi never builds lag. When the Wi-Fi cannot
/// carry that many, the pictures get smaller (1280 › 960 › 720 on the long
/// side) rather than fewer, and sharper again once it can.
final class CameraSender: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate,
                          URLSessionWebSocketDelegate {
    @Published private(set) var status = "Camera off"
    @Published private(set) var running = false
    @Published private(set) var rate = 0
    @Published private(set) var size = 1280
    /// Held sideways: the PC gets a wide picture.
    @Published private(set) var wide = false

    let session = AVCaptureSession()

    static let frameGap: CFTimeInterval = 1.0 / 24
    static let maxInFlight = 3
    /// Picture sizes, sharpest first (long side, JPEG quality).
    static let levels: [(longSide: CGFloat, quality: CGFloat)] = [(1280, 0.6), (960, 0.55), (720, 0.5)]

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
    private var nextDue: CFTimeInterval = 0
    private var level = 0
    private var slowReadings = 0
    private var fastReadings = 0
    private var counted = 0
    private var countedAt: CFTimeInterval = 0
    private var orientation: CGImagePropertyOrientation = .right

    // How the phone is held, from the motion sensor: iOS stops telling the
    // app when the screen's rotation lock is on, the sensor still knows.
    private let motion = CMMotionManager()
    private let motionQueue: OperationQueue = {
        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1
        return queue
    }()
    private var held: UIDeviceOrientation = .unknown      // motionQueue only

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
        startWatchingHowHeld()
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
            motion.stopAccelerometerUpdates()
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
        guard open, let task = sending, inFlight < CameraSender.maxInFlight else { return }
        let now = CACurrentMediaTime()
        // Paced by when the next one is due, not by the last one sent: the
        // camera gives 30 a second, and "33 ms since the last" would halve that.
        guard now >= nextDue, let pixels = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        nextDue = max(nextDue + CameraSender.frameGap, now - CameraSender.frameGap)
        let (longSide, compression) = CameraSender.levels[level]
        var image = CIImage(cvPixelBuffer: pixels).oriented(orientation)
        let long = max(image.extent.width, image.extent.height)
        if long > longSide {
            let scale = longSide / long
            image = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        }
        let quality = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let jpeg = context.jpegRepresentation(of: image, colorSpace: space,
                                                    options: [quality: compression]) else { return }
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
                    self.adapt(rate)
                    let size = Int(CameraSender.levels[self.level].longSide)
                    Task { @MainActor in
                        self.rate = rate
                        self.size = size
                    }
                }
            }
        }
    }

    /// Camera queue: smaller pictures when the Wi-Fi carries fewer than 16 a
    /// second (twice running), sharper again after 30 s of 21 or more.
    private func adapt(_ rate: Int) {
        if rate < 16 {
            slowReadings += 1
            fastReadings = 0
            if slowReadings >= 2 && level < CameraSender.levels.count - 1 {
                level += 1
                slowReadings = 0
            }
        } else if rate >= 21 {
            fastReadings += 1
            slowReadings = 0
            if fastReadings >= 15 && level > 0 {
                level -= 1
                fastReadings = 0
            }
        } else {
            slowReadings = 0
            fastReadings = 0
        }
    }

    /// Pictures go to the PC upright and the right way round: wide when
    /// the phone is held sideways, tall when upright, also with the
    /// rotation lock on.
    @MainActor
    private func startWatchingHowHeld() {
        let back = self.back
        let scene = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.first
        // Until the sensor answers: as the screen is turned.
        let first: UIDeviceOrientation
        switch scene?.interfaceOrientation {
        case .landscapeRight?: first = .landscapeLeft      // (the screen's names are the other way round)
        case .landscapeLeft?: first = .landscapeRight
        default: first = .portrait
        }
        motionQueue.addOperation { self.held = .unknown }
        apply(first, back: back)
        guard motion.isAccelerometerAvailable else { return }
        motion.accelerometerUpdateInterval = 0.1
        motion.startAccelerometerUpdates(to: motionQueue) { [weak self] data, _ in
            guard let self, let gravity = data?.acceleration,
                  let way = CameraSender.howHeld(x: gravity.x, y: gravity.y, z: gravity.z, was: self.held),
                  way != self.held else { return }
            self.held = way
            self.apply(way, back: back)
        }
    }

    private func apply(_ way: UIDeviceOrientation, back: Bool) {
        let value: CGImagePropertyOrientation
        switch way {
        case .portraitUpsideDown: value = .left
        case .landscapeLeft: value = back ? .up : .down          // home button (or the side button) right
        case .landscapeRight: value = back ? .down : .up
        default: value = .right
        }
        let wide = way == .landscapeLeft || way == .landscapeRight
        work.async { self.orientation = value }
        Task { @MainActor in self.wide = wide }
    }

    /// Which way is down, from gravity (x right, y up the screen): nil while
    /// the phone lies flat or is half-way between two ways (it keeps the last).
    static func howHeld(x: Double, y: Double, z: Double, was: UIDeviceOrientation) -> UIDeviceOrientation? {
        if abs(z) > 0.85 { return nil }
        if was != .unknown && abs(abs(x) - abs(y)) < 0.3 { return nil }
        if abs(y) >= abs(x) { return y < 0 ? .portrait : .portraitUpsideDown }
        return x < 0 ? .landscapeLeft : .landscapeRight
    }

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
            self.nextDue = 0
            self.counted = 0
            self.countedAt = 0
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
