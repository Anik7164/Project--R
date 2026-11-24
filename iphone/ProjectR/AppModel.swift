import AVFoundation
import Foundation

/// The app's brain: which PC, whether it answers, and what is on screen.
@MainActor
final class AppModel: ObservableObject {
    enum Screen: Equatable {
        case choose
        case connecting(String)
        case match(URL)
    }

    @Published var screen: Screen = .choose
    @Published var message = ""
    @Published var scanning = false
    @Published var showCamera = false
    @Published private(set) var current: PC?
    @Published private(set) var base: URL?
    @Published private(set) var last: PC? = Store.lastPC

    let camera = CameraSender()
    private var attempt = 0
    private var launched = false

    /// At start: straight back to the PC used last time.
    func launch() {
        guard !launched else { return }
        launched = true
        if let pc = last { connect(pc) }
    }

    /// Tries the PC's addresses in turn and opens the first that answers.
    func connect(_ given: PC) {
        attempt += 1
        let mine = attempt
        var pc = given
        if pc.key.isEmpty {
            pc.key = Store.key(for: pc.name)
        } else {
            Store.setKey(pc.key, for: pc.name)
        }
        message = ""
        scanning = false
        screen = .connecting(pc.name)
        Task {
            let timeout: TimeInterval = pc.hosts.count > 1 ? 4 : 8
            for host in pc.hosts {
                guard mine == attempt else { return }
                guard let base = pc.base(host) else { continue }
                if await AppModel.answers(base, key: pc.key, timeout: timeout) {
                    guard mine == attempt else { return }
                    var chosen = pc
                    chosen.hosts = [host] + pc.hosts.filter { $0 != host }   // the one that worked first next time
                    current = chosen
                    self.base = base
                    last = chosen
                    Store.lastPC = chosen
                    screen = .match(AppModel.page(base, key: pc.key))
                    return
                }
            }
            guard mine == attempt else { return }
            screen = .choose
            message = "Could not reach \(pc.name) (\(pc.hosts.joined(separator: ", ")), port \(pc.port)). " +
                "Is Project R running on the PC with “phones on the Wi-Fi can join”? Is this phone on the " +
                "same Wi-Fi? Is Local Network on for Project R (Settings › Privacy & Security › Local Network)?"
        }
    }

    func cancel() {
        attempt += 1
        screen = .choose
    }

    /// Back to the list of PCs ("PC … change" on the match page).
    func disconnect() {
        attempt += 1
        camera.stop()
        showCamera = false
        screen = .choose
    }

    func failed(_ text: String) {
        guard case .match = screen else { return }
        camera.stop()
        showCamera = false
        screen = .choose
        message = "Lost the PC: \(text)"
    }

    func startScan() {
        message = ""
        AVCaptureDevice.requestAccess(for: .video) { granted in
            Task { @MainActor in
                if granted {
                    self.scanning = true
                } else {
                    self.message = "The camera is not allowed for Project R. Turn it on in Settings › Project R."
                }
            }
        }
    }

    func scanned(_ text: String) {
        scanning = false
        if let pc = PC.parse(text) {
            connect(pc)
        } else {
            message = "That QR code is not a Project R code. On the PC, click “Connect a phone”."
        }
    }

    /// projectr://connect?… opened by the iPhone camera.
    func open(_ url: URL) {
        scanned(url.absoluteString)
    }

    /// Does a Project R server answer at `base`?
    static func answers(_ base: URL, key: String, timeout: TimeInterval) async -> Bool {
        var request = URLRequest(url: base.appendingPathComponent("api/state"), timeoutInterval: timeout)
        request.setValue(key, forHTTPHeaderField: "X-Key")
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = timeout
        configuration.timeoutIntervalForResource = timeout
        configuration.waitsForConnectivity = false
        let session = URLSession(configuration: configuration)
        defer { session.finishTasksAndInvalidate() }
        guard let (data, response) = try? await session.data(for: request),
              (response as? HTTPURLResponse)?.statusCode == 200 else { return false }
        return String(decoding: data.prefix(4096), as: UTF8.self).contains("\"operator\"")
    }

    /// The match page, as the operator when the key is known.
    static func page(_ base: URL, key: String) -> URL {
        guard var parts = URLComponents(url: base, resolvingAgainstBaseURL: false) else { return base }
        parts.path = "/"
        if !key.isEmpty { parts.queryItems = [URLQueryItem(name: "key", value: key)] }
        return parts.url ?? base
    }
}
