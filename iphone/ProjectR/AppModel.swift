import AVFoundation
import Foundation

/// The app's brain: which PC, whether it answers, watching or operating,
/// and what is on screen.
@MainActor
final class AppModel: ObservableObject {
    enum Screen: Equatable {
        case choose
        case connecting(String)
        case role                   // found the PC: watch, or operate with the password
        case match(URL)
    }

    /// What the PC said about this phone (/api/state).
    struct Answer {
        var isOperator: Bool        // this key runs the match
        var password: Bool          // operating needs the operator password
    }

    @Published var screen: Screen = .choose
    @Published var message = ""
    @Published var scanning = false
    @Published var showCamera = false
    @Published private(set) var checking = false
    @Published private(set) var current: PC?
    @Published private(set) var base: URL?
    @Published private(set) var last: PC? = Store.lastPC
    @Published private(set) var needsPassword = true
    /// The operator password while operating; empty while watching.
    @Published private(set) var operatorKey = ""

    let camera = CameraSender()
    private var attempt = 0
    private var launched = false

    /// At start: straight back to the PC used last time, as before
    /// (watching, or operating when the password still works).
    func launch() {
        guard !launched else { return }
        launched = true
        if let pc = last { connect(pc, auto: true) }
    }

    /// Tries the PC's addresses in turn. The first that answers: the
    /// Watch or Operate screen (at launch: what was chosen last time).
    func connect(_ given: PC, auto: Bool = false) {
        attempt += 1
        let mine = attempt
        var pc = given
        if pc.key.isEmpty {
            pc.key = Store.key(for: pc.name)
        } else {
            Store.setKey(pc.key, for: pc.name)      // an older PC's QR code carried it
        }
        message = ""
        scanning = false
        screen = .connecting(pc.name)
        Task {
            let timeout: TimeInterval = pc.hosts.count > 1 ? 4 : 8
            for host in pc.hosts {
                guard mine == attempt else { return }
                guard let base = pc.base(host),
                      let answer = await AppModel.ask(base, key: pc.key, timeout: timeout) else { continue }
                guard mine == attempt else { return }
                var chosen = pc
                chosen.hosts = [host] + pc.hosts.filter { $0 != host }   // the one that worked first next time
                current = chosen
                self.base = base
                last = chosen
                Store.lastPC = chosen
                needsPassword = answer.password
                let role = Store.role(for: pc.name)
                if auto && role == "watch" {
                    watch()
                } else if auto && role == "operate" && answer.isOperator {
                    openMatch(operating: pc.key)
                } else {
                    if auto && role == "operate" {
                        message = "The operator password was changed on the PC. Type the new one, or watch."
                    }
                    screen = .role
                }
                return
            }
            guard mine == attempt else { return }
            screen = .choose
            message = "Could not reach \(pc.name) (\(pc.hosts.joined(separator: ", ")), port \(pc.port)). " +
                "Is Project R running on the PC with “phones on the Wi-Fi can join”? Is this phone on the " +
                "same Wi-Fi? Is Local Network on for Project R (Settings › Privacy & Security › Local Network)?"
        }
    }

    /// The password last used on this PC, filled in for the person.
    var savedKey: String {
        guard let current else { return "" }
        return Store.key(for: current.name)
    }

    /// Watch: the picture, the lines and the voice; no buttons.
    func watch() {
        guard current != nil else { return }
        message = ""
        if let current { Store.setRole("watch", for: current.name) }
        openMatch(operating: "")
    }

    /// Operate: only when the PC says the password is right.
    func operate(password given: String) {
        guard let base, let pc = current, !checking else { return }
        let password = given.trimmingCharacters(in: .whitespacesAndNewlines)
        if needsPassword && password.isEmpty {
            message = "Type the operator password. The PC shows it under Connect a phone."
            return
        }
        checking = true
        message = ""
        Task {
            let answer = await AppModel.ask(base, key: password, timeout: 8)
            checking = false
            guard screen == .role, current == pc else { return }
            guard let answer else {
                message = "No answer from \(pc.name). Is Project R still running on the PC?"
                return
            }
            guard answer.isOperator else {
                message = "That password is not right. The PC shows it under Connect a phone."
                return
            }
            Store.setKey(password, for: pc.name)
            Store.setRole("operate", for: pc.name)
            var known = pc
            known.key = password
            current = known
            last = known
            Store.lastPC = known
            openMatch(operating: password)
        }
    }

    private func openMatch(operating key: String) {
        guard let base else { return }
        operatorKey = key
        screen = .match(AppModel.page(base, key: key))
    }

    func cancel() {
        attempt += 1
        screen = .choose
    }

    /// "PC … change" or "Operate" on the match page: back to Watch or Operate.
    func chooseRole() {
        camera.stop()
        showCamera = false
        message = ""
        screen = current == nil ? .choose : .role
    }

    /// Back to the list of PCs.
    func disconnect() {
        attempt += 1
        camera.stop()
        showCamera = false
        message = ""
        screen = .choose
    }

    func failed(_ text: String) {
        guard case .match = screen else { return }
        camera.stop()
        showCamera = false
        screen = .choose
        message = "Lost the PC: \(text)"
    }

    /// The operator chose this iPhone as a camera on the PC: film now.
    func cameraRequested(token: String) {
        guard let base, case .match = screen else { return }
        showCamera = true
        guard !camera.running else { return }
        camera.start(base: base, key: operatorKey, name: Store.phoneName, back: Store.cameraBack,
                     token: token, app: Store.deviceID)
    }

    /// The PC no longer uses this phone as a camera (another device was
    /// chosen, fewer cameras, or the match stopped): stop and close.
    func cameraReleased() {
        camera.stop()
        showCamera = false
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

    /// Does a Project R server answer at `base`, and is `key` its operator password?
    static func ask(_ base: URL, key: String, timeout: TimeInterval) async -> Answer? {
        var request = URLRequest(url: base.appendingPathComponent("api/state"), timeoutInterval: timeout)
        request.setValue(key, forHTTPHeaderField: "X-Key")
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = timeout
        configuration.timeoutIntervalForResource = timeout
        configuration.waitsForConnectivity = false
        let session = URLSession(configuration: configuration)
        defer { session.finishTasksAndInvalidate() }
        guard let (data, response) = try? await session.data(for: request),
              (response as? HTTPURLResponse)?.statusCode == 200,
              let state = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let runs = state["operator"] as? Bool else { return nil }
        // An older PC does not say: it had a key whenever phones could join.
        return Answer(isOperator: runs, password: state["password"] as? Bool ?? true)
    }

    /// The match page. "?key=" is always given (empty to watch), so the page
    /// forgets a password it remembered from operating before.
    static func page(_ base: URL, key: String) -> URL {
        guard var parts = URLComponents(url: base, resolvingAgainstBaseURL: false) else { return base }
        parts.path = "/"
        parts.queryItems = [URLQueryItem(name: "key", value: key)]
        return parts.url ?? base
    }
}
