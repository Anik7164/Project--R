import Foundation
import Network

/// Lists the PCs that announce "_projectr._tcp" on the Wi-Fi (Bonjour).
/// The PC puts its addresses, port and name in the TXT record
/// (webapp/pairing.py). Browsing is also what makes iOS ask for Local Network.
final class ServerFinder: ObservableObject {
    @Published private(set) var found: [PC] = []
    @Published private(set) var problem = ""

    private var browser: NWBrowser?

    func start() {
        guard browser == nil else { return }
        let browser = NWBrowser(for: .bonjourWithTXTRecord(type: "_projectr._tcp", domain: nil),
                                using: NWParameters())
        browser.stateUpdateHandler = { [weak self] state in
            switch state {
            case .failed(let error), .waiting(let error):
                self?.problem = ServerFinder.explain(error)
            case .ready:
                self?.problem = ""
            default:
                break
            }
        }
        browser.browseResultsChangedHandler = { [weak self] results, _ in
            let pcs = results.compactMap(ServerFinder.pc(from:))
            self?.found = pcs.sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
        }
        browser.start(queue: .main)
        self.browser = browser
    }

    func restart() {
        browser?.cancel()
        browser = nil
        found = []
        start()
    }

    private static func pc(from result: NWBrowser.Result) -> PC? {
        guard case let .service(serviceName, _, _, _) = result.endpoint else { return nil }
        var txt: [String: String] = [:]
        if case let .bonjour(record) = result.metadata {
            txt = record.dictionary
        }
        let hosts = (txt["hosts"] ?? "").split(separator: ",").map(String.init).filter { !$0.isEmpty }
        guard !hosts.isEmpty else { return nil }
        let name = (txt["name"] ?? "").isEmpty ? serviceName : txt["name"]!
        return PC(name: name, hosts: hosts, port: Int(txt["port"] ?? "") ?? 8000, key: Store.key(for: name))
    }

    private static func explain(_ error: NWError) -> String {
        if case let .dns(code) = error, code == -65570 {        // kDNSServiceErr_PolicyDenied
            return "Local Network is off for Project R. Turn it on in Settings › Privacy & Security › Local Network."
        }
        return "Cannot look for PCs: \(error.localizedDescription)"
    }
}
