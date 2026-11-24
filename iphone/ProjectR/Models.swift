import Foundation
import UIKit

/// A PC that runs Project R. One PC can have several addresses (Wi-Fi,
/// Ethernet, hotspot), so it keeps the whole list and the app tries each.
struct PC: Identifiable, Codable, Equatable {
    var name: String
    var hosts: [String]
    var port: Int
    var key: String = ""

    var id: String { "\(name)|\(hosts.joined(separator: ","))|\(port)" }

    func base(_ host: String) -> URL? {
        URL(string: "http://\(host):\(port)")
    }

    /// Reads what a QR code or the person gives:
    /// - the pairing link  projectr://connect?h=192.168.1.10,10.0.0.5&p=8000&n=PC
    ///   (it only finds the PC; older PCs also put the key in it: &k=KEY)
    /// - an operator address  http://192.168.1.10:8000/?key=KEY  (older PCs)
    /// - a typed  192.168.1.10, 192.168.1.10:8000  or  ws://192.168.1.10:8000/ws
    static func parse(_ text: String) -> PC? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        if trimmed.lowercased().hasPrefix("projectr:") {
            guard let parts = URLComponents(string: trimmed) else { return nil }
            var query: [String: String] = [:]
            for item in parts.queryItems ?? [] { query[item.name] = item.value ?? "" }
            let hosts = (query["h"] ?? "")
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty }
            guard !hosts.isEmpty else { return nil }
            let name = (query["n"] ?? "").isEmpty ? hosts[0] : query["n"]!
            return PC(name: name, hosts: hosts, port: Int(query["p"] ?? "") ?? 8000, key: query["k"] ?? "")
        }
        var address = trimmed
        for prefix in ["ws://", "wss://"] where address.lowercased().hasPrefix(prefix) {
            address = "http://" + address.dropFirst(prefix.count)
        }
        if !address.lowercased().hasPrefix("http://") && !address.lowercased().hasPrefix("https://") {
            address = "http://" + address
        }
        guard let parts = URLComponents(string: address), let host = parts.host, !host.isEmpty else {
            return nil
        }
        let key = parts.queryItems?.first(where: { $0.name == "key" })?.value ?? ""
        return PC(name: host, hosts: [host], port: parts.port ?? 8000, key: key)
    }
}

/// What the app remembers between launches.
enum Store {
    private static let defaults = UserDefaults.standard

    static var lastPC: PC? {
        get {
            guard let data = defaults.data(forKey: "lastPC") else { return nil }
            return try? JSONDecoder().decode(PC.self, from: data)
        }
        set {
            if let newValue, let data = try? JSONEncoder().encode(newValue) {
                defaults.set(data, forKey: "lastPC")
            } else {
                defaults.removeObject(forKey: "lastPC")
            }
        }
    }

    /// The operator password of a PC, by its name, once it was right.
    static func key(for name: String) -> String {
        defaults.string(forKey: "key \(name)") ?? ""
    }

    static func setKey(_ key: String, for name: String) {
        defaults.set(key, forKey: "key \(name)")
    }

    /// "watch" or "operate": what this phone chose on that PC last time.
    static func role(for name: String) -> String {
        defaults.string(forKey: "role \(name)") ?? ""
    }

    static func setRole(_ role: String, for name: String) {
        defaults.set(role, forKey: "role \(name)")
    }

    /// What the person called this phone (the PC lists it as a camera by it).
    static var cameraName: String {
        get { defaults.string(forKey: "cameraName") ?? "" }
        set { defaults.set(newValue, forKey: "cameraName") }
    }

    @MainActor static var phoneName: String {
        let given = cameraName.trimmingCharacters(in: .whitespaces)
        return given.isEmpty ? UIDevice.current.name : given
    }

    /// The back camera (true) or the front one, as last chosen.
    static var cameraBack: Bool {
        get { defaults.object(forKey: "cameraBack") as? Bool ?? true }
        set { defaults.set(newValue, forKey: "cameraBack") }
    }

    /// This phone, the same every time: the PC keeps one camera entry for it.
    static var deviceID: String {
        if let id = defaults.string(forKey: "deviceID"), !id.isEmpty { return id }
        let id = UUID().uuidString.lowercased()
        defaults.set(id, forKey: "deviceID")
        return id
    }
}
