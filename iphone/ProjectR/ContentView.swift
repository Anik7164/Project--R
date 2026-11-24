import SwiftUI

extension Color {
    static let pitch = Color(red: 0.208, green: 0.761, blue: 0.416)
    static let night = Color(red: 0.043, green: 0.071, blue: 0.063)
}

struct ContentView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Group {
            switch model.screen {
            case .choose:
                ConnectView()
            case .connecting(let name):
                ConnectingView(name: name)
            case .match(let url):
                MatchView(url: url)
            }
        }
        .sheet(isPresented: $model.scanning) {
            ScannerSheet().environmentObject(model)
        }
    }
}

/// Which PC? Found on the Wi-Fi, scanned, or typed.
struct ConnectView: View {
    @EnvironmentObject private var model: AppModel
    @EnvironmentObject private var finder: ServerFinder
    @State private var address = ""
    @State private var key = ""

    var body: some View {
        NavigationStack {
            List {
                Section {
                    HStack(spacing: 14) {
                        Image(systemName: "soccerball")
                            .font(.system(size: 34))
                            .foregroundStyle(Color.pitch)
                        VStack(alignment: .leading, spacing: 4) {
                            Text("লাইভ ধারাভাষ্য").font(.title2.bold())
                            Text("Connect to the PC that runs Project R, on the same Wi-Fi.")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .padding(.vertical, 6)
                }

                if !model.message.isEmpty {
                    Section {
                        Text(model.message)
                            .font(.footnote)
                            .foregroundStyle(.orange)
                    }
                }

                Section {
                    Button {
                        model.startScan()
                    } label: {
                        Label("Scan the QR code on the PC", systemImage: "qrcode.viewfinder")
                            .font(.headline)
                    }
                } footer: {
                    Text("On the PC, start Project R (phones on the Wi-Fi can join) and click Connect a phone. The QR code also makes this phone the operator.")
                }

                Section {
                    if finder.found.isEmpty {
                        HStack(spacing: 10) {
                            if finder.problem.isEmpty { ProgressView() }
                            Text(finder.problem.isEmpty ? "Looking for PCs…" : finder.problem)
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                    }
                    ForEach(finder.found) { pc in
                        Button {
                            model.connect(pc)
                        } label: {
                            HStack {
                                Image(systemName: "desktopcomputer")
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(pc.name).foregroundStyle(.primary)
                                    Text(pc.hosts.joined(separator: ", ") + " · port \(pc.port)" +
                                         (pc.key.isEmpty ? " · watch only (scan to operate)" : " · operator"))
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                            }
                        }
                    }
                } header: {
                    Text("PCs on this Wi-Fi")
                }

                if let last = model.last {
                    Section {
                        Button {
                            model.connect(last)
                        } label: {
                            Label("\(last.name)  (\(last.hosts.first ?? ""):\(last.port))",
                                  systemImage: "clock.arrow.circlepath")
                        }
                    } header: {
                        Text("Last PC")
                    }
                }

                Section {
                    TextField("192.168.0.5:8000", text: $address)
                        .keyboardType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    TextField("Operator key (empty: only watch)", text: $key)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    Button("Connect") {
                        guard var pc = PC.parse(address) else {
                            model.message = "That does not look like an address. Type it as the PC shows it, for example 192.168.0.5:8000."
                            return
                        }
                        let typed = key.trimmingCharacters(in: .whitespaces)
                        if !typed.isEmpty { pc.key = typed }
                        model.connect(pc)
                    }
                } header: {
                    Text("Or type the address")
                } footer: {
                    Text("For Wi-Fi that hides PCs from each other (universities, hotels). The PC shows its address and key under Settings.")
                }
            }
            .navigationTitle("Project R")
            .refreshable { finder.restart() }
        }
    }
}

struct ConnectingView: View {
    @EnvironmentObject private var model: AppModel
    let name: String

    var body: some View {
        VStack(spacing: 18) {
            ProgressView().controlSize(.large)
            Text("Connecting to \(name)…").font(.headline)
            Text("If iOS asks to find devices on your local network, tap Allow.")
                .font(.footnote)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            Button("Cancel") { model.cancel() }
                .buttonStyle(.bordered)
        }
        .padding(30)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.night.ignoresSafeArea())
    }
}

struct MatchView: View {
    @EnvironmentObject private var model: AppModel
    let url: URL

    var body: some View {
        WebPage(url: url,
                onChangePC: { model.disconnect() },
                onCamera: { model.showCamera = true },
                onFailed: { model.failed($0) })
            .ignoresSafeArea(edges: [.bottom, .horizontal])
            .background(Color.night.ignoresSafeArea())
            .sheet(isPresented: $model.showCamera) {
                if let base = model.base {
                    CameraScreen(sender: model.camera, base: base, key: model.current?.key ?? "")
                }
            }
    }
}
