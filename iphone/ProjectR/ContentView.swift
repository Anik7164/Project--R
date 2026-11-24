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
            case .role:
                RoleView()
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
                    Text("On the PC, start Project R (phones on the Wi-Fi can join) and click Connect a phone. Then watch the match, or operate it with the password the PC shows.")
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
                                    Text(pc.hosts.joined(separator: ", ") + " · port \(pc.port)")
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
                    Button("Connect") {
                        guard let pc = PC.parse(address) else {
                            model.message = "That does not look like an address. Type it as the PC shows it, for example 192.168.0.5:8000."
                            return
                        }
                        model.connect(pc)
                    }
                } header: {
                    Text("Or type the address")
                } footer: {
                    Text("For Wi-Fi that hides PCs from each other (universities, hotels). The PC shows its address under Connect a phone.")
                }
            }
            .navigationTitle("Project R")
            .refreshable { finder.restart() }
        }
    }
}

/// Found the PC: watch the match, or operate it with the operator password.
struct RoleView: View {
    @EnvironmentObject private var model: AppModel
    @State private var password = ""
    @State private var name = Store.cameraName
    @FocusState private var typing: Bool

    var body: some View {
        NavigationStack {
            List {
                Section {
                    HStack(spacing: 14) {
                        Image(systemName: "desktopcomputer")
                            .font(.system(size: 30))
                            .foregroundStyle(Color.pitch)
                        VStack(alignment: .leading, spacing: 4) {
                            Text(model.current?.name ?? "PC").font(.title3.bold())
                            Text("Connected. Watch the match, or operate it with the password.")
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
                        keepName()
                        model.watch()
                    } label: {
                        Label("Watch the match", systemImage: "play.tv")
                            .font(.headline)
                    }
                } header: {
                    Text("Viewer")
                } footer: {
                    Text("The picture, the Bengali commentary and the voice. No password.")
                }

                Section {
                    if model.needsPassword {
                        SecureField("Operator password", text: $password)
                            .textContentType(.password)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            .submitLabel(.go)
                            .focused($typing)
                            .onSubmit(operate)
                    }
                    Button(action: operate) {
                        HStack {
                            Label("Operate", systemImage: "slider.horizontal.3")
                                .font(.headline)
                            if model.checking {
                                Spacer()
                                ProgressView()
                            }
                        }
                    }
                    .disabled(model.checking)
                } header: {
                    Text("Operator")
                } footer: {
                    Text(model.needsPassword
                         ? "Start and stop the match, choose the cameras, press the event buttons. The PC shows the password under Connect a phone."
                         : "This PC needs no password.")
                }

                Section {
                    TextField("This phone's name, for example: Goal side", text: $name)
                        .autocorrectionDisabled()
                        .onSubmit(keepName)
                } header: {
                    Text("This phone as a camera")
                } footer: {
                    Text("While the app is open, the PC lists this phone as a camera. When the operator chooses it there, it starts filming by itself.")
                }

                Section {
                    Button("Another PC") { model.disconnect() }
                }
            }
            .navigationTitle("Watch or operate")
            .onAppear {
                if password.isEmpty { password = model.savedKey }
            }
        }
    }

    private func keepName() {
        Store.cameraName = name.trimmingCharacters(in: .whitespaces)
    }

    private func operate() {
        typing = false
        keepName()
        model.operate(password: password)
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
                device: (id: Store.deviceID, name: Store.phoneName),
                onChangePC: { model.chooseRole() },
                onCamera: { model.showCamera = true },
                onCameraRequest: { model.cameraRequested(token: $0) },
                onCameraStop: { model.cameraReleased() },
                onFailed: { model.failed($0) })
            .ignoresSafeArea(edges: [.bottom, .horizontal])
            .background(Color.night.ignoresSafeArea())
            .sheet(isPresented: $model.showCamera) {
                if let base = model.base {
                    CameraScreen(sender: model.camera, base: base, key: model.operatorKey)
                }
            }
    }
}
