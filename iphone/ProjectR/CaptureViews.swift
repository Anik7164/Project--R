import AVFoundation
import SwiftUI
import UIKit

/// The camera picture on screen (the QR scanner and the match camera).
final class PreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

    var preview: AVCaptureVideoPreviewLayer {
        // swiftlint:disable:next force_cast
        layer as! AVCaptureVideoPreviewLayer
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        guard let connection = preview.connection, connection.isVideoOrientationSupported,
              let scene = window?.windowScene else { return }
        switch scene.interfaceOrientation {
        case .landscapeLeft: connection.videoOrientation = .landscapeLeft
        case .landscapeRight: connection.videoOrientation = .landscapeRight
        case .portraitUpsideDown: connection.videoOrientation = .portraitUpsideDown
        default: connection.videoOrientation = .portrait
        }
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    var fill = false

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.backgroundColor = .black
        view.preview.session = session
        view.preview.videoGravity = fill ? .resizeAspectFill : .resizeAspect
        return view
    }

    func updateUIView(_ view: PreviewView, context: Context) {
        view.setNeedsLayout()
    }
}

/// Reads the PC's QR code with the back camera.
final class QRReader: NSObject, ObservableObject, AVCaptureMetadataOutputObjectsDelegate {
    let session = AVCaptureSession()
    var found: ((String) -> Void)?
    private var done = false
    private var configured = false

    func start() {
        done = false
        let session = self.session
        DispatchQueue.global(qos: .userInitiated).async {
            if !self.configured { self.configure() }
            if !session.isRunning { session.startRunning() }
        }
    }

    func stop() {
        let session = self.session
        DispatchQueue.global(qos: .userInitiated).async {
            if session.isRunning { session.stopRunning() }
        }
    }

    private func configure() {
        configured = true
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        guard let device = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: device),
              session.canAddInput(input) else { return }
        session.addInput(input)
        let output = AVCaptureMetadataOutput()
        guard session.canAddOutput(output) else { return }
        session.addOutput(output)
        output.setMetadataObjectsDelegate(self, queue: .main)
        output.metadataObjectTypes = [.qr]
    }

    func metadataOutput(_ output: AVCaptureMetadataOutput, didOutput metadataObjects: [AVMetadataObject],
                        from connection: AVCaptureConnection) {
        guard !done,
              let text = metadataObjects.compactMap({ $0 as? AVMetadataMachineReadableCodeObject }).first?.stringValue
        else { return }
        done = true
        stop()
        found?(text)
    }
}

struct ScannerSheet: View {
    @EnvironmentObject private var model: AppModel
    @StateObject private var reader = QRReader()

    var body: some View {
        ZStack(alignment: .bottom) {
            CameraPreview(session: reader.session, fill: true)
                .ignoresSafeArea()
            RoundedRectangle(cornerRadius: 24)
                .stroke(Color.white.opacity(0.85), lineWidth: 3)
                .frame(width: 250, height: 250)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            VStack(spacing: 10) {
                Text("Point at the QR code on the PC")
                    .font(.headline)
                Text("Project R on the PC › Connect a phone")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                Button("Cancel") { model.scanning = false }
                    .buttonStyle(.bordered)
            }
            .padding()
            .frame(maxWidth: .infinity)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
            .padding()
        }
        .background(Color.black)
        .onAppear {
            reader.found = { text in model.scanned(text) }
            reader.start()
        }
        .onDisappear { reader.stop() }
    }
}

/// "Use this device as a camera" on the match page opens this, and so does
/// the operator choosing this iPhone on the PC (then it is already filming).
struct CameraScreen: View {
    @ObservedObject var sender: CameraSender
    let base: URL
    let key: String
    @Environment(\.dismiss) private var dismiss
    @State private var name = Store.cameraName
    @State private var back = Store.cameraBack

    var body: some View {
        NavigationStack {
            VStack(spacing: 14) {
                ZStack(alignment: .topLeading) {
                    CameraPreview(session: sender.session)
                    if sender.running {
                        Text("LIVE · \(sender.rate) pictures/s · \(sender.size) · \(sender.wide ? "wide" : "tall")")
                            .font(.caption.bold())
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.red, in: Capsule())
                            .padding(10)
                    } else {
                        Text("The camera starts when you press Start, or when the operator chooses this iPhone on the PC.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                }
                .background(Color.black)
                .clipShape(RoundedRectangle(cornerRadius: 12))

                Text(sender.status)
                    .font(.subheadline)
                    .multilineTextAlignment(.center)

                if sender.running {
                    Button(role: .destructive) {
                        sender.stop()
                    } label: {
                        Text("Stop").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                } else {
                    TextField("Camera name, for example: Goal side", text: $name)
                        .textFieldStyle(.roundedBorder)
                    Picker("Lens", selection: $back) {
                        Text("Back camera").tag(true)
                        Text("Front camera").tag(false)
                    }
                    .pickerStyle(.segmented)
                    Button {
                        Store.cameraName = name.trimmingCharacters(in: .whitespaces)
                        Store.cameraBack = back
                        sender.start(base: base, key: key, name: Store.phoneName, back: back, app: Store.deviceID)
                    } label: {
                        Text("Start sending to the PC").frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                }
                Text("On the PC, choose this iPhone (“\(Store.phoneName)”) for Camera 1, 2 or 3. Keep this screen open while it films; the phone stays awake.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            .padding()
            .navigationTitle("This phone as a camera")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
            }
        }
        .onDisappear { sender.stop() }
    }
}
