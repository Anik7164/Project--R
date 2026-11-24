import AVFoundation
import SwiftUI

@main
struct ProjectRApp: App {
    @StateObject private var model = AppModel()
    @StateObject private var finder = ServerFinder()

    init() {
        // The commentary voice plays also with the silent switch on.
        try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .spokenAudio, options: [])
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(model)
                .environmentObject(finder)
                .preferredColorScheme(.dark)
                .tint(Color.pitch)
                .onAppear {
                    finder.start()          // also what makes iOS ask for Local Network
                    model.launch()
                }
                .onOpenURL { url in
                    model.open(url)         // projectr://connect?… from the iPhone camera
                }
        }
    }
}
