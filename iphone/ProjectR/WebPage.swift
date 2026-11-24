import SwiftUI
import UIKit
import WebKit

/// The PC's match page (the same page Safari shows), inside the app.
/// The page knows it is in the app (window.PROJECT_R_NATIVE): its
/// "PC … change" and "Operate" call back here, and "Use this device as a
/// camera" opens the native camera, which can film over plain http where
/// Safari cannot. It also knows which phone this is (window.PROJECT_R_DEVICE):
/// the PC lists the phone as a camera, and when the operator chooses it the
/// page passes the PC's request on ({action: "camera", token}), and its
/// "no longer needed" ({action: "camera_stop"}).
struct WebPage: UIViewRepresentable {
    let url: URL
    let device: (id: String, name: String)
    let onChangePC: () -> Void
    let onCamera: () -> Void
    let onCameraRequest: (String) -> Void
    let onCameraStop: () -> Void
    let onFailed: (String) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.allowsInlineMediaPlayback = true
        configuration.mediaTypesRequiringUserActionForPlayback = []     // the commentary voice plays by itself
        let flag = WKUserScript(source: "window.PROJECT_R_NATIVE = true;" + WebPage.deviceScript(device),
                                injectionTime: .atDocumentStart, forMainFrameOnly: true)
        configuration.userContentController.addUserScript(flag)
        configuration.userContentController.add(context.coordinator, name: "projectr")
        let web = WKWebView(frame: .zero, configuration: configuration)
        web.navigationDelegate = context.coordinator
        web.isOpaque = false
        web.backgroundColor = UIColor(red: 0.043, green: 0.071, blue: 0.063, alpha: 1)
        web.scrollView.backgroundColor = web.backgroundColor
        web.allowsBackForwardNavigationGestures = false
        context.coordinator.loaded = url
        web.load(URLRequest(url: url))
        return web
    }

    func updateUIView(_ web: WKWebView, context: Context) {
        context.coordinator.parent = self
        if context.coordinator.loaded != url {
            context.coordinator.loaded = url
            web.load(URLRequest(url: url))
        }
    }

    /// window.PROJECT_R_DEVICE = {"id": …, "name": …}, as JSON so any name is safe.
    static func deviceScript(_ device: (id: String, name: String)) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: ["id": device.id, "name": device.name]),
              let json = String(data: data, encoding: .utf8) else { return "" }
        return " window.PROJECT_R_DEVICE = \(json);"
    }

    static func dismantleUIView(_ web: WKWebView, coordinator: Coordinator) {
        web.configuration.userContentController.removeScriptMessageHandler(forName: "projectr")
        web.stopLoading()
    }

    final class Coordinator: NSObject, WKNavigationDelegate, WKScriptMessageHandler {
        var parent: WebPage
        var loaded: URL?

        init(_ parent: WebPage) {
            self.parent = parent
        }

        func userContentController(_ userContentController: WKUserContentController,
                                   didReceive message: WKScriptMessage) {
            if let text = message.body as? String {
                if text == "change" { parent.onChangePC() }
                return
            }
            guard let body = message.body as? [String: Any] else { return }
            switch body["action"] as? String {
            case "camera":
                if let token = body["token"] as? String, !token.isEmpty { parent.onCameraRequest(token) }
            case "camera_stop":
                parent.onCameraStop()          // the PC no longer uses this phone's camera
            default:
                break
            }
        }

        func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
                     decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
            guard let target = navigationAction.request.url else {
                decisionHandler(.allow)
                return
            }
            if target.path.hasSuffix("camera.html") {
                decisionHandler(.cancel)
                parent.onCamera()
                return
            }
            let mainFrame = navigationAction.targetFrame?.isMainFrame ?? true
            if mainFrame, let scheme = target.scheme, scheme.hasPrefix("http"),
               let host = target.host, host != loaded?.host {
                decisionHandler(.cancel)                 // another site: Safari
                UIApplication.shared.open(target)
                return
            }
            decisionHandler(.allow)
        }

        func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!,
                     withError error: Error) {
            if (error as NSError).code == NSURLErrorCancelled { return }
            parent.onFailed(error.localizedDescription)
        }

        func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
            webView.reload()
        }
    }
}
