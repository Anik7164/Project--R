import SwiftUI
import UIKit
import WebKit

/// The PC's match page (the same page Safari shows), inside the app.
/// The page knows it is in the app (window.PROJECT_R_NATIVE): its
/// "PC … change" calls back here, and "Use this device as a camera" opens
/// the native camera, which can film over plain http where Safari cannot.
struct WebPage: UIViewRepresentable {
    let url: URL
    let onChangePC: () -> Void
    let onCamera: () -> Void
    let onFailed: (String) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.allowsInlineMediaPlayback = true
        configuration.mediaTypesRequiringUserActionForPlayback = []     // the commentary voice plays by itself
        let flag = WKUserScript(source: "window.PROJECT_R_NATIVE = true;", injectionTime: .atDocumentStart,
                                forMainFrameOnly: true)
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
            if (message.body as? String) == "change" { parent.onChangePC() }
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
