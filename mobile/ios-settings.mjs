// After `cap add ios` / `cap sync ios`: what the generated Xcode project
// still needs for Project R. Safe to run again.
//
//   - the camera (this phone as a match camera) and the local network
//     (the PC that runs the commentary) with the texts iOS shows when asking
//   - plain http and ws to the PC on the Wi-Fi (it has no certificate)
//   - the app icon and a dark launch screen
//
//   node ios-settings.mjs

import { copyFileSync, existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const app = join(here, "ios", "App", "App");
const plistPath = join(app, "Info.plist");
if (!existsSync(plistPath)) {
  console.error("ios/ is not there yet. Run: npx cap add ios");
  process.exit(1);
}

const wanted = {
  NSCameraUsageDescription:
    "<string>Project R sends this camera's picture to your PC, which turns the match into live commentary.</string>",
  NSLocalNetworkUsageDescription:
    "<string>Project R connects to the PC on your Wi-Fi that runs the commentary.</string>",
  NSAppTransportSecurity:
    "<dict>\n\t\t<key>NSAllowsArbitraryLoads</key>\n\t\t<true/>\n" +
    "\t\t<key>NSAllowsLocalNetworking</key>\n\t\t<true/>\n\t</dict>",
};

let plist = readFileSync(plistPath, "utf8");
let added = 0;
for (const [name, value] of Object.entries(wanted)) {
  if (plist.includes(`<key>${name}</key>`)) continue;
  const end = plist.lastIndexOf("</dict>");
  plist = `${plist.slice(0, end)}\t<key>${name}</key>\n\t${value}\n${plist.slice(end)}`;
  added += 1;
}
writeFileSync(plistPath, plist);

const assets = join(app, "Assets.xcassets");
copyFileSync(join(here, "assets", "icon.png"), join(assets, "AppIcon.appiconset", "AppIcon-512@2x.png"));
const splash = join(assets, "Splash.imageset");
for (const file of readdirSync(splash)) {
  if (file.endsWith(".png")) copyFileSync(join(here, "assets", "splash.png"), join(splash, file));
}
console.log(`iOS settings: ${added} Info.plist entries added, icon and launch screen copied`);
