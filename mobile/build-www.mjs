// Puts the web app's pages into www/, the folder the iPhone app ships.
// They are the very same files the PC serves (webapp/static), plus a
// shell.js that tells them they live in the app and must ask for the PC.
//
//   node build-www.mjs

import { cpSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const source = join(here, "..", "webapp", "static");
const www = join(here, "www");

rmSync(www, { recursive: true, force: true });      // www/ is only ever a copy
mkdirSync(join(www, "static"), { recursive: true });
cpSync(source, join(www, "static"), { recursive: true });
for (const page of ["index.html", "camera.html"]) cpSync(join(source, page), join(www, page));
writeFileSync(join(www, "static", "shell.js"),
  "// In the iPhone app: the pages ask which PC to connect to (common.js).\n" +
  "window.PROJECT_R_SHELL = true;\n");
console.log(`www/ is ready (${www})`);
