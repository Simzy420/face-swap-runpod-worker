/** Production HTTP adapter for the existing web app and API handlers. */
import { createServer } from "node:http";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { extname, isAbsolute, relative, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { handleGenerateRequest } from "./gradioProxy.js";
import { handleRunpodJobRequest, handleRunpodResultRequest } from "./runpodRoutes.js";
import { handleVideoRequest } from "./videoProxy.js";
import { handleBalanceRequest, handleStopRequest } from "./runpodAdmin.js";

const API_HANDLERS = {
  "/api/generate": handleGenerateRequest,
  "/api/job": handleRunpodJobRequest,
  "/api/result": handleRunpodResultRequest,
  "/api/video": handleVideoRequest,
  "/api/balance": handleBalanceRequest,
  "/api/stop": handleStopRequest,
};
const CONTENT_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".webmanifest": "application/manifest+json",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

function json(res, status, body) {
  res.writeHead(status, { "Content-Type": "application/json", "Cache-Control": "no-store" });
  res.end(JSON.stringify(body));
}

function readBody(req, maxBytes) {
  return new Promise((resolveBody, reject) => {
    let size = 0;
    let chunks = [];
    req.on("data", (chunk) => {
      size += chunk.length;
      if (size > maxBytes) {
        chunks = [];
        reject(Object.assign(new Error("Upload is too large."), { statusCode: 413 }));
      } else {
        chunks.push(chunk);
      }
    });
    req.once("end", () => resolveBody(Buffer.concat(chunks)));
    req.once("error", reject);
  });
}

export function createApp({
  staticDir = fileURLToPath(new URL("../dist/", import.meta.url)),
  handlers = API_HANDLERS,
  maxBodyBytes = 8 * 1024 * 1024,
} = {}) {
  const root = resolve(staticDir);
  return createServer(async (req, res) => {
    try {
      const url = new URL(req.url, "http://app.internal");
      if (url.pathname === "/healthz" && req.method === "GET") {
        return json(res, 200, { status: "ok" });
      }
      const handler = handlers[url.pathname];
      if (handler) {
        const headers = new Headers();
        for (const [name, value] of Object.entries(req.headers)) {
          if (value != null) headers.set(name, Array.isArray(value) ? value.join(", ") : value);
        }
        const init = { method: req.method, headers };
        if (req.method !== "GET" && req.method !== "HEAD") init.body = await readBody(req, maxBodyBytes);
        const response = await handler(new Request(url, init));
        res.writeHead(response.status, Object.fromEntries(response.headers));
        if (req.method === "HEAD" || !response.body) return res.end();
        await pipeline(Readable.fromWeb(response.body), res);
        return;
      }
      if (url.pathname.startsWith("/api/")) return json(res, 404, { error: "Unknown API route." });
      if (req.method !== "GET" && req.method !== "HEAD") return json(res, 405, { error: "Method not allowed." });
      let pathname;
      try {
        pathname = decodeURIComponent(url.pathname);
      } catch {
        return json(res, 400, { error: "Invalid URL." });
      }
      if (pathname.split("/").some((part) => part.startsWith("."))) return json(res, 404, { error: "Not found." });
      let file = resolve(root, `.${pathname}`);
      const rel = relative(root, file);
      if (rel === ".." || rel.startsWith(`..${sep}`) || isAbsolute(rel)) return json(res, 404, { error: "Not found." });
      let info;
      try {
        info = await stat(file);
        if (info.isDirectory()) {
          file = resolve(file, "index.html");
          info = await stat(file);
        }
      } catch (error) {
        if (error.code === "ENOENT" || error.code === "ENOTDIR") return json(res, 404, { error: "Not found." });
        throw error;
      }
      if (!info.isFile()) return json(res, 404, { error: "Not found." });
      res.writeHead(200, {
        "Content-Type": CONTENT_TYPES[extname(file)] || "application/octet-stream",
        "Content-Length": info.size,
        "Cache-Control": pathname.startsWith("/assets/") ? "public, max-age=31536000, immutable" : "no-cache",
      });
      if (req.method === "HEAD") return res.end();
      await pipeline(createReadStream(file), res);
    } catch (error) {
      if (res.headersSent) return res.destroy();
      const status = error.statusCode === 413 ? 413 : 500;
      json(res, status, { error: status === 413 ? "Upload is too large." : "The request could not be completed." });
    }
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const server = createApp();
  const port = Number(process.env.PORT || 7860);
  server.listen(port, "0.0.0.0", () => console.log(`Web app listening on port ${port}`));
  process.once("SIGTERM", () => server.close());
  process.once("SIGINT", () => server.close());
}
