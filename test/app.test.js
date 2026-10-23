import { after, before, test } from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createApp } from "../server/app.js";

let root;
let server;
let origin;

before(async () => {
  root = await mkdtemp(join(tmpdir(), "creative-tools-"));
  await mkdir(join(root, "extend"));
  await writeFile(join(root, "index.html"), "<h1>Templates</h1>");
  await writeFile(join(root, "extend/index.html"), "<h1>Extend</h1>");
  await writeFile(join(root, "config.js"), "window.CONFIG = {};");
  await writeFile(join(root, ".env.local"), "SHOULD_NOT_BE_SERVED");
  server = createApp({ staticDir: root });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  origin = `http://127.0.0.1:${server.address().port}`;
});

after(async () => {
  await new Promise((resolve) => server.close(resolve));
  await rm(root, { recursive: true, force: true });
});

test("serves the site, tool directory indexes, and HEAD responses", async () => {
  assert.equal(await (await fetch(origin)).text(), "<h1>Templates</h1>");
  assert.equal(await (await fetch(`${origin}/extend/`)).text(), "<h1>Extend</h1>");
  const head = await fetch(`${origin}/config.js`, { method: "HEAD" });
  assert.match(head.headers.get("content-type"), /javascript/);
  assert.equal(await head.text(), "");
});

test("denies hidden files, traversal, missing files, and unsupported static methods", async () => {
  for (const path of ["/.env.local", "/%2e%2e%2fpackage.json", "/package.json", "/missing"]) {
    assert.equal((await fetch(origin + path)).status, 404, path);
  }
  assert.equal((await fetch(`${origin}/%ZZ`)).status, 400);
  assert.equal((await fetch(origin, { method: "POST" })).status, 405);
});

test("mounts production API handlers without making provider mutations", async () => {
  assert.deepEqual(await (await fetch(`${origin}/healthz`)).json(), { status: "ok" });
  assert.equal((await fetch(`${origin}/api/generate`)).status, 405);
  assert.equal((await fetch(`${origin}/api/stop`)).status, 405);
  assert.equal((await fetch(`${origin}/api/job`)).status, 400);
  assert.equal((await fetch(`${origin}/api/missing`)).status, 404);
});

test("preserves multipart uploads and streamed API responses and bounds request bodies", async () => {
  const app = createApp({
    staticDir: root,
    maxBodyBytes: 1024,
    handlers: {
      "/api/upload": async (request) => {
        const form = await request.formData();
        assert.equal(form.get("photo").name, "photo.png");
        assert.equal(await form.get("photo").text(), "fixture");
        assert.equal(form.get("payload"), "test");
        return new Response(new Blob(["result"]).stream(), { status: 206, headers: { "content-range": "bytes 0-5/6" } });
      },
    },
  });
  await new Promise((resolve) => app.listen(0, "127.0.0.1", resolve));
  const base = `http://127.0.0.1:${app.address().port}`;
  try {
    const form = new FormData();
    form.set("photo", new File(["fixture"], "photo.png", { type: "image/png" }));
    form.set("payload", "test");
    const response = await fetch(`${base}/api/upload`, { method: "POST", body: form });
    assert.equal(response.status, 206);
    assert.equal(response.headers.get("content-range"), "bytes 0-5/6");
    assert.equal(await response.text(), "result");
    const tooLarge = await fetch(`${base}/api/upload`, { method: "POST", body: "a".repeat(1025) });
    assert.equal(tooLarge.status, 413);
  } finally {
    await new Promise((resolve) => app.close(resolve));
  }
});
