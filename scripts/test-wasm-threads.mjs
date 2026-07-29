#!/usr/bin/env node

import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, extname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = resolve(scriptDirectory, "..");
const smokePage = "/wasm/tests/threaded-smoke.html";

const contentTypes = new Map([
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".mjs", "text/javascript; charset=utf-8"],
  [".of", "text/plain; charset=utf-8"],
  [".wasm", "application/wasm"],
]);

const server = createServer(async (request, response) => {
  response.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  response.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  response.setHeader("Cross-Origin-Resource-Policy", "same-origin");
  response.setHeader("Cache-Control", "no-store");

  try {
    const url = new URL(request.url ?? "/", "http://127.0.0.1");
    const relativePath = (url.pathname === "/" ? smokePage : url.pathname).replace(
      /^\/+/,
      "",
    );
    const filePath = resolve(repositoryRoot, relativePath);
    if (filePath !== repositoryRoot && !filePath.startsWith(`${repositoryRoot}${sep}`)) {
      response.writeHead(403).end("forbidden");
      return;
    }

    const body = await readFile(filePath);
    response.setHeader("Content-Type", contentTypes.get(extname(filePath)) ?? "application/octet-stream");
    response.writeHead(200).end(body);
  } catch (error) {
    response.writeHead(404).end(error instanceof Error ? error.message : String(error));
  }
});

function delay(milliseconds) {
  return new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));
}

async function waitForPageTarget(devToolsUrl, targetUrl) {
  const endpoint = new URL(devToolsUrl);
  const listUrl = `http://${endpoint.hostname}:${endpoint.port}/json/list`;
  const deadline = Date.now() + 10_000;
  while (Date.now() < deadline) {
    const targets = await fetch(listUrl).then((response) => response.json());
    const page = targets.find((target) => target.type === "page" && target.url === targetUrl);
    if (page?.webSocketDebuggerUrl) {
      return page.webSocketDebuggerUrl;
    }
    await delay(100);
  }
  throw new Error("headless Chrome did not expose the threaded WASM page");
}

async function evaluateWhenFinished(webSocketUrl) {
  const socket = new WebSocket(webSocketUrl);
  await new Promise((resolveOpen, rejectOpen) => {
    socket.addEventListener("open", resolveOpen, { once: true });
    socket.addEventListener("error", rejectOpen, { once: true });
  });

  const expression = `new Promise((resolve) => {
    const deadline = Date.now() + 30000;
    const check = () => {
      const status = document.body?.dataset.status;
      if (status === "passed" || status === "failed") {
        resolve({ status, message: document.querySelector("#result")?.textContent ?? "" });
      } else if (Date.now() >= deadline) {
        resolve({ status: "timed-out", message: document.documentElement.outerHTML });
      } else {
        setTimeout(check, 100);
      }
    };
    check();
  })`;

  try {
    return await new Promise((resolveResult, rejectResult) => {
      const deadline = Date.now() + 35_000;
      let requestId = 0;
      let retry;
      const timeout = setTimeout(() => {
        rejectResult(new Error("Chrome DevTools evaluation timed out"));
      }, 35_000);

      const finish = (callback, value) => {
        clearTimeout(timeout);
        clearTimeout(retry);
        callback(value);
      };

      const evaluate = () => {
        requestId += 1;
        socket.send(
          JSON.stringify({
            id: requestId,
            method: "Runtime.evaluate",
            params: { expression, awaitPromise: true, returnByValue: true },
          }),
        );
      };

      socket.addEventListener("error", () => {
        finish(rejectResult, new Error("Chrome DevTools WebSocket failed"));
      });
      socket.addEventListener("close", () => {
        finish(rejectResult, new Error("Chrome DevTools WebSocket closed before evaluation"));
      });
      socket.addEventListener("message", (event) => {
        const message = JSON.parse(String(event.data));
        if (message.id !== requestId) {
          return;
        }
        if (message.error) {
          const contextIsStarting =
            message.error.code === -32000 &&
            /execution context/i.test(message.error.message ?? "");
          if (contextIsStarting && Date.now() < deadline) {
            retry = setTimeout(evaluate, 100);
            return;
          }
          finish(
            rejectResult,
            new Error(
              `Chrome DevTools evaluation failed (${message.error.code}): ${message.error.message}`,
            ),
          );
        } else if (message.result?.exceptionDetails) {
          finish(rejectResult, new Error(message.result.exceptionDetails.text));
        } else {
          finish(resolveResult, message.result?.result?.value);
        }
      });
      evaluate();
    });
  } finally {
    socket.close();
  }
}

await new Promise((resolveListening, rejectListening) => {
  server.once("error", rejectListening);
  server.listen(0, "127.0.0.1", resolveListening);
});

let browser;
let chromeProfile;
try {
  const address = server.address();
  if (address === null || typeof address === "string") {
    throw new Error("threaded WASM smoke server did not expose a TCP port");
  }

  const chrome = process.env.CHROME_BIN || "google-chrome";
  const targetUrl = `http://127.0.0.1:${address.port}${smokePage}`;
  chromeProfile = await mkdtemp(join(tmpdir(), "openferric-wasm-chrome-"));
  browser = spawn(
    chrome,
    [
      "--headless=new",
      "--no-sandbox",
      "--disable-dev-shm-usage",
      "--disable-gpu",
      "--remote-debugging-port=0",
      `--user-data-dir=${chromeProfile}`,
      targetUrl,
    ],
    { stdio: ["ignore", "pipe", "pipe"] },
  );

  let stderr = "";
  const devToolsUrl = await new Promise((resolveUrl, rejectUrl) => {
    const timeout = setTimeout(() => {
      rejectUrl(new Error(`headless Chrome did not start DevTools\n${stderr}`));
    }, 30_000);
    browser.once("error", (error) => {
      clearTimeout(timeout);
      rejectUrl(error);
    });
    browser.once("exit", (code, signal) => {
      clearTimeout(timeout);
      rejectUrl(
        new Error(
          `headless Chrome exited before starting DevTools (code ${code}, signal ${signal})\n${stderr}`,
        ),
      );
    });
    browser.stderr.setEncoding("utf8").on("data", (chunk) => {
      stderr += chunk;
      const match = stderr.match(/DevTools listening on (ws:\/\/\S+)/);
      if (match) {
        clearTimeout(timeout);
        resolveUrl(match[1]);
      }
    });
  });

  const pageWebSocket = await waitForPageTarget(devToolsUrl, targetUrl);
  const result = await evaluateWhenFinished(pageWebSocket);
  if (result?.status !== "passed") {
    throw new Error(`threaded WASM smoke test did not pass: ${JSON.stringify(result)}\n${stderr}`);
  }

  process.stdout.write("threaded WASM browser smoke test passed\n");
} finally {
  if (browser && browser.exitCode === null) {
    browser.kill("SIGTERM");
    await Promise.race([
      new Promise((resolveExit) => browser.once("close", resolveExit)),
      delay(5_000),
    ]);
  }
  if (chromeProfile) {
    await rm(chromeProfile, { recursive: true, force: true });
  }
  await new Promise((resolveClosed) => server.close(resolveClosed));
}
