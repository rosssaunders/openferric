import { execSync } from "node:child_process";
import { createReadStream, existsSync, mkdirSync, readFileSync, statSync } from "node:fs";
import { createServer } from "node:https";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
const PORT = 3000;
const DIRECTORY = fileURLToPath(new URL(".", import.meta.url));
const certDirectory = join(DIRECTORY, ".certs");
const keyPath = join(certDirectory, "key.pem");
const certPath = join(certDirectory, "cert.pem");
const MIME = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".wasm": "application/wasm",
    ".xml": "application/xml; charset=utf-8"
};
function ensureCertificate() {
    if (!existsSync(certDirectory)) {
        mkdirSync(certDirectory);
    }
    if (existsSync(keyPath) && existsSync(certPath)) {
        return;
    }
    console.info("Generating self-signed certificate for localhost...");
    execSync(`openssl req -x509 -newkey rsa:2048 -keyout "${keyPath}" -out "${certPath}" -days 365 -nodes -subj "/CN=localhost"`, { stdio: "inherit" });
}
function sanitizeUrlPath(urlPath) {
    const withoutQuery = urlPath.split("?")[0] ?? "/";
    const decoded = decodeURIComponent(withoutQuery);
    const normalized = normalize(decoded).replaceAll("\\\\", "/");
    const path = normalized === "/" ? "/taskpane.html" : normalized;
    if (path.includes("..")) {
        return "/404";
    }
    return path;
}
function resolveFilePath(pathFromRequest) {
    const trimmed = pathFromRequest.startsWith("/") ? pathFromRequest.slice(1) : pathFromRequest;
    return join(DIRECTORY, trimmed);
}
function getContentType(filePath) {
    return MIME[extname(filePath)] ?? "application/octet-stream";
}
function startServer() {
    ensureCertificate();
    const server = createServer({ key: readFileSync(keyPath), cert: readFileSync(certPath) }, (request, response) => {
        const requestPath = sanitizeUrlPath(request.url ?? "/");
        const filePath = resolveFilePath(requestPath);
        if (!existsSync(filePath) || !statSync(filePath).isFile()) {
            response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
            response.end("Not found");
            return;
        }
        response.writeHead(200, {
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "no-cache",
            "Content-Type": getContentType(filePath)
        });
        createReadStream(filePath).pipe(response);
    });
    server.listen(PORT, () => {
        console.info("\nOpenFerric Excel Add-in server");
        console.info(`https://localhost:${String(PORT)}/\n`);
        console.info("Sideload manifest.xml in Excel to get started.");
        console.info("Press Ctrl+C to stop.\n");
    });
}
startServer();
//# sourceMappingURL=serve.js.map