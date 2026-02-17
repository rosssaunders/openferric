#!/usr/bin/env node
/**
 * Development server for the OpenFerric Excel Add-in.
 * Serves files over HTTPS (required by Office Add-ins).
 *
 * Usage:
 *   node serve.js
 *
 * Then sideload manifest.xml into Excel:
 *   - Excel Online: Insert → Office Add-ins → Upload My Add-in → manifest.xml
 *   - Excel Desktop: https://learn.microsoft.com/office/dev/add-ins/testing/sideload-office-add-ins
 */

const https = require('https');
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const PORT = 3000;
const DIR = __dirname;

// Generate self-signed cert if not present
const certDir = path.join(DIR, '.certs');
if (!fs.existsSync(certDir)) fs.mkdirSync(certDir);

const keyPath = path.join(certDir, 'key.pem');
const certPath = path.join(certDir, 'cert.pem');

if (!fs.existsSync(keyPath)) {
  console.log('Generating self-signed certificate...');
  execSync(
    `openssl req -x509 -newkey rsa:2048 -keyout "${keyPath}" -out "${certPath}" ` +
    `-days 365 -nodes -subj "/CN=localhost"`,
    { stdio: 'inherit' }
  );
}

const MIME = {
  '.html': 'text/html',
  '.js':   'application/javascript',
  '.mjs':  'application/javascript',
  '.json': 'application/json',
  '.xml':  'application/xml',
  '.wasm': 'application/wasm',
  '.css':  'text/css',
  '.png':  'image/png',
  '.svg':  'image/svg+xml',
};

const server = https.createServer(
  { key: fs.readFileSync(keyPath), cert: fs.readFileSync(certPath) },
  (req, res) => {
    let urlPath = req.url.split('?')[0];
    if (urlPath === '/') urlPath = '/taskpane.html';

    const filePath = path.join(DIR, urlPath);
    if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
      res.writeHead(404);
      res.end('Not found');
      return;
    }

    const ext = path.extname(filePath);
    const contentType = MIME[ext] || 'application/octet-stream';

    res.writeHead(200, {
      'Content-Type': contentType,
      'Access-Control-Allow-Origin': '*',
      'Cache-Control': 'no-cache',
    });
    fs.createReadStream(filePath).pipe(res);
  }
);

server.listen(PORT, () => {
  console.log(`\n  🔥 OpenFerric Excel Add-in server`);
  console.log(`  https://localhost:${PORT}/\n`);
  console.log(`  Sideload manifest.xml in Excel to get started.`);
  console.log(`  Press Ctrl+C to stop.\n`);
});
