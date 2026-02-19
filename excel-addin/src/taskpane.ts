interface WasmInitModule {
  default: () => Promise<unknown>;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isWasmInitModule(value: unknown): value is WasmInitModule {
  return isRecord(value) && typeof value["default"] === "function";
}

function getStatusElement(): HTMLElement {
  const element = document.getElementById("status");
  if (!element) {
    throw new Error("Missing #status element in taskpane.html");
  }

  return element;
}

function setStatus(statusElement: HTMLElement, className: string, message: string): void {
  statusElement.className = className;
  statusElement.textContent = message;
}

async function loadWasmIntoTaskpane(statusElement: HTMLElement): Promise<void> {
  const moduleUrl = new URL("./pkg/openferric.js", import.meta.url).href;
  const moduleCandidate: unknown = await import(moduleUrl);
  if (!isWasmInitModule(moduleCandidate)) {
    throw new TypeError("Invalid OpenFerric WASM module shape loaded in taskpane.");
  }
  const module = moduleCandidate;
  await module.default();
  setStatus(statusElement, "status ok", "WASM engine loaded");
}

async function bootstrapTaskpane(): Promise<void> {
  const statusElement = getStatusElement();

  try {
    await loadWasmIntoTaskpane(statusElement);
  } catch (error) {
    setStatus(
      statusElement,
      "status error",
      "WASM not available - build with: wasm-pack build --target web --out-dir excel-addin/pkg"
    );
    console.error(error);
  }
}

void bootstrapTaskpane();
