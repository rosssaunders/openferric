import * as vscode from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
} from "vscode-languageclient/node";

const LANGUAGE_ID = "openferric";
let client: LanguageClient | undefined;

export function activate(context: vscode.ExtensionContext): void {
  const config = vscode.workspace.getConfiguration("openferric");
  const lspPath = config.get<string>(
    "lsp.path",
    "openferric-lsp"
  );

  const serverOptions: ServerOptions = {
    command: lspPath,
  };

  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ language: LANGUAGE_ID }],
    initializationOptions: {
      pricing: {
        enabled: config.get<boolean>("pricing.enabled", true),
        numPaths: config.get<number>("pricing.numPaths", 50000),
        numSteps: config.get<number>("pricing.numSteps", 100),
        seed: config.get<number>("pricing.seed", 42),
      },
      market: config.get("market.default"),
    },
  };

  client = new LanguageClient(
    "openferric",
    "OpenFerric",
    serverOptions,
    clientOptions
  );

  client.start();
  context.subscriptions.push({
    dispose: () => {
      if (client) {
        client.stop();
      }
    },
  });
}

export function deactivate(): Promise<void> | undefined {
  if (client) {
    return client.stop();
  }
  return undefined;
}
