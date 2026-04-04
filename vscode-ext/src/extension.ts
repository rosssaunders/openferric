import * as vscode from "vscode";
import type {
  LanguageClientOptions,
  ServerOptions} from "vscode-languageclient/node";
import {
  LanguageClient
} from "vscode-languageclient/node";

import type { PricingResult, MarketSnapshot } from "./pricingPanel";
import { PricingPanelProvider } from "./pricingPanel";

const LANGUAGE_ID = "openferric";
const MARKET_STATE_KEY = "openferric.marketState";
let client: LanguageClient | undefined;

export function activate(context: vscode.ExtensionContext): void {
  const config = vscode.workspace.getConfiguration("openferric");
  const lspPath = config.get(
    "lsp.path",
    "openferric-lsp"
  );

  // Restore market state from workspace or fall back to settings.
  const savedMarket = context.workspaceState.get<MarketSnapshot>(MARKET_STATE_KEY);
  const initialMarket = savedMarket ?? config.get("market.default");

  const serverOptions: ServerOptions = {
    command: lspPath,
  };

  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ language: LANGUAGE_ID }],
    initializationOptions: {
      pricing: {
        enabled: config.get("pricing.enabled", true),
        numPaths: config.get("pricing.numPaths", 50000),
        numSteps: config.get("pricing.numSteps", 100),
        seed: config.get("pricing.seed", 42),
      },
      market: initialMarket,
    },
  };

  client = new LanguageClient(
    "openferric",
    "OpenFerric",
    serverOptions,
    clientOptions
  );

  // Register pricing dashboard panel.
  const pricingProvider = new PricingPanelProvider();
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      PricingPanelProvider.viewType,
      pricingProvider
    )
  );

  // Forward market edits from webview to LSP and persist.
  pricingProvider.onMarketUpdate((market: MarketSnapshot) => {
    void context.workspaceState.update(MARKET_STATE_KEY, market);
    if (client) {
      void client.sendNotification("openferric/updateMarket", market);
    }
  });

  // Start client and listen for pricing notifications.
  void (async () => {
    await client.start();
    client.onNotification(
      "openferric/pricingResult",
      (result: PricingResult) => {
        pricingProvider.update(result);
      }
    );
  })();

  context.subscriptions.push({
    dispose: () => {
      if (client) {
        void client.stop();
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
