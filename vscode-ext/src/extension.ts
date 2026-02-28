import * as vscode from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
} from "vscode-languageclient/node";
import { PricingPanelProvider, PricingResult, MarketSnapshot } from "./pricingPanel";

const LANGUAGE_ID = "openferric";
const MARKET_STATE_KEY = "openferric.marketState";
let client: LanguageClient | undefined;

export function activate(context: vscode.ExtensionContext): void {
  const config = vscode.workspace.getConfiguration("openferric");
  const lspPath = config.get<string>(
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
        enabled: config.get<boolean>("pricing.enabled", true),
        numPaths: config.get<number>("pricing.numPaths", 50000),
        numSteps: config.get<number>("pricing.numSteps", 100),
        seed: config.get<number>("pricing.seed", 42),
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
    context.workspaceState.update(MARKET_STATE_KEY, market);
    if (client) {
      client.sendNotification("openferric/updateMarket", market);
    }
  });

  // Start client and listen for pricing notifications.
  client.start().then(() => {
    client!.onNotification(
      "openferric/pricingResult",
      (result: PricingResult) => {
        pricingProvider.update(result);
      }
    );
  });

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
