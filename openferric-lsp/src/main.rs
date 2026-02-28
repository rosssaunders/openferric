mod backend;
mod codelens;
mod completion;
mod diagnostics;
mod document_symbols;
mod goto_def;
mod hover;
mod notification;
mod semantic_tokens;
mod symbols;

use tower_lsp::{LspService, Server};

#[tokio::main]
async fn main() {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(backend::Backend::new);
    Server::new(stdin, stdout, socket).serve(service).await;
}
