import * as path from "path";
import { workspace, ExtensionContext } from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
} from "vscode-languageclient/node";

let client: LanguageClient;

export function activate(context: ExtensionContext) {
  // Launch the Python LSP server in stdio mode
  const serverOptions: ServerOptions = {
    command: "python",
    args: ["-m", "ewm_engine.serialization.lsp_server"],
  };

  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: "file", language: "wsl" }],
    synchronize: {
      fileEvents: workspace.createFileSystemWatcher("**/*.wsl"),
    },
  };

  client = new LanguageClient(
    "wslLanguageServer",
    "World Specification Language Server",
    serverOptions,
    clientOptions
  );

  client.start();
}

export function deactivate(): Thenable<void> | undefined {
  if (!client) {
    return undefined;
  }
  return client.stop();
}
