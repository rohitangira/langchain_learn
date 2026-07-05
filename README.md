(Minimal instructions)

Using Ollama locally
- Install Ollama (see https://ollama.com/download) and ensure the `ollama` CLI is available.
- Pull a model you want to use, for example:

```bash
ollama pull gpt-oss:7b
```

- Ensure the Ollama daemon is running (depending on your install this may be automatic). You can also run a model interactively for quick tests:

```bash
ollama run gpt-oss:7b --prompt "Hello"
```

- Set the model used by the script via environment variable `OLLAMA_MODEL` (defaults to `gpt-oss:7b`). Example:

```bash
export OLLAMA_MODEL=gpt-oss:7b
```

- Run the script with your project's virtualenv Python:

```bash
.venv/bin/python main.py
```

If you see a connection error, make sure the Ollama service is running and the requested model is pulled.

