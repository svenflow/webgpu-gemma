# webgpu-gemma

Run Gemma 3 1B locally in the browser via WebGPU. Q8_0 quantized, streaming generation, multi-turn chat with KV cache reuse. Zero dependencies.

**[Live Demo](https://svenflow.github.io/webgpu-gemma/)** · **61KB** min · **12KB** gzip

## Features

- **Gemma 3 1B and 270M** — runs entirely in-browser, no server needed
- **Q8_0 quantization** — high quality inference at ~1GB model size
- **Streaming generation** — async iterator API, tokens streamed as generated
- **Multi-turn chat** — KV cache reuse for fast follow-up messages
- **Range request loading** — streams weights layer-by-layer, works on iPhone
- **12KB gzipped** — zero dependencies, pure WebGPU compute shaders

## Install

```bash
npm install webgpu-gemma
```

## Usage

```typescript
import { createGemmaEngine } from 'webgpu-gemma'

const engine = await createGemmaEngine({
  model: '1b', // '1b', '270m', or a full URL to a .gguf file
  onProgress: (p) => console.log(p.status),
});

// Multi-turn conversation
engine.addUserMessage('What is the capital of France?');
for await (const token of engine.generate({ temperature: 0.7 })) {
  process.stdout.write(token);
}

// Follow-up reuses KV cache — near-instant prefill
engine.addUserMessage('And what about Germany?');
for await (const token of engine.generate()) {
  process.stdout.write(token);
}

// Reset conversation
engine.resetConversation();

// Cleanup
engine.dispose();
```

## API

### `createGemmaEngine(options?)`

Creates and initializes a Gemma engine. Downloads and loads the model weights.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model` | `string` | `'1b'` | Model to load: `'1b'`, `'270m'`, or a URL to a `.gguf` file |
| `onProgress` | `function` | — | Progress callback: `({ loaded, total, status }) => void` |
| `contextLength` | `number` | `2048` | Maximum context length in tokens |

### `engine.addUserMessage(text)`

Add a user message to the conversation history.

### `engine.generate(options?)`

Returns an `AsyncGenerator<string>` that yields decoded tokens.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `temperature` | `number` | `0.7` | Sampling temperature. 0 = greedy |
| `topP` | `number` | `0.9` | Top-P nucleus sampling threshold |
| `repPenalty` | `number` | `1.2` | Repetition penalty. 1.0 = none |
| `maxTokens` | `number` | `32768` | Maximum tokens to generate |
| `toolsJson` | `string` | `'[]'` | JSON array of tool declarations for function calling |

### `engine.resetConversation()`

Clears conversation history and resets KV cache.

### `engine.dispose()`

Releases all GPU resources.

### `engine.config`

Read-only model configuration (hidden size, layers, vocab size, etc).

## Requirements

- A browser with [WebGPU support](https://caniuse.com/webgpu) (Chrome 113+, Edge 113+, Safari 18+)
- For the 1B model: ~1GB download + ~1.5GB GPU memory
- For the 270M model: ~300MB download + ~500MB GPU memory

## How It Works

1. **GGUF parsing** — reads model metadata and tokenizer vocabulary from the GGUF header
2. **Range request streaming** — fetches weights layer-by-layer via HTTP Range requests (~44MB each), uploads to GPU, frees JS memory. Peak JS memory is ~50MB instead of ~1GB
3. **WebGPU compute shaders** — 18 custom WGSL shaders for embedding lookup, RMS norm, RoPE, attention, FFN, and sampling
4. **KV cache reuse** — follow-up messages only prefill new tokens, making multi-turn conversations fast

## License

MIT
