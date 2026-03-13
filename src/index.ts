/**
 * webgpu-gemma
 *
 * Run Gemma 3 locally in the browser via WebGPU. Q8_0 quantized, streaming
 * generation, multi-turn chat with KV cache reuse.
 *
 * @example
 * ```typescript
 * import { createGemmaEngine } from 'webgpu-gemma'
 *
 * const engine = await createGemmaEngine({
 *   model: '1b',
 *   onProgress: (p) => console.log(p.status),
 * });
 *
 * engine.addUserMessage('What is the capital of France?');
 * for await (const token of engine.generate({ temperature: 0.7 })) {
 *   process.stdout.write(token);
 * }
 *
 * // Follow-up reuses KV cache
 * engine.addUserMessage('And what about Germany?');
 * for await (const token of engine.generate()) {
 *   process.stdout.write(token);
 * }
 *
 * engine.dispose();
 * ```
 */

import { GemmaEngineImpl } from './engine.js';
import type { GemmaEngine, GemmaEngineOptions } from './types.js';

export async function createGemmaEngine(options: GemmaEngineOptions = {}): Promise<GemmaEngine> {
  const engine = new GemmaEngineImpl(options);
  await engine.init(options);
  return engine;
}

export type {
  GemmaEngine,
  GemmaEngineOptions,
  GemmaConfig,
  GenerateOptions,
  ProgressInfo,
  ConversationTurn,
} from './types.js';
