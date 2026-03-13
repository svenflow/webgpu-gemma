/** Model configuration — defaults for Gemma 3 270M, updated from GGUF metadata. */
interface GemmaConfig {
    hidden_size: number;
    q_dim: number;
    kv_dim: number;
    num_q_heads: number;
    num_kv_heads: number;
    head_dim: number;
    intermediate_size: number;
    vocab_size: number;
    num_layers: number;
    context_length: number;
    rms_norm_eps: number;
    rope_theta_global: number;
    rope_theta_swa: number;
    swa_period: number;
}
/** Options for creating a GemmaEngine. */
interface GemmaEngineOptions {
    /** Model to load: '270m', '1b', or a full URL to a .gguf file. */
    model?: string;
    /** Progress callback during weight loading. */
    onProgress?: (progress: ProgressInfo) => void;
    /** Maximum context length (tokens). Defaults to 2048. */
    contextLength?: number;
}
interface ProgressInfo {
    /** Bytes downloaded so far. */
    loaded: number;
    /** Total bytes to download. */
    total: number;
    /** Human-readable status message. */
    status: string;
}
interface GenerateOptions {
    /** Sampling temperature. 0 = greedy. Default: 0.7 */
    temperature?: number;
    /** Top-P nucleus sampling threshold. Default: 0.9 */
    topP?: number;
    /** Repetition penalty. 1.0 = no penalty. Default: 1.2 */
    repPenalty?: number;
    /** Maximum tokens to generate. Default: 32768 */
    maxTokens?: number;
    /** Tools JSON string for function calling. Default: '[]' */
    toolsJson?: string;
    /** AbortSignal to cancel generation mid-stream. */
    signal?: AbortSignal;
}
/** A single turn in conversation history. */
interface ConversationTurn {
    role: 'user' | 'model';
    text: string;
}
interface GemmaEngine {
    /** Add a user message to the conversation. */
    addUserMessage(text: string): void;
    /** Generate a response as an async iterator of token strings. */
    generate(options?: GenerateOptions): AsyncGenerator<string, void, undefined>;
    /** Reset conversation history and KV cache. */
    resetConversation(): void;
    /** Release all GPU resources. */
    dispose(): void;
    /** Current config (read-only). */
    readonly config: Readonly<GemmaConfig>;
}

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

declare function createGemmaEngine(options?: GemmaEngineOptions): Promise<GemmaEngine>;

export { type ConversationTurn, type GemmaConfig, type GemmaEngine, type GemmaEngineOptions, type GenerateOptions, type ProgressInfo, createGemmaEngine };
