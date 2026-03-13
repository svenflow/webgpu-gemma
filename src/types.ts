/** Model configuration — defaults for Gemma 3 270M, updated from GGUF metadata. */
export interface GemmaConfig {
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

export function defaultConfig(): GemmaConfig {
  return {
    hidden_size: 640,
    q_dim: 1024,
    kv_dim: 256,
    num_q_heads: 4,
    num_kv_heads: 1,
    head_dim: 256,
    intermediate_size: 2048,
    vocab_size: 262144,
    num_layers: 18,
    context_length: 2048,
    rms_norm_eps: 1e-6,
    rope_theta_global: 1000000.0,
    rope_theta_swa: 1000000.0,
    swa_period: 6,
  };
}

export function isSwaLayer(il: number, config: GemmaConfig): boolean {
  return (il % config.swa_period) < (config.swa_period - 1);
}

export function getRopeTheta(il: number, config: GemmaConfig): number {
  return isSwaLayer(il, config) ? config.rope_theta_swa : config.rope_theta_global;
}

/** Options for creating a GemmaEngine. */
export interface GemmaEngineOptions {
  /** Model to load: '270m', '1b', or a full URL to a .gguf file. */
  model?: string;
  /** Progress callback during weight loading. */
  onProgress?: (progress: ProgressInfo) => void;
  /** Maximum context length (tokens). Defaults to 2048. */
  contextLength?: number;
}

export interface ProgressInfo {
  /** Bytes downloaded so far. */
  loaded: number;
  /** Total bytes to download. */
  total: number;
  /** Human-readable status message. */
  status: string;
}

export interface GenerateOptions {
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
export interface ConversationTurn {
  role: 'user' | 'model';
  text: string;
}

export interface GemmaEngine {
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

/** Parsed GGUF file structure. */
export interface GGUFParsed {
  version: number;
  tensor_count: bigint;
  kv_count: bigint;
  kv: Map<string, GGUFValue>;
  tensors: GGUFTensor[];
  dataOffset: number;
}

export interface GGUFValue {
  type: string;
  value: any;
}

export interface GGUFTensor {
  name: string;
  dims: bigint[];
  type: number;
  offset: bigint;
}

export interface ModelBuffers {
  embeddingQ8: GPUBuffer | null;
  layers: Record<string, GPUBuffer>[];
  finalNorm: GPUBuffer | null;
}

export interface WorkBuffers {
  hidden: GPUBuffer;
  hiddenReadback: GPUBuffer;
  residual: GPUBuffer;
  normed: GPUBuffer;
  q: GPUBuffer;
  k: GPUBuffer;
  v: GPUBuffer;
  attnOut: GPUBuffer;
  attnProj: GPUBuffer;
  postAttnNormed: GPUBuffer;
  attnScores: GPUBuffer;
  ffnGate: GPUBuffer;
  ffnUp: GPUBuffer;
  ffnMul: GPUBuffer;
  ffnDown: GPUBuffer;
  postFfnNormed: GPUBuffer;
  logits: GPUBuffer;
  logitsReadback: GPUBuffer;
  argmaxResult: GPUBuffer;
  argmaxReadback: GPUBuffer;
  topk256Result: GPUBuffer;
  topk256Readback: GPUBuffer;
}

export interface KVCache {
  k: GPUBuffer;
  v: GPUBuffer;
}

export interface UniformBuffers {
  rmsNorm: GPUBuffer;
  perHeadRmsNormQ: GPUBuffer;
  perHeadRmsNormK: GPUBuffer;
  linearQ8_Q_H: GPUBuffer;
  linearQ8_KV_H: GPUBuffer;
  linearQ8_H_Q: GPUBuffer;
  linearQ8_I_H: GPUBuffer;
  linearQ8_H_I: GPUBuffer;
  sizeH: GPUBuffer;
  sizeI: GPUBuffer;
  embeddingLookup: GPUBuffer;
  ropeQ: GPUBuffer[];
  ropeK: GPUBuffer[];
  kvCacheStore: GPUBuffer;
  attnScore: GPUBuffer;
  softmax: GPUBuffer;
  attnOutput: GPUBuffer;
  linearQ8_V_H: GPUBuffer;
  argmaxSize: GPUBuffer;
  fusedNormRopeQ: GPUBuffer[];
  fusedNormRopeK: GPUBuffer[];
}

export interface BindGroupCache {
  embeddingLookup: GPUBindGroup;
  finalNorm: GPUBindGroup;
  lmHead: GPUBindGroup;
  argmax: GPUBindGroup;
  topk256: GPUBindGroup;
  layers: LayerBindGroups[];
}

export interface LayerBindGroups {
  attnNorm: GPUBindGroup;
  linearQ: GPUBindGroup;
  linearK: GPUBindGroup;
  linearV: GPUBindGroup;
  ropeQ: GPUBindGroup;
  ropeK: GPUBindGroup;
  qNorm: GPUBindGroup;
  kNorm: GPUBindGroup;
  fusedNormRopeQ: GPUBindGroup;
  fusedNormRopeK: GPUBindGroup;
  kvStore: GPUBindGroup;
  attnScore: GPUBindGroup;
  softmax: GPUBindGroup;
  attnOutput: GPUBindGroup;
  linearAttnOut: GPUBindGroup;
  postAttnNorm: GPUBindGroup;
  residualAdd1: GPUBindGroup;
  ffnNorm: GPUBindGroup;
  ffnGate: GPUBindGroup;
  ffnUp: GPUBindGroup;
  geluMul: GPUBindGroup;
  ffnDown: GPUBindGroup;
  postFfnNorm: GPUBindGroup;
  residualAdd2: GPUBindGroup;
  fusedPostAttnNormAdd: GPUBindGroup;
  fusedPostFfnNormAdd: GPUBindGroup;
}
