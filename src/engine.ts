import { SHADERS } from './shaders.js';
import { GGUFParser, tensorByteSize, updateConfigFromGGUF } from './gguf.js';
import { Tokenizer } from './tokenizer.js';
import { buildChatPrompt } from './conversation.js';
import {
  defaultConfig,
  getRopeTheta,
  type GemmaConfig,
  type GemmaEngine,
  type GemmaEngineOptions,
  type GenerateOptions,
  type ProgressInfo,
  type ConversationTurn,
  type GGUFTensor,
  type ModelBuffers,
  type WorkBuffers,
  type KVCache,
  type UniformBuffers,
  type BindGroupCache,
  type LayerBindGroups,
} from './types.js';


const MODELS: Record<string, string> = {
  '1b': 'https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q8_0.gguf',
  '270m': 'https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q8_0.gguf',
  'func': 'https://huggingface.co/unsloth/functiongemma-270m-it-GGUF/resolve/main/functiongemma-270m-it-Q8_0.gguf',
};

export class GemmaEngineImpl implements GemmaEngine {
  config: GemmaConfig;
  private device!: GPUDevice;
  private pipelines!: Record<string, GPUComputePipeline>;
  private modelBuffers!: ModelBuffers;
  private workBuffers!: WorkBuffers;
  private uniformBuffers!: UniformBuffers;
  private kvCaches!: KVCache[];
  private bindGroupCache!: BindGroupCache;
  private tokenizer!: Tokenizer;

  private conversationHistory: ConversationTurn[] = [];
  private kvPosition: number = 0;
  private onProgress?: (progress: ProgressInfo) => void;
  private deviceLost: boolean = false;

  constructor(options: GemmaEngineOptions) {
    this.config = defaultConfig();
    if (options.contextLength) {
      this.config.context_length = options.contextLength;
    }
    this.onProgress = options.onProgress;
  }

  async init(options: GemmaEngineOptions): Promise<void> {
    await this.initWebGPU();

    const modelKey = options.model || '1b';
    const modelFile = MODELS[modelKey] || modelKey;

    this.reportProgress(0, 1, 'Downloading header...');
    const HEADER_FETCH_SIZE = 20 * 1024 * 1024;
    const headerResp = await fetch(modelFile, { headers: { Range: `bytes=0-${HEADER_FETCH_SIZE - 1}` } });
    const supportsRange = headerResp.status === 206;

    if (supportsRange) {
      const headerBuf = new Uint8Array(await headerResp.arrayBuffer());
      const parser = new GGUFParser(headerBuf);
      const gguf = parser.parse();
      updateConfigFromGGUF(gguf, this.config, options.contextLength);

      this.tokenizer = new Tokenizer();
      this.tokenizer.extractFromGGUF(gguf);

      const tensors = gguf.tensors;
      const dataOffset = gguf.dataOffset;

      this.createPipelines();
      this.createUniformBuffers();

      await this.uploadWeightsStreaming(modelFile, tensors, dataOffset);
    } else {
      const contentLength = headerResp.headers.get('content-length');
      const total = contentLength ? parseInt(contentLength) : 1100000000;
      const reader = headerResp.body!.getReader();
      const buffer = new Uint8Array(total);
      let received = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer.set(value, received);
        received += value.length;
        this.reportProgress(received, total, `Downloading model...`);
      }

      const parser = new GGUFParser(buffer);
      const gguf = parser.parse();
      updateConfigFromGGUF(gguf, this.config, options.contextLength);

      this.tokenizer = new Tokenizer();
      this.tokenizer.extractFromGGUF(gguf);

      this.createPipelines();
      this.createUniformBuffers();
      this.uploadWeightsFromBuffer(parser, gguf);
      await this.device.queue.onSubmittedWorkDone();
    }

    this.createWorkBuffers();
    this.createBindGroups();
  }

  /** writeBuffer wrapper to handle @webgpu/types ArrayBuffer vs ArrayBufferLike strictness */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private wb(buf: GPUBuffer, offset: number, data: any): void {
    this.device.queue.writeBuffer(buf, offset, data);
  }

  private reportProgress(loaded: number, total: number, status: string): void {
    if (this.onProgress) {
      this.onProgress({ loaded, total, status });
    }
  }

  private async initWebGPU(): Promise<void> {
    if (!navigator.gpu) throw new Error('WebGPU not supported');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter found');
    this.device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
    this.device.lost.then((info) => {
      this.deviceLost = true;
      console.error(`WebGPU device lost: ${info.message} (reason: ${info.reason})`);
    });
  }

  private createPipelines(): void {
    this.pipelines = {};
    for (const [name, code] of Object.entries(SHADERS)) {
      const module = this.device.createShaderModule({ code });
      this.pipelines[name] = this.device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
      });
    }
  }

  private makeUniformMixed(values: ({ u: number } | { f: number })[]): GPUBuffer {
    const size = Math.max(values.length * 4, 16);
    const ab = new ArrayBuffer(size);
    const u32 = new Uint32Array(ab);
    const f32 = new Float32Array(ab);
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if ('u' in v) u32[i] = v.u;
      else f32[i] = v.f;
    }
    const buf = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint8Array(buf.getMappedRange()).set(new Uint8Array(ab));
    buf.unmap();
    return buf;
  }

  private createUniformBuffers(): void {
    const C = this.config;
    const H = C.hidden_size;
    const Q = C.q_dim;
    const KV = C.kv_dim;
    const I = C.intermediate_size;
    const MAX_SEQ = C.context_length;
    const V = C.vocab_size;
    const HD = C.head_dim;
    const NQH = C.num_q_heads;
    const NKH = C.num_kv_heads;

    this.uniformBuffers = {
      rmsNorm: this.makeUniformMixed([{ u: H }, { f: C.rms_norm_eps }]),
      perHeadRmsNormQ: this.makeUniformMixed([{ u: NQH }, { u: HD }, { f: C.rms_norm_eps }, { u: 0 }]),
      perHeadRmsNormK: this.makeUniformMixed([{ u: NKH }, { u: HD }, { f: C.rms_norm_eps }, { u: 0 }]),
      linearQ8_Q_H: this.makeUniformMixed([{ u: Q }, { u: H }]),
      linearQ8_KV_H: this.makeUniformMixed([{ u: KV }, { u: H }]),
      linearQ8_H_Q: this.makeUniformMixed([{ u: H }, { u: Q }]),
      linearQ8_I_H: this.makeUniformMixed([{ u: I }, { u: H }]),
      linearQ8_H_I: this.makeUniformMixed([{ u: H }, { u: I }]),
      sizeH: this.makeUniformMixed([{ u: H }]),
      sizeI: this.makeUniformMixed([{ u: I }]),
      embeddingLookup: this.makeUniformMixed([{ u: H }, { u: 0 }]),
      ropeQ: [],
      ropeK: [],
      kvCacheStore: this.makeUniformMixed([{ u: NKH }, { u: HD }, { u: 0 }, { u: MAX_SEQ }]),
      attnScore: this.makeUniformMixed([
        { u: NQH }, { u: NKH }, { u: HD },
        { u: 0 }, { f: 1.0 / Math.sqrt(HD) }, { u: 0 },
      ]),
      softmax: this.makeUniformMixed([{ u: NQH }, { u: 0 }]),
      attnOutput: this.makeUniformMixed([{ u: NQH }, { u: NKH }, { u: HD }, { u: 0 }]),
      linearQ8_V_H: this.makeUniformMixed([{ u: V }, { u: H }]),
      argmaxSize: this.makeUniformMixed([{ u: V }]),
      fusedNormRopeQ: [],
      fusedNormRopeK: [],
    };

    for (let il = 0; il < C.num_layers; il++) {
      const theta = getRopeTheta(il, C);
      this.uniformBuffers.ropeQ.push(this.makeUniformMixed([{ u: NQH }, { u: HD }, { u: 0 }, { f: theta }]));
      this.uniformBuffers.ropeK.push(this.makeUniformMixed([{ u: NKH }, { u: HD }, { u: 0 }, { f: theta }]));
      this.uniformBuffers.fusedNormRopeQ.push(this.makeUniformMixed([{ u: NQH }, { u: HD }, { f: C.rms_norm_eps }, { u: 0 }, { f: theta }, { u: 0 }]));
      this.uniformBuffers.fusedNormRopeK.push(this.makeUniformMixed([{ u: NKH }, { u: HD }, { f: C.rms_norm_eps }, { u: 0 }, { f: theta }, { u: 0 }]));
    }
  }

  private uploadWeightsFromBuffer(parser: GGUFParser, gguf: { tensors: GGUFTensor[]; dataOffset: number }): void {
    const tensors = gguf.tensors;
    const dataOffset = gguf.dataOffset;

    this.modelBuffers = {
      embeddingQ8: null,
      layers: [],
      finalNorm: null,
    };

    const embedTensor = tensors.find(t => t.name === 'token_embd.weight');
    if (embedTensor) {
      if (embedTensor.type === 8) {
        const packedEmbedding = parser.getTensorDataQ8Packed(embedTensor, dataOffset);
        this.modelBuffers.embeddingQ8 = this.device.createBuffer({
          size: packedEmbedding.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.wb(this.modelBuffers.embeddingQ8, 0, packedEmbedding);
      } else {
        const rawData = parser.getTensorData(embedTensor, dataOffset);
        this.modelBuffers.embeddingQ8 = this.device.createBuffer({
          size: rawData.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.wb(this.modelBuffers.embeddingQ8, 0, rawData);
      }
    }

    const finalNormTensor = tensors.find(t => t.name === 'output_norm.weight');
    if (finalNormTensor) {
      const data = parser.getTensorData(finalNormTensor, dataOffset);
      this.modelBuffers.finalNorm = this.device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.wb(this.modelBuffers.finalNorm, 0, data);
    }

    const q8WeightNames = ['attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate', 'ffn_up', 'ffn_down'];
    const f32WeightNames = ['attn_norm', 'ffn_norm', 'attn_q_norm', 'attn_k_norm', 'post_attention_norm', 'post_ffw_norm'];

    for (let i = 0; i < this.config.num_layers; i++) {
      const prefix = `blk.${i}.`;
      const layer: Record<string, GPUBuffer> = {};
      for (const key of f32WeightNames) {
        const tensor = tensors.find(t => t.name === prefix + key + '.weight');
        if (tensor) {
          const data = parser.getTensorData(tensor, dataOffset);
          layer[key] = this.device.createBuffer({
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          });
          this.wb(layer[key], 0, data);
        }
      }
      for (const key of q8WeightNames) {
        const tensor = tensors.find(t => t.name === prefix + key + '.weight');
        if (tensor) {
          const data = parser.getTensorDataQ8Packed(tensor, dataOffset);
          layer[key] = this.device.createBuffer({
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          });
          this.wb(layer[key], 0, data);
        }
      }
      this.modelBuffers.layers.push(layer);
    }
  }

  private async uploadWeightsStreaming(modelFile: string, tensors: GGUFTensor[], dataOffset: number): Promise<void> {
    this.modelBuffers = {
      embeddingQ8: null,
      layers: [],
      finalNorm: null,
    };

    const fetchRange = async (start: number, size: number): Promise<Uint8Array> => {
      const resp = await fetch(modelFile, { headers: { Range: `bytes=${start}-${start + size - 1}` } });
      return new Uint8Array(await resp.arrayBuffer());
    };

    const uploadTensor = (bytes: Uint8Array, localOffset: number, tensor: GGUFTensor): { buf: GPUBuffer; packed: boolean; byteLength: number; numElements: number } => {
      const tempParser = new GGUFParser(bytes);
      const numElements = Number(tensor.dims.reduce((a, b) => a * b, 1n));
      if (tensor.type === 8) {
        const data = tempParser.packQ8_0ForGPU(localOffset, numElements);
        const buf = this.device.createBuffer({
          size: data.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.wb(buf, 0, data);
        return { buf, packed: true, byteLength: data.byteLength, numElements };
      } else {
        const count = numElements;
        let data: Float32Array;
        if (tensor.type === 0) {
          data = new Float32Array(count);
          const src = new Uint8Array(bytes.buffer, bytes.byteOffset + localOffset, count * 4);
          new Uint8Array(data.buffer).set(src);
        } else {
          data = tempParser.dequantizeF16(localOffset, count);
        }
        const buf = this.device.createBuffer({
          size: data.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.wb(buf, 0, data);
        return { buf, packed: false, byteLength: data.byteLength, numElements };
      }
    };

    let totalUploaded = 0;
    const totalWeightBytes = tensors.reduce((sum, t) => sum + tensorByteSize(t), 0);

    const q8WeightNames = ['attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate', 'ffn_up', 'ffn_down'];
    const f32WeightNames = ['attn_norm', 'ffn_norm', 'attn_q_norm', 'attn_k_norm', 'post_attention_norm', 'post_ffw_norm'];

    // Upload embedding
    const embedTensor = tensors.find(t => t.name === 'token_embd.weight');
    if (embedTensor) {
      const fileOffset = dataOffset + Number(embedTensor.offset);
      const size = tensorByteSize(embedTensor);
      const bytes = await fetchRange(fileOffset, size);
      const result = uploadTensor(bytes, 0, embedTensor);
      this.modelBuffers.embeddingQ8 = result.buf;
      totalUploaded += size;
      this.reportProgress(totalUploaded, totalWeightBytes, 'Streaming weights to GPU...');
    }

    // Upload final norm
    const finalNormTensor = tensors.find(t => t.name === 'output_norm.weight');
    if (finalNormTensor) {
      const fileOffset = dataOffset + Number(finalNormTensor.offset);
      const size = tensorByteSize(finalNormTensor);
      const bytes = await fetchRange(fileOffset, size);
      const result = uploadTensor(bytes, 0, finalNormTensor);
      this.modelBuffers.finalNorm = result.buf;
      totalUploaded += size;
    }

    // Upload layers one at a time via Range requests
    for (let i = 0; i < this.config.num_layers; i++) {
      const prefix = `blk.${i}.`;
      const layerTensors = tensors.filter(t => t.name.startsWith(prefix));
      if (layerTensors.length === 0) continue;

      let minOffset = Infinity, maxEnd = 0;
      for (const t of layerTensors) {
        const off = Number(t.offset);
        const end = off + tensorByteSize(t);
        if (off < minOffset) minOffset = off;
        if (end > maxEnd) maxEnd = end;
      }

      const layerStart = dataOffset + minOffset;
      const layerSize = maxEnd - minOffset;
      const layerBytes = await fetchRange(layerStart, layerSize);

      const layer: Record<string, GPUBuffer> = {};
      for (const key of f32WeightNames) {
        const tensor = layerTensors.find(t => t.name === prefix + key + '.weight');
        if (tensor) {
          const localOffset = Number(tensor.offset) - minOffset;
          const result = uploadTensor(layerBytes, localOffset, tensor);
          layer[key] = result.buf;
        }
      }
      for (const key of q8WeightNames) {
        const tensor = layerTensors.find(t => t.name === prefix + key + '.weight');
        if (tensor) {
          const localOffset = Number(tensor.offset) - minOffset;
          const result = uploadTensor(layerBytes, localOffset, tensor);
          layer[key] = result.buf;
        }
      }
      this.modelBuffers.layers.push(layer);
      totalUploaded += layerSize;
      this.reportProgress(totalUploaded, totalWeightBytes, `Layer ${i + 1}/${this.config.num_layers}`);
    }
  }

  private createWorkBuffers(): void {
    const C = this.config;
    const H = C.hidden_size;
    const Q = C.q_dim;
    const KV = C.kv_dim;
    const I = C.intermediate_size;
    const MAX_SEQ = C.context_length;
    const V = C.vocab_size;
    const HD = C.head_dim;
    const NKH = C.num_kv_heads;

    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
    const SD = S | GPUBufferUsage.COPY_DST;
    this.workBuffers = {
      hidden: this.device.createBuffer({ size: H * 4, usage: SD }),
      hiddenReadback: this.device.createBuffer({ size: H * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
      residual: this.device.createBuffer({ size: H * 4, usage: S }),
      normed: this.device.createBuffer({ size: H * 4, usage: S }),
      q: this.device.createBuffer({ size: Q * 4, usage: S }),
      k: this.device.createBuffer({ size: KV * 4, usage: S }),
      v: this.device.createBuffer({ size: KV * 4, usage: S }),
      attnOut: this.device.createBuffer({ size: Q * 4, usage: S }),
      attnProj: this.device.createBuffer({ size: H * 4, usage: S }),
      postAttnNormed: this.device.createBuffer({ size: H * 4, usage: S }),
      attnScores: this.device.createBuffer({ size: C.num_q_heads * MAX_SEQ * 4, usage: S }),
      ffnGate: this.device.createBuffer({ size: I * 4, usage: S }),
      ffnUp: this.device.createBuffer({ size: I * 4, usage: S }),
      ffnMul: this.device.createBuffer({ size: I * 4, usage: S }),
      ffnDown: this.device.createBuffer({ size: H * 4, usage: S }),
      postFfnNormed: this.device.createBuffer({ size: H * 4, usage: S }),
      logits: this.device.createBuffer({ size: V * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
      logitsReadback: this.device.createBuffer({ size: V * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
      argmaxResult: this.device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
      argmaxReadback: this.device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
      topk256Result: this.device.createBuffer({ size: 256 * 2 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
      topk256Readback: this.device.createBuffer({ size: 256 * 2 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
    };

    const kvSize = MAX_SEQ * NKH * HD * 4;
    this.kvCaches = [];
    for (let i = 0; i < C.num_layers; i++) {
      this.kvCaches.push({
        k: this.device.createBuffer({ size: kvSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC }),
        v: this.device.createBuffer({ size: kvSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC }),
      });
    }
  }

  private createBindGroups(): void {
    const bgc: BindGroupCache = {
      embeddingLookup: this.device.createBindGroup({
        layout: this.pipelines.embeddingLookup.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.modelBuffers.embeddingQ8! } },
          { binding: 1, resource: { buffer: this.workBuffers.hidden } },
          { binding: 2, resource: { buffer: this.uniformBuffers.embeddingLookup } },
        ],
      }),
      finalNorm: this.device.createBindGroup({
        layout: this.pipelines.rmsNorm.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.workBuffers.hidden } },
          { binding: 1, resource: { buffer: this.modelBuffers.finalNorm! } },
          { binding: 2, resource: { buffer: this.workBuffers.normed } },
          { binding: 3, resource: { buffer: this.uniformBuffers.rmsNorm } },
        ],
      }),
      lmHead: this.device.createBindGroup({
        layout: this.pipelines.linearQ8.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.workBuffers.normed } },
          { binding: 1, resource: { buffer: this.modelBuffers.embeddingQ8! } },
          { binding: 2, resource: { buffer: this.workBuffers.logits } },
          { binding: 3, resource: { buffer: this.uniformBuffers.linearQ8_V_H } },
        ],
      }),
      argmax: this.device.createBindGroup({
        layout: this.pipelines.argmax.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.workBuffers.logits } },
          { binding: 1, resource: { buffer: this.workBuffers.argmaxResult } },
          { binding: 2, resource: { buffer: this.uniformBuffers.argmaxSize } },
        ],
      }),
      topk256: this.device.createBindGroup({
        layout: this.pipelines.topk256.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.workBuffers.logits } },
          { binding: 1, resource: { buffer: this.workBuffers.topk256Result } },
          { binding: 2, resource: { buffer: this.uniformBuffers.argmaxSize } },
        ],
      }),
      layers: [],
    };

    for (let i = 0; i < this.config.num_layers; i++) {
      const layer = this.modelBuffers.layers[i];
      const kv = this.kvCaches[i];
      const lb: LayerBindGroups = {
        attnNorm: this.device.createBindGroup({
          layout: this.pipelines.rmsNorm.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.hidden } },
            { binding: 1, resource: { buffer: layer.attn_norm } },
            { binding: 2, resource: { buffer: this.workBuffers.normed } },
            { binding: 3, resource: { buffer: this.uniformBuffers.rmsNorm } },
          ],
        }),
        linearQ: this.device.createBindGroup({
          layout: this.pipelines.linearQ8.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.normed } },
            { binding: 1, resource: { buffer: layer.attn_q } },
            { binding: 2, resource: { buffer: this.workBuffers.q } },
            { binding: 3, resource: { buffer: this.uniformBuffers.linearQ8_Q_H } },
          ],
        }),
        linearK: this.device.createBindGroup({
          layout: this.pipelines.linearQ8.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.normed } },
            { binding: 1, resource: { buffer: layer.attn_k } },
            { binding: 2, resource: { buffer: this.workBuffers.k } },
            { binding: 3, resource: { buffer: this.uniformBuffers.linearQ8_KV_H } },
          ],
        }),
        linearV: this.device.createBindGroup({
          layout: this.pipelines.linearQ8.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.normed } },
            { binding: 1, resource: { buffer: layer.attn_v } },
            { binding: 2, resource: { buffer: this.workBuffers.v } },
            { binding: 3, resource: { buffer: this.uniformBuffers.linearQ8_KV_H } },
          ],
        }),
        ropeQ: this.device.createBindGroup({
          layout: this.pipelines.rope.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.q } },
            { binding: 1, resource: { buffer: this.uniformBuffers.ropeQ[i] } },
          ],
        }),
        ropeK: this.device.createBindGroup({
          layout: this.pipelines.rope.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.k } },
            { binding: 1, resource: { buffer: this.uniformBuffers.ropeK[i] } },
          ],
        }),
        qNorm: this.device.createBindGroup({
          layout: this.pipelines.perHeadRmsNorm.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.q } },
            { binding: 1, resource: { buffer: layer.attn_q_norm } },
            { binding: 2, resource: { buffer: this.uniformBuffers.perHeadRmsNormQ } },
          ],
        }),
        kNorm: this.device.createBindGroup({
          layout: this.pipelines.perHeadRmsNorm.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.k } },
            { binding: 1, resource: { buffer: layer.attn_k_norm } },
            { binding: 2, resource: { buffer: this.uniformBuffers.perHeadRmsNormK } },
          ],
        }),
        fusedNormRopeQ: this.device.createBindGroup({
          layout: this.pipelines.fusedPerHeadNormRope.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.q } },
            { binding: 1, resource: { buffer: layer.attn_q_norm } },
            { binding: 2, resource: { buffer: this.uniformBuffers.fusedNormRopeQ[i] } },
          ],
        }),
        fusedNormRopeK: this.device.createBindGroup({
          layout: this.pipelines.fusedPerHeadNormRope.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.k } },
            { binding: 1, resource: { buffer: layer.attn_k_norm } },
            { binding: 2, resource: { buffer: this.uniformBuffers.fusedNormRopeK[i] } },
          ],
        }),
        kvStore: this.device.createBindGroup({
          layout: this.pipelines.kvCacheStore.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.k } },
            { binding: 1, resource: { buffer: this.workBuffers.v } },
            { binding: 2, resource: { buffer: kv.k } },
            { binding: 3, resource: { buffer: kv.v } },
            { binding: 4, resource: { buffer: this.uniformBuffers.kvCacheStore } },
          ],
        }),
        attnScore: this.device.createBindGroup({
          layout: this.pipelines.attnScore.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.q } },
            { binding: 1, resource: { buffer: kv.k } },
            { binding: 2, resource: { buffer: this.workBuffers.attnScores } },
            { binding: 3, resource: { buffer: this.uniformBuffers.attnScore } },
          ],
        }),
        softmax: this.device.createBindGroup({
          layout: this.pipelines.softmax.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.attnScores } },
            { binding: 1, resource: { buffer: this.uniformBuffers.softmax } },
          ],
        }),
        attnOutput: this.device.createBindGroup({
          layout: this.pipelines.attnOutput.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.attnScores } },
            { binding: 1, resource: { buffer: kv.v } },
            { binding: 2, resource: { buffer: this.workBuffers.attnOut } },
            { binding: 3, resource: { buffer: this.uniformBuffers.attnOutput } },
          ],
        }),
        linearAttnOut: this.device.createBindGroup({
          layout: this.pipelines.linearQ8.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.attnOut } },
            { binding: 1, resource: { buffer: layer.attn_output } },
            { binding: 2, resource: { buffer: this.workBuffers.attnProj } },
            { binding: 3, resource: { buffer: this.uniformBuffers.linearQ8_H_Q } },
          ],
        }),
        postAttnNorm: this.device.createBindGroup({
          layout: this.pipelines.rmsNorm.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.attnProj } },
            { binding: 1, resource: { buffer: layer.post_attention_norm } },
            { binding: 2, resource: { buffer: this.workBuffers.postAttnNormed } },
            { binding: 3, resource: { buffer: this.uniformBuffers.rmsNorm } },
          ],
        }),
        residualAdd1: this.device.createBindGroup({
          layout: this.pipelines.add.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.hidden } },
            { binding: 1, resource: { buffer: this.workBuffers.postAttnNormed } },
            { binding: 2, resource: { buffer: this.workBuffers.residual } },
            { binding: 3, resource: { buffer: this.uniformBuffers.sizeH } },
          ],
        }),
        ffnNorm: this.device.createBindGroup({
          layout: this.pipelines.rmsNorm.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.residual } },
            { binding: 1, resource: { buffer: layer.ffn_norm } },
            { binding: 2, resource: { buffer: this.workBuffers.normed } },
            { binding: 3, resource: { buffer: this.uniformBuffers.rmsNorm } },
          ],
        }),
        ffnGate: this.device.createBindGroup({
          layout: this.pipelines.linearQ8.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.normed } },
            { binding: 1, resource: { buffer: layer.ffn_gate } },
            { binding: 2, resource: { buffer: this.workBuffers.ffnGate } },
            { binding: 3, resource: { buffer: this.uniformBuffers.linearQ8_I_H } },
          ],
        }),
        ffnUp: this.device.createBindGroup({
          layout: this.pipelines.linearQ8.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.normed } },
            { binding: 1, resource: { buffer: layer.ffn_up } },
            { binding: 2, resource: { buffer: this.workBuffers.ffnUp } },
            { binding: 3, resource: { buffer: this.uniformBuffers.linearQ8_I_H } },
          ],
        }),
        geluMul: this.device.createBindGroup({
          layout: this.pipelines.geluMul.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.ffnGate } },
            { binding: 1, resource: { buffer: this.workBuffers.ffnUp } },
            { binding: 2, resource: { buffer: this.workBuffers.ffnMul } },
            { binding: 3, resource: { buffer: this.uniformBuffers.sizeI } },
          ],
        }),
        ffnDown: this.device.createBindGroup({
          layout: this.pipelines.linearQ8.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.ffnMul } },
            { binding: 1, resource: { buffer: layer.ffn_down } },
            { binding: 2, resource: { buffer: this.workBuffers.ffnDown } },
            { binding: 3, resource: { buffer: this.uniformBuffers.linearQ8_H_I } },
          ],
        }),
        postFfnNorm: this.device.createBindGroup({
          layout: this.pipelines.rmsNorm.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.ffnDown } },
            { binding: 1, resource: { buffer: layer.post_ffw_norm } },
            { binding: 2, resource: { buffer: this.workBuffers.postFfnNormed } },
            { binding: 3, resource: { buffer: this.uniformBuffers.rmsNorm } },
          ],
        }),
        residualAdd2: this.device.createBindGroup({
          layout: this.pipelines.add.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.residual } },
            { binding: 1, resource: { buffer: this.workBuffers.postFfnNormed } },
            { binding: 2, resource: { buffer: this.workBuffers.hidden } },
            { binding: 3, resource: { buffer: this.uniformBuffers.sizeH } },
          ],
        }),
        fusedPostAttnNormAdd: this.device.createBindGroup({
          layout: this.pipelines.fusedNormAdd.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.attnProj } },
            { binding: 1, resource: { buffer: layer.post_attention_norm } },
            { binding: 2, resource: { buffer: this.workBuffers.hidden } },
            { binding: 3, resource: { buffer: this.workBuffers.residual } },
            { binding: 4, resource: { buffer: this.uniformBuffers.rmsNorm } },
          ],
        }),
        fusedPostFfnNormAdd: this.device.createBindGroup({
          layout: this.pipelines.fusedNormAdd.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.ffnDown } },
            { binding: 1, resource: { buffer: layer.post_ffw_norm } },
            { binding: 2, resource: { buffer: this.workBuffers.residual } },
            { binding: 3, resource: { buffer: this.workBuffers.hidden } },
            { binding: 4, resource: { buffer: this.uniformBuffers.rmsNorm } },
          ],
        }),
      };
      bgc.layers.push(lb);
    }
    this.bindGroupCache = bgc;
  }

  /** Encode transformer layers into a command encoder (shared between forward pass variants). */
  private encodeTransformerPass(encoder: GPUCommandEncoder, tokenId: number, position: number): void {
    const C = this.config;
    const H = C.hidden_size;
    const Q = C.q_dim;
    const KV = C.kv_dim;
    const I = C.intermediate_size;
    const HD = C.head_dim;
    const NQH = C.num_q_heads;
    const NKH = C.num_kv_heads;
    const seqLen = position + 1;

    this.wb(this.uniformBuffers.embeddingLookup, 4, new Uint32Array([tokenId]));
    const posU32 = new Uint32Array([position]);
    for (let il = 0; il < C.num_layers; il++) {
      this.wb(this.uniformBuffers.fusedNormRopeQ[il], 12, posU32);
      this.wb(this.uniformBuffers.fusedNormRopeK[il], 12, posU32);
    }
    this.wb(this.uniformBuffers.kvCacheStore, 8, posU32);
    const seqU32 = new Uint32Array([seqLen]);
    this.wb(this.uniformBuffers.attnScore, 12, seqU32);
    this.wb(this.uniformBuffers.softmax, 4, seqU32);
    this.wb(this.uniformBuffers.attnOutput, 12, seqU32);

    let pass: GPUComputePassEncoder;

    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.embeddingLookup);
    pass.setBindGroup(0, this.bindGroupCache.embeddingLookup);
    pass.dispatchWorkgroups(Math.ceil(H / 256));
    pass.end();

    for (let layerIdx = 0; layerIdx < C.num_layers; layerIdx++) {
      const lb = this.bindGroupCache.layers[layerIdx];
      pass = encoder.beginComputePass(); pass.setPipeline(this.pipelines.rmsNorm); pass.setBindGroup(0, lb.attnNorm); pass.dispatchWorkgroups(1); pass.end();
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.linearQ8);
      pass.setBindGroup(0, lb.linearQ); pass.dispatchWorkgroups(Math.ceil(Q / 256));
      pass.setBindGroup(0, lb.linearK); pass.dispatchWorkgroups(Math.ceil(KV / 256));
      pass.setBindGroup(0, lb.linearV); pass.dispatchWorkgroups(Math.ceil(KV / 256));
      pass.end();
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.fusedPerHeadNormRope);
      pass.setBindGroup(0, lb.fusedNormRopeQ); pass.dispatchWorkgroups(NQH);
      pass.setBindGroup(0, lb.fusedNormRopeK); pass.dispatchWorkgroups(NKH);
      pass.end();
      pass = encoder.beginComputePass(); pass.setPipeline(this.pipelines.kvCacheStore); pass.setBindGroup(0, lb.kvStore); pass.dispatchWorkgroups(Math.ceil(KV / 256)); pass.end();
      pass = encoder.beginComputePass(); pass.setPipeline(this.pipelines.attnScore); pass.setBindGroup(0, lb.attnScore); pass.dispatchWorkgroups(Math.ceil((NQH * seqLen) / 256)); pass.end();
      pass = encoder.beginComputePass(); pass.setPipeline(this.pipelines.softmax); pass.setBindGroup(0, lb.softmax); pass.dispatchWorkgroups(NQH); pass.end();
      pass = encoder.beginComputePass(); pass.setPipeline(this.pipelines.attnOutput); pass.setBindGroup(0, lb.attnOutput); pass.dispatchWorkgroups(Math.ceil((NQH * HD) / 256)); pass.end();
      pass = encoder.beginComputePass(); pass.setPipeline(this.pipelines.linearQ8); pass.setBindGroup(0, lb.linearAttnOut); pass.dispatchWorkgroups(Math.ceil(H / 256)); pass.end();
      pass = encoder.beginComputePass(); pass.setPipeline(this.pipelines.fusedNormAdd); pass.setBindGroup(0, lb.fusedPostAttnNormAdd); pass.dispatchWorkgroups(1); pass.end();
      pass = encoder.beginComputePass(); pass.setPipeline(this.pipelines.rmsNorm); pass.setBindGroup(0, lb.ffnNorm); pass.dispatchWorkgroups(1); pass.end();
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.linearQ8);
      pass.setBindGroup(0, lb.ffnGate); pass.dispatchWorkgroups(Math.ceil(I / 256));
      pass.setBindGroup(0, lb.ffnUp); pass.dispatchWorkgroups(Math.ceil(I / 256));
      pass.end();
      pass = encoder.beginComputePass(); pass.setPipeline(this.pipelines.geluMul); pass.setBindGroup(0, lb.geluMul); pass.dispatchWorkgroups(Math.ceil(I / 256)); pass.end();
      pass = encoder.beginComputePass(); pass.setPipeline(this.pipelines.linearQ8); pass.setBindGroup(0, lb.ffnDown); pass.dispatchWorkgroups(Math.ceil(H / 256)); pass.end();
      pass = encoder.beginComputePass(); pass.setPipeline(this.pipelines.fusedNormAdd); pass.setBindGroup(0, lb.fusedPostFfnNormAdd); pass.dispatchWorkgroups(1); pass.end();
    }

    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.rmsNorm);
    pass.setBindGroup(0, this.bindGroupCache.finalNorm);
    pass.dispatchWorkgroups(1);
    pass.end();
  }

  /** Encode LM head + sampling into encoder, submit, and read back the selected token. */
  private async sampleNextToken(
    encoder: GPUCommandEncoder,
    temperature: number,
    topP: number,
    repPenalty: number,
    allTokens: number[],
  ): Promise<number> {
    if (this.deviceLost) throw new Error('WebGPU device lost');

    const V = this.config.vocab_size;
    let pass: GPUComputePassEncoder;

    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.linearQ8);
    pass.setBindGroup(0, this.bindGroupCache.lmHead);
    pass.dispatchWorkgroups(Math.ceil(V / 256));
    pass.end();

    const useGreedyFast = (temperature === 0 && repPenalty <= 1.0);
    if (useGreedyFast) {
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.argmax);
      pass.setBindGroup(0, this.bindGroupCache.argmax);
      pass.dispatchWorkgroups(1);
      pass.end();
      encoder.copyBufferToBuffer(this.workBuffers.argmaxResult, 0, this.workBuffers.argmaxReadback, 0, 4);
    } else {
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.topk256);
      pass.setBindGroup(0, this.bindGroupCache.topk256);
      pass.dispatchWorkgroups(1);
      pass.end();
      encoder.copyBufferToBuffer(this.workBuffers.topk256Result, 0, this.workBuffers.topk256Readback, 0, 256 * 2 * 4);
    }

    this.device.queue.submit([encoder.finish()]);

    if (useGreedyFast) {
      try {
        await this.workBuffers.argmaxReadback.mapAsync(GPUMapMode.READ);
      } catch (e) {
        throw new Error(`GPU readback failed (device lost?): ${e}`);
      }
      const resultArray = new Uint32Array(this.workBuffers.argmaxReadback.getMappedRange());
      const tokenResult = resultArray[0];
      this.workBuffers.argmaxReadback.unmap();
      return tokenResult;
    }

    try {
      await this.workBuffers.topk256Readback.mapAsync(GPUMapMode.READ);
    } catch (e) {
      throw new Error(`GPU readback failed (device lost?): ${e}`);
    }
    const topkData = new Float32Array(this.workBuffers.topk256Readback.getMappedRange().slice(0));
    this.workBuffers.topk256Readback.unmap();

    const candidates = new Array(256);
    const topkU32 = new Uint32Array(topkData.buffer.slice(0));
    for (let i = 0; i < 256; i++) {
      candidates[i] = { val: topkData[i * 2], id: topkU32[i * 2 + 1] };
    }

    if (repPenalty > 1.0 && allTokens.length > 0) {
      const seen = new Set(allTokens);
      for (let i = 0; i < 256; i++) {
        if (seen.has(candidates[i].id)) {
          if (candidates[i].val > 0) candidates[i].val /= repPenalty;
          else candidates[i].val *= repPenalty;
        }
      }
    }

    candidates.sort((a: { val: number }, b: { val: number }) => b.val - a.val);
    if (temperature === 0) return candidates[0].id;

    const maxLogit = candidates[0].val;
    let sumExp = 0;
    const probs = new Float32Array(256);
    for (let i = 0; i < 256; i++) {
      probs[i] = Math.exp((candidates[i].val - maxLogit) / temperature);
      sumExp += probs[i];
    }

    let cumProb = 0, cutoff = 256;
    for (let i = 0; i < 256; i++) {
      cumProb += probs[i] / sumExp;
      if (cumProb >= topP) { cutoff = i + 1; break; }
    }
    let subsetSum = 0;
    for (let i = 0; i < cutoff; i++) subsetSum += probs[i];
    let r = Math.random() * subsetSum;
    for (let i = 0; i < cutoff; i++) {
      r -= probs[i];
      if (r <= 0) return candidates[i].id;
    }
    return candidates[cutoff - 1].id;
  }

  private async forwardPassAndGetToken(
    tokenId: number,
    position: number,
    temperature: number = 0,
    topP: number = 0.9,
    repPenalty: number = 1.0,
    allTokens: number[] = [],
  ): Promise<number> {
    const encoder = this.device.createCommandEncoder();
    this.encodeTransformerPass(encoder, tokenId, position);
    return this.sampleNextToken(encoder, temperature, topP, repPenalty, allTokens);
  }

  private forwardPassOnly(tokenId: number, position: number): void {
    const encoder = this.device.createCommandEncoder();
    this.encodeTransformerPass(encoder, tokenId, position);
    this.device.queue.submit([encoder.finish()]);
  }

  private async prefillBatched(tokens: number[], startPos: number = 0): Promise<void> {
    for (let i = 0; i < tokens.length; i++) {
      this.forwardPassOnly(tokens[i], startPos + i);
    }
    await this.device.queue.onSubmittedWorkDone();
  }

  private resetKVCaches(): void {
    const HD = this.config.head_dim;
    const NKH = this.config.num_kv_heads;
    const MAX_SEQ = this.config.context_length;
    const zeros = new Float32Array(MAX_SEQ * NKH * HD);
    for (let i = 0; i < this.config.num_layers; i++) {
      this.wb(this.kvCaches[i].k, 0, zeros);
      this.wb(this.kvCaches[i].v, 0, zeros);
    }
  }

  private async getFirstTokenAfterPrefill(
    temperature: number,
    topP: number,
    repPenalty: number,
    allTokens: number[],
  ): Promise<number> {
    const encoder = this.device.createCommandEncoder();
    return this.sampleNextToken(encoder, temperature, topP, repPenalty, allTokens);
  }

  // ─── Public API ────────────────────────────────────────────────

  addUserMessage(text: string): void {
    this.conversationHistory.push({ role: 'user', text });
  }

  async *generate(options: GenerateOptions = {}): AsyncGenerator<string, void, undefined> {
    if (this.deviceLost) throw new Error('WebGPU device lost — call dispose() and recreate the engine');

    const temperature = options.temperature ?? 0.7;
    const topP = options.topP ?? 0.9;
    const repPenalty = options.repPenalty ?? 1.2;
    const maxTokens = options.maxTokens ?? 32768;
    const toolsJson = options.toolsJson ?? '[]';
    const signal = options.signal;

    let newTokens: number[];
    if (this.kvPosition === 0) {
      const fullPrompt = buildChatPrompt(this.conversationHistory, toolsJson);
      newTokens = this.tokenizer.encode(fullPrompt);
    } else {
      const lastUser = this.conversationHistory[this.conversationHistory.length - 1];
      const suffix = `<end_of_turn>\n<start_of_turn>user\n${lastUser.text}<end_of_turn>\n<start_of_turn>model\n`;
      newTokens = this.tokenizer.encode(suffix).slice(1); // Remove BOS
    }

    // Check context overflow — reset if needed
    if (this.kvPosition + newTokens.length >= this.config.context_length - 10) {
      const lastUser = this.conversationHistory[this.conversationHistory.length - 1];
      this.conversationHistory = [{ role: 'user', text: lastUser.text }];
      const freshPrompt = buildChatPrompt(this.conversationHistory, toolsJson);
      newTokens = this.tokenizer.encode(freshPrompt);
      this.resetKVCaches();
      this.kvPosition = 0;
    }

    // Prefill
    await this.prefillBatched(newTokens, this.kvPosition);
    const allTokens = [...newTokens];
    this.kvPosition += newTokens.length;

    // Get first token (LM head on prefill output)
    let nextToken = await this.getFirstTokenAfterPrefill(temperature, topP, repPenalty, allTokens);
    allTokens.push(nextToken);

    const endFuncCall = this.tokenizer.funcTokens['<end_function_call>'];
    const generatedTokens: number[] = [nextToken];
    let genKVWrites = 0;

    yield this.tokenizer.decodeToken(nextToken);

    // Auto-regressive loop
    for (let step = 1; step < maxTokens; step++) {
      if (nextToken === 1 || nextToken === 0 || nextToken === 106) break;
      if (endFuncCall && nextToken === endFuncCall) break;
      if (signal?.aborted) break;

      const pos = this.kvPosition + genKVWrites;
      if (pos >= this.config.context_length - 1) break;

      nextToken = await this.forwardPassAndGetToken(nextToken, pos, temperature, topP, repPenalty, allTokens);
      genKVWrites++;

      if (nextToken === 1 || nextToken === 0 || nextToken === 106) break;
      if (endFuncCall && nextToken === endFuncCall) {
        allTokens.push(nextToken);
        generatedTokens.push(nextToken);
        break;
      }

      allTokens.push(nextToken);
      generatedTokens.push(nextToken);

      yield this.tokenizer.decodeToken(nextToken);
    }

    // Save model response and update KV position
    const rawOutput = this.tokenizer.decodeTokens(generatedTokens);
    this.conversationHistory.push({ role: 'model', text: rawOutput });
    this.kvPosition += genKVWrites;
  }

  resetConversation(): void {
    this.conversationHistory = [];
    this.kvPosition = 0;
    this.resetKVCaches();
  }

  dispose(): void {
    // Destroy all GPU buffers
    const destroyBuffer = (buf: GPUBuffer | null) => { if (buf) buf.destroy(); };

    destroyBuffer(this.modelBuffers?.embeddingQ8);
    destroyBuffer(this.modelBuffers?.finalNorm);
    if (this.modelBuffers?.layers) {
      for (const layer of this.modelBuffers.layers) {
        for (const buf of Object.values(layer)) {
          destroyBuffer(buf);
        }
      }
    }

    if (this.workBuffers) {
      for (const buf of Object.values(this.workBuffers)) {
        destroyBuffer(buf as GPUBuffer);
      }
    }

    if (this.kvCaches) {
      for (const kv of this.kvCaches) {
        destroyBuffer(kv.k);
        destroyBuffer(kv.v);
      }
    }

    if (this.uniformBuffers) {
      for (const val of Object.values(this.uniformBuffers)) {
        if (Array.isArray(val)) {
          for (const buf of val) destroyBuffer(buf);
        } else {
          destroyBuffer(val as GPUBuffer);
        }
      }
    }

    this.device?.destroy();
  }
}
