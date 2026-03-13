import type { GGUFParsed, GGUFTensor, GGUFValue, GemmaConfig } from './types.js';

export class GGUFParser {
  buffer: ArrayBuffer;
  view: DataView;
  offset: number;
  textDecoder: TextDecoder;

  constructor(buffer: ArrayBuffer | Uint8Array) {
    if (buffer instanceof Uint8Array) {
      // Handle Uint8Array views with non-zero byteOffset
      this.buffer = buffer.buffer as ArrayBuffer;
      this.view = new DataView(this.buffer, buffer.byteOffset, buffer.byteLength);
      this.offset = 0;
    } else {
      this.buffer = buffer;
      this.view = new DataView(this.buffer);
      this.offset = 0;
    }
    this.textDecoder = new TextDecoder('utf-8');
  }

  readUint32(): number {
    const val = this.view.getUint32(this.offset, true);
    this.offset += 4;
    return val;
  }

  readUint64(): bigint {
    const val = this.view.getBigUint64(this.offset, true);
    this.offset += 8;
    return val;
  }

  readString(): string {
    const length = Number(this.readUint64());
    const bytes = new Uint8Array(this.buffer, this.offset, length);
    this.offset += length;
    return this.textDecoder.decode(bytes);
  }

  readValue(type: number): GGUFValue {
    switch (type) {
      case 0: { const val = this.view.getUint8(this.offset); this.offset += 1; return { type: 'uint8', value: val }; }
      case 1: { const val = this.view.getInt8(this.offset); this.offset += 1; return { type: 'int8', value: val }; }
      case 2: { const val = this.view.getUint16(this.offset, true); this.offset += 2; return { type: 'uint16', value: val }; }
      case 3: { const val = this.view.getInt16(this.offset, true); this.offset += 2; return { type: 'int16', value: val }; }
      case 4: return { type: 'uint32', value: this.readUint32() };
      case 5: { const val = this.view.getInt32(this.offset, true); this.offset += 4; return { type: 'int32', value: val }; }
      case 6: { const val = this.view.getFloat32(this.offset, true); this.offset += 4; return { type: 'float32', value: val }; }
      case 7: { const val = this.view.getUint8(this.offset); this.offset += 1; return { type: 'bool', value: val !== 0 }; }
      case 8: return { type: 'string', value: this.readString() };
      case 9: {
        const elemType = this.readUint32();
        const count = Number(this.readUint64());
        const arr: GGUFValue[] = [];
        for (let i = 0; i < count; i++) arr.push(this.readValue(elemType));
        return { type: 'array', value: arr };
      }
      case 10: return { type: 'uint64', value: this.readUint64() };
      case 11: { const val = this.view.getBigInt64(this.offset, true); this.offset += 8; return { type: 'int64', value: val }; }
      case 12: { const val = this.view.getFloat64(this.offset, true); this.offset += 8; return { type: 'float64', value: val }; }
      default: return { type: 'unknown', value: null };
    }
  }

  parse(): GGUFParsed {
    const magic = this.readUint32();
    if (magic !== 0x46554747) throw new Error(`Invalid GGUF magic: 0x${magic.toString(16)}`);
    const version = this.readUint32();
    const tensor_count = this.readUint64();
    const kv_count = this.readUint64();
    const kv = new Map<string, GGUFValue>();
    for (let i = 0n; i < kv_count; i++) {
      const key = this.readString();
      const valueType = this.readUint32();
      const value = this.readValue(valueType);
      kv.set(key, value);
    }
    const tensors: GGUFTensor[] = [];
    for (let i = 0n; i < tensor_count; i++) {
      const name = this.readString();
      const n_dims = this.readUint32();
      const dims: bigint[] = [];
      for (let d = 0; d < n_dims; d++) dims.push(this.readUint64());
      const type = this.readUint32();
      const offset = this.readUint64();
      tensors.push({ name, dims, type, offset });
    }
    const alignment = 32;
    const dataOffset = Math.ceil(this.offset / alignment) * alignment;
    return { version, tensor_count, kv_count, kv, tensors, dataOffset };
  }

  f16ToF32(h: number): number {
    const sign = (h >> 15) & 0x1;
    const exp = (h >> 10) & 0x1f;
    const mant = h & 0x3ff;
    if (exp === 0) {
      if (mant === 0) return sign ? -0 : 0;
      return (sign ? -1 : 1) * Math.pow(2, -14) * (mant / 1024);
    }
    if (exp === 31) return mant === 0 ? (sign ? -Infinity : Infinity) : NaN;
    return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + mant / 1024);
  }

  dequantizeQ8_0(offset: number, count: number): Float32Array {
    const blockSize = 32;
    const result = new Float32Array(count);
    let resultIdx = 0;
    let blockOffset = offset;
    while (resultIdx < count) {
      const scaleBits = this.view.getUint16(blockOffset, true);
      const scale = this.f16ToF32(scaleBits);
      blockOffset += 2;
      const elemsInBlock = Math.min(blockSize, count - resultIdx);
      for (let i = 0; i < elemsInBlock; i++) {
        const q = this.view.getInt8(blockOffset + i);
        result[resultIdx++] = q * scale;
      }
      blockOffset += blockSize;
    }
    return result;
  }

  packQ8_0ForGPU(offset: number, totalElements: number): Uint32Array {
    const blockSize = 32;
    const numBlocks = totalElements / blockSize;
    const result = new ArrayBuffer(numBlocks * 36);
    const resultU32 = new Uint32Array(result);
    const resultF32 = new Float32Array(result);
    let srcOff = offset;
    for (let b = 0; b < numBlocks; b++) {
      const scaleBits = this.view.getUint16(srcOff, true);
      const scale = this.f16ToF32(scaleBits);
      resultF32[b * 9] = scale;
      srcOff += 2;
      for (let j = 0; j < 8; j++) {
        let packed = 0;
        for (let k = 0; k < 4; k++) {
          const val = this.view.getUint8(srcOff + j * 4 + k);
          packed |= val << (k * 8);
        }
        resultU32[b * 9 + 1 + j] = packed;
      }
      srcOff += 32;
    }
    return new Uint32Array(result);
  }

  dequantizeF16(offset: number, count: number): Float32Array {
    const result = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const bits = this.view.getUint16(offset + i * 2, true);
      result[i] = this.f16ToF32(bits);
    }
    return result;
  }

  getTensorData(tensor: GGUFTensor, dataOffset: number): Float32Array {
    const absOffset = dataOffset + Number(tensor.offset);
    const count = Number(tensor.dims.reduce((a, b) => a * b, 1n));
    if (tensor.type === 0) {
      return new Float32Array(this.buffer, absOffset, count);
    } else if (tensor.type === 1) {
      return this.dequantizeF16(absOffset, count);
    } else if (tensor.type === 8) {
      return this.dequantizeQ8_0(absOffset, count);
    }
    throw new Error(`Unsupported type: ${tensor.type}`);
  }

  getTensorDataQ8Packed(tensor: GGUFTensor, dataOffset: number): Uint32Array {
    const absOffset = dataOffset + Number(tensor.offset);
    const count = Number(tensor.dims.reduce((a, b) => a * b, 1n));
    return this.packQ8_0ForGPU(absOffset, count);
  }
}

/** Compute byte size of a tensor in a GGUF file. */
export function tensorByteSize(tensor: GGUFTensor): number {
  const numElements = Number(tensor.dims.reduce((a, b) => a * b, 1n));
  if (tensor.type === 8) return (numElements / 32) * 34; // Q8_0: 34 bytes per block of 32
  if (tensor.type === 0) return numElements * 4; // F32
  if (tensor.type === 1) return numElements * 2; // F16
  throw new Error(`Unknown tensor type: ${tensor.type}`);
}

/** Update a GemmaConfig from GGUF metadata. */
export function updateConfigFromGGUF(gguf: GGUFParsed, config: GemmaConfig, maxContextLength?: number): string {
  const getKV = (key: string): number | null => {
    const entry = gguf.kv.get(key);
    return entry ? (typeof entry.value === 'object' ? Number(entry.value) : Number(entry.value)) : null;
  };

  const hidden = getKV('gemma3.embedding_length');
  const layers = getKV('gemma3.block_count');
  const intermediate = getKV('gemma3.feed_forward_length');
  const qHeads = getKV('gemma3.attention.head_count');
  const kvHeads = getKV('gemma3.attention.head_count_kv');
  const headDim = getKV('gemma3.attention.key_length');
  const ctxLen = getKV('gemma3.context_length');

  if (hidden !== null) config.hidden_size = hidden;
  if (layers !== null) config.num_layers = layers;
  if (intermediate !== null) config.intermediate_size = intermediate;
  if (qHeads !== null) config.num_q_heads = qHeads;
  if (kvHeads !== null) config.num_kv_heads = kvHeads;
  if (headDim !== null) config.head_dim = headDim;

  const maxCtx = maxContextLength ?? 2048;
  if (ctxLen !== null) config.context_length = Math.min(Number(ctxLen), maxCtx);

  // Recompute derived values
  config.q_dim = config.num_q_heads * config.head_dim;
  config.kv_dim = config.num_kv_heads * config.head_dim;

  const modelSize = config.num_layers <= 18 ? '270M' : '1B';
  return modelSize;
}
