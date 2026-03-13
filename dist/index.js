var C={embeddingLookup:`
struct Params { hidden_size: u32, token_id: u32 }
@group(0) @binding(0) var<storage, read> embedding_q8: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.hidden_size) { return; }
  let blocks_per_row = params.hidden_size / 32u;
  let row_offset = params.token_id * blocks_per_row * 9u;
  let block_idx = i / 32u;
  let elem_in_block = i % 32u;
  let block_off = row_offset + block_idx * 9u;
  let scale = bitcast<f32>(embedding_q8[block_off]);
  let packed_idx = elem_in_block / 4u;
  let byte_idx = elem_in_block % 4u;
  let packed = embedding_q8[block_off + 1u + packed_idx];
  let q = f32(extractBits(bitcast<i32>(packed), byte_idx * 8u, 8u));
  output[i] = q * scale * sqrt(f32(params.hidden_size));
}`,rmsNorm:`
struct Params { hidden_size: u32, eps: f32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let hidden_size = params.hidden_size;
  var partial_sum: f32 = 0.0;
  var i = tid;
  while (i < hidden_size) {
    let val = input[i];
    partial_sum += val * val;
    i += 256u;
  }
  shared_sum[tid] = partial_sum;
  workgroupBarrier();
  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }
    workgroupBarrier();
  }
  let rms = sqrt(shared_sum[0] / f32(hidden_size) + params.eps);
  i = tid;
  while (i < hidden_size) {
    output[i] = input[i] * weight[i] / rms;
    i += 256u;
  }
}`,perHeadRmsNorm:`
struct Params { num_heads: u32, head_dim: u32, eps: f32, pad: u32 }
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let tid = lid.x;
  let head = wid.x;
  if (head >= params.num_heads) { return; }
  let base = head * params.head_dim;
  var partial: f32 = 0.0;
  var i = tid;
  while (i < params.head_dim) {
    let val = data[base + i];
    partial += val * val;
    i += 256u;
  }
  shared_sum[tid] = partial;
  workgroupBarrier();
  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }
    workgroupBarrier();
  }
  let rms = sqrt(shared_sum[0] / f32(params.head_dim) + params.eps);
  i = tid;
  while (i < params.head_dim) {
    data[base + i] = data[base + i] * weight[i] / rms;
    i += 256u;
  }
}`,linearQ8:`
struct Params { M: u32, N: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight_q8: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
var<workgroup> shared_tile: array<f32, 2048>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let m = gid.x;
  let tid = lid.x;
  let N = params.N;
  let blocks_per_row = N / 32u;
  let row_offset = m * blocks_per_row * 9u;
  var total_sum: f32 = 0.0;
  let TILE: u32 = 2048u;

  var tile_start: u32 = 0u;
  while (tile_start < N) {
    let tile_size = min(TILE, N - tile_start);
    var i = tid;
    while (i < tile_size) {
      shared_tile[i] = input[tile_start + i];
      i += 256u;
    }
    workgroupBarrier();
    if (m < params.M) {
      let block_start = tile_start / 32u;
      let block_end = (tile_start + tile_size) / 32u;
      for (var b = block_start; b < block_end; b++) {
        let block_off = row_offset + b * 9u;
        let scale = bitcast<f32>(weight_q8[block_off]);
        let local_base = b * 32u - tile_start;
        var block_sum: f32 = 0.0;
        for (var chunk: u32 = 0u; chunk < 4u; chunk++) {
          let c = local_base + chunk * 8u;
          let qs_base = block_off + 1u + chunk * 2u;
          let packed0 = weight_q8[qs_base];
          let packed1 = weight_q8[qs_base + 1u];
          block_sum += f32(extractBits(bitcast<i32>(packed0), 0u, 8u))  * shared_tile[c + 0u];
          block_sum += f32(extractBits(bitcast<i32>(packed0), 8u, 8u))  * shared_tile[c + 1u];
          block_sum += f32(extractBits(bitcast<i32>(packed0), 16u, 8u)) * shared_tile[c + 2u];
          block_sum += f32(extractBits(bitcast<i32>(packed0), 24u, 8u)) * shared_tile[c + 3u];
          block_sum += f32(extractBits(bitcast<i32>(packed1), 0u, 8u))  * shared_tile[c + 4u];
          block_sum += f32(extractBits(bitcast<i32>(packed1), 8u, 8u))  * shared_tile[c + 5u];
          block_sum += f32(extractBits(bitcast<i32>(packed1), 16u, 8u)) * shared_tile[c + 6u];
          block_sum += f32(extractBits(bitcast<i32>(packed1), 24u, 8u)) * shared_tile[c + 7u];
        }
        total_sum += block_sum * scale;
      }
    }
    workgroupBarrier();
    tile_start += TILE;
  }
  if (m < params.M) {
    output[m] = total_sum;
  }
}`,geluMul:`
const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
const GELU_COEF_A: f32 = 0.044715;

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> size: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= size) { return; }
  let x = gate[i];
  let tanh_arg = clamp(SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x), -15.0, 15.0);
  let gelu = 0.5 * x * (1.0 + tanh(tanh_arg));
  output[i] = gelu * up[i];
}`,add:`
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> size: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= size) { return; }
  output[i] = a[i] + b[i];
}`,rope:`
struct Params { num_heads: u32, head_dim: u32, position: u32, theta: f32 }
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let half_dim = params.head_dim / 2u;
  let total_pairs = params.num_heads * half_dim;
  if (idx >= total_pairs) { return; }
  let head = idx / half_dim;
  let i = idx % half_dim;
  let base = head * params.head_dim;
  let freq = 1.0 / pow(params.theta, f32(i * 2u) / f32(params.head_dim));
  let angle = f32(params.position) * freq;
  let cos_a = cos(angle);
  let sin_a = sin(angle);
  let x0 = data[base + i];
  let x1 = data[base + i + half_dim];
  data[base + i] = x0 * cos_a - x1 * sin_a;
  data[base + i + half_dim] = x0 * sin_a + x1 * cos_a;
}`,kvCacheStore:`
struct Params { num_kv_heads: u32, head_dim: u32, position: u32, max_seq_len: u32 }
@group(0) @binding(0) var<storage, read> k_in: array<f32>;
@group(0) @binding(1) var<storage, read> v_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let total = params.num_kv_heads * params.head_dim;
  if (i >= total) { return; }
  let head = i / params.head_dim;
  let d = i % params.head_dim;
  let cache_idx = params.position * total + head * params.head_dim + d;
  k_cache[cache_idx] = k_in[i];
  v_cache[cache_idx] = v_in[i];
}`,attnScore:`
struct Params { num_q_heads: u32, num_kv_heads: u32, head_dim: u32, seq_len: u32, scale: f32, pad0: u32 }
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.num_q_heads * params.seq_len;
  if (idx >= total) { return; }
  let head = idx / params.seq_len;
  let pos = idx % params.seq_len;
  let kv_head = head * params.num_kv_heads / params.num_q_heads;
  let q_offset = head * params.head_dim;
  let kv_stride = params.num_kv_heads * params.head_dim;
  let k_offset = pos * kv_stride + kv_head * params.head_dim;
  var dot: f32 = 0.0;
  for (var d: u32 = 0u; d < params.head_dim; d++) {
    dot += q[q_offset + d] * k_cache[k_offset + d];
  }
  scores[head * params.seq_len + pos] = dot * params.scale;
}`,softmax:`
struct Params { num_heads: u32, seq_len: u32 }
@group(0) @binding(0) var<storage, read_write> scores: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let tid = lid.x;
  let head = wid.x;
  if (head >= params.num_heads) { return; }
  let base = head * params.seq_len;
  var local_max: f32 = -1e30;
  var i = tid;
  while (i < params.seq_len) {
    local_max = max(local_max, scores[base + i]);
    i += 256u;
  }
  shared_max[tid] = local_max;
  workgroupBarrier();
  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]); }
    workgroupBarrier();
  }
  let max_val = shared_max[0];
  var local_sum: f32 = 0.0;
  i = tid;
  while (i < params.seq_len) {
    let e = exp(scores[base + i] - max_val);
    scores[base + i] = e;
    local_sum += e;
    i += 256u;
  }
  shared_sum[tid] = local_sum;
  workgroupBarrier();
  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }
    workgroupBarrier();
  }
  let sum_val = shared_sum[0];
  i = tid;
  while (i < params.seq_len) {
    scores[base + i] = scores[base + i] / sum_val;
    i += 256u;
  }
}`,attnOutput:`
struct Params { num_q_heads: u32, num_kv_heads: u32, head_dim: u32, seq_len: u32 }
@group(0) @binding(0) var<storage, read> probs: array<f32>;
@group(0) @binding(1) var<storage, read> v_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = params.num_q_heads * params.head_dim;
  if (idx >= total) { return; }
  let head = idx / params.head_dim;
  let d = idx % params.head_dim;
  let kv_head = head * params.num_kv_heads / params.num_q_heads;
  let kv_stride = params.num_kv_heads * params.head_dim;
  var sum: f32 = 0.0;
  for (var pos: u32 = 0u; pos < params.seq_len; pos++) {
    let prob = probs[head * params.seq_len + pos];
    let v_idx = pos * kv_stride + kv_head * params.head_dim + d;
    sum += prob * v_cache[v_idx];
  }
  output[head * params.head_dim + d] = sum;
}`,argmax:`
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<u32>;
@group(0) @binding(2) var<uniform> size: u32;
var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_idx: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  var local_max: f32 = -1e30;
  var local_idx: u32 = 0u;
  var i = tid;
  while (i < size) {
    let val = logits[i];
    if (val > local_max) {
      local_max = val;
      local_idx = i;
    }
    i += 256u;
  }
  shared_max[tid] = local_max;
  shared_idx[tid] = local_idx;
  workgroupBarrier();
  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride && shared_max[tid + stride] > shared_max[tid]) {
      shared_max[tid] = shared_max[tid + stride];
      shared_idx[tid] = shared_idx[tid + stride];
    }
    workgroupBarrier();
  }
  if (tid == 0u) {
    result[0] = shared_idx[0];
  }
}`,topk256:`
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;
@group(0) @binding(2) var<uniform> size: u32;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  var local_max: f32 = -1e30;
  var local_idx: u32 = 0u;
  var i = tid;
  while (i < size) {
    let val = logits[i];
    if (val > local_max) {
      local_max = val;
      local_idx = i;
    }
    i += 256u;
  }
  result[tid * 2u] = local_max;
  result[tid * 2u + 1u] = bitcast<f32>(local_idx);
}`,fusedNormAdd:`
struct Params { size: u32, eps: f32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> residual: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let N = params.size;
  var partial: f32 = 0.0;
  var i = tid;
  while (i < N) {
    let val = input[i];
    partial += val * val;
    i += 256u;
  }
  shared_sum[tid] = partial;
  workgroupBarrier();
  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }
    workgroupBarrier();
  }
  let rms = sqrt(shared_sum[0] / f32(N) + params.eps);
  i = tid;
  while (i < N) {
    output[i] = residual[i] + input[i] * weight[i] / rms;
    i += 256u;
  }
}`,fusedPerHeadNormRope:`
struct Params { num_heads: u32, head_dim: u32, eps: f32, position: u32, theta: f32, pad: u32 }
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
  let head = wid.x;
  if (head >= params.num_heads) { return; }
  let tid = lid.x;
  let base = head * params.head_dim;
  var partial: f32 = 0.0;
  var i = tid;
  while (i < params.head_dim) {
    let val = data[base + i];
    partial += val * val;
    i += 256u;
  }
  shared_sum[tid] = partial;
  workgroupBarrier();
  for (var stride: u32 = 128u; stride > 0u; stride >>= 1u) {
    if (tid < stride) { shared_sum[tid] += shared_sum[tid + stride]; }
    workgroupBarrier();
  }
  let rms = sqrt(shared_sum[0] / f32(params.head_dim) + params.eps);
  i = tid;
  while (i < params.head_dim) {
    data[base + i] = data[base + i] * weight[i] / rms;
    i += 256u;
  }
  workgroupBarrier();
  let half_dim = params.head_dim / 2u;
  i = tid;
  while (i < half_dim) {
    let freq = 1.0 / pow(params.theta, f32(i * 2u) / f32(params.head_dim));
    let angle = f32(params.position) * freq;
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    let x0 = data[base + i];
    let x1 = data[base + i + half_dim];
    data[base + i] = x0 * cos_a - x1 * sin_a;
    data[base + i + half_dim] = x0 * sin_a + x1 * cos_a;
    i += 256u;
  }
}`};var w=class{buffer;view;offset;textDecoder;constructor(e){e instanceof Uint8Array?(this.buffer=e.buffer,this.view=new DataView(this.buffer,e.byteOffset,e.byteLength),this.offset=0):(this.buffer=e,this.view=new DataView(this.buffer),this.offset=0),this.textDecoder=new TextDecoder("utf-8")}readUint32(){let e=this.view.getUint32(this.offset,!0);return this.offset+=4,e}readUint64(){let e=this.view.getBigUint64(this.offset,!0);return this.offset+=8,e}readString(){let e=Number(this.readUint64()),r=new Uint8Array(this.buffer,this.offset,e);return this.offset+=e,this.textDecoder.decode(r)}readValue(e){switch(e){case 0:{let r=this.view.getUint8(this.offset);return this.offset+=1,{type:"uint8",value:r}}case 1:{let r=this.view.getInt8(this.offset);return this.offset+=1,{type:"int8",value:r}}case 2:{let r=this.view.getUint16(this.offset,!0);return this.offset+=2,{type:"uint16",value:r}}case 3:{let r=this.view.getInt16(this.offset,!0);return this.offset+=2,{type:"int16",value:r}}case 4:return{type:"uint32",value:this.readUint32()};case 5:{let r=this.view.getInt32(this.offset,!0);return this.offset+=4,{type:"int32",value:r}}case 6:{let r=this.view.getFloat32(this.offset,!0);return this.offset+=4,{type:"float32",value:r}}case 7:{let r=this.view.getUint8(this.offset);return this.offset+=1,{type:"bool",value:r!==0}}case 8:return{type:"string",value:this.readString()};case 9:{let r=this.readUint32(),t=Number(this.readUint64()),s=[];for(let n=0;n<t;n++)s.push(this.readValue(r));return{type:"array",value:s}}case 10:return{type:"uint64",value:this.readUint64()};case 11:{let r=this.view.getBigInt64(this.offset,!0);return this.offset+=8,{type:"int64",value:r}}case 12:{let r=this.view.getFloat64(this.offset,!0);return this.offset+=8,{type:"float64",value:r}}default:return{type:"unknown",value:null}}}parse(){let e=this.readUint32();if(e!==1179993927)throw new Error(`Invalid GGUF magic: 0x${e.toString(16)}`);let r=this.readUint32(),t=this.readUint64(),s=this.readUint64(),n=new Map;for(let o=0n;o<s;o++){let a=this.readString(),p=this.readUint32(),d=this.readValue(p);n.set(a,d)}let c=[];for(let o=0n;o<t;o++){let a=this.readString(),p=this.readUint32(),d=[];for(let i=0;i<p;i++)d.push(this.readUint64());let l=this.readUint32(),m=this.readUint64();c.push({name:a,dims:d,type:l,offset:m})}let u=32,f=Math.ceil(this.offset/u)*u;return{version:r,tensor_count:t,kv_count:s,kv:n,tensors:c,dataOffset:f}}f16ToF32(e){let r=e>>15&1,t=e>>10&31,s=e&1023;return t===0?s===0?r?-0:0:(r?-1:1)*Math.pow(2,-14)*(s/1024):t===31?s===0?r?-1/0:1/0:NaN:(r?-1:1)*Math.pow(2,t-15)*(1+s/1024)}dequantizeQ8_0(e,r){let s=new Float32Array(r),n=0,c=e;for(;n<r;){let u=this.view.getUint16(c,!0),f=this.f16ToF32(u);c+=2;let o=Math.min(32,r-n);for(let a=0;a<o;a++){let p=this.view.getInt8(c+a);s[n++]=p*f}c+=32}return s}packQ8_0ForGPU(e,r){let s=r/32,n=new ArrayBuffer(s*36),c=new Uint32Array(n),u=new Float32Array(n),f=e;for(let o=0;o<s;o++){let a=this.view.getUint16(f,!0),p=this.f16ToF32(a);u[o*9]=p,f+=2;for(let d=0;d<8;d++){let l=0;for(let m=0;m<4;m++){let i=this.view.getUint8(f+d*4+m);l|=i<<m*8}c[o*9+1+d]=l}f+=32}return new Uint32Array(n)}dequantizeF16(e,r){let t=new Float32Array(r);for(let s=0;s<r;s++){let n=this.view.getUint16(e+s*2,!0);t[s]=this.f16ToF32(n)}return t}getTensorData(e,r){let t=r+Number(e.offset),s=Number(e.dims.reduce((n,c)=>n*c,1n));if(e.type===0)return new Float32Array(this.buffer,t,s);if(e.type===1)return this.dequantizeF16(t,s);if(e.type===8)return this.dequantizeQ8_0(t,s);throw new Error(`Unsupported type: ${e.type}`)}getTensorDataQ8Packed(e,r){let t=r+Number(e.offset),s=Number(e.dims.reduce((n,c)=>n*c,1n));return this.packQ8_0ForGPU(t,s)}};function P(_){let e=Number(_.dims.reduce((r,t)=>r*t,1n));if(_.type===8)return e/32*34;if(_.type===0)return e*4;if(_.type===1)return e*2;throw new Error(`Unknown tensor type: ${_.type}`)}function z(_,e,r){let t=l=>{let m=_.kv.get(l);return m?(typeof m.value=="object",Number(m.value)):null},s=t("gemma3.embedding_length"),n=t("gemma3.block_count"),c=t("gemma3.feed_forward_length"),u=t("gemma3.attention.head_count"),f=t("gemma3.attention.head_count_kv"),o=t("gemma3.attention.key_length"),a=t("gemma3.context_length");s!==null&&(e.hidden_size=s),n!==null&&(e.num_layers=n),c!==null&&(e.intermediate_size=c),u!==null&&(e.num_q_heads=u),f!==null&&(e.num_kv_heads=f),o!==null&&(e.head_dim=o);let p=r??2048;return a!==null&&(e.context_length=Math.min(Number(a),p)),e.q_dim=e.num_q_heads*e.head_dim,e.kv_dim=e.num_kv_heads*e.head_dim,e.num_layers<=18?"270M":"1B"}var R={"<start_of_turn>":105,"<end_of_turn>":106,"<eos>":1,"<bos>":2},q=["<start_function_declaration>","<end_function_declaration>","<start_function_call>","<end_function_call>","<start_function_response>","<end_function_response>","<escape>"],U=class{vocab=[];vocabByLength=[];tokenByText=new Map;maxTokenLen=0;specialTokens={...R};funcTokens={};specialPatternRegex=/\\<start_of_turn\\>|\\<end_of_turn\\>|\\<eos\\>|\\<bos\\>/g;extractFromGGUF(e){let r=e.kv.get("tokenizer.ggml.tokens");if(r&&r.type==="array")this.vocab=r.value.map(t=>t.value);else throw new Error("No tokenizer found in GGUF metadata");this.vocabByLength=[];for(let t=0;t<this.vocab.length;t++)this.vocab[t]&&this.vocab[t].length>0&&this.vocabByLength.push([t,this.vocab[t]]);this.buildTokenIndex(),this.initFunctionTokens()}buildTokenIndex(){this.tokenByText=new Map,this.maxTokenLen=0;for(let e=0;e<this.vocab.length;e++){let r=this.vocab[e];r&&r.length>0&&(this.tokenByText.has(r)||this.tokenByText.set(r,e),r.length>this.maxTokenLen&&(this.maxTokenLen=r.length))}}initFunctionTokens(){for(let e of q){let r=this.tokenByText.get(e);r!==void 0&&(this.funcTokens[e]=r,this.specialTokens[e]=r)}this.rebuildSpecialPattern()}rebuildSpecialPattern(){let e=Object.keys(this.specialTokens);e.sort((t,s)=>s.length-t.length);let r=e.map(t=>t.replace(/[<>]/g,s=>"\\"+s)).join("|");this.specialPatternRegex=new RegExp(r,"g")}encodeSegment(e,r=!0){let t=[],s=e.replace(/ /g,"\u2581");for(r&&(s="\u2581"+s);s.length>0;){let n=0,c=-1,u=Math.min(s.length,this.maxTokenLen);for(let f=u;f>=1;f--){let o=s.substring(0,f),a=this.tokenByText.get(o);if(a!==void 0){n=f,c=a;break}}n===0?s=s.slice(1):(t.push(c),s=s.slice(n))}return t}encode(e){let r=[2],t=new RegExp(this.specialPatternRegex.source,"g"),s=0,n,c=!1;for(;(n=t.exec(e))!==null;){let f=e.slice(s,n.index);f.length>0&&r.push(...this.encodeSegment(f,!c)),r.push(this.specialTokens[n[0]]),s=n.index+n[0].length,c=!0}let u=e.slice(s);return u.length>0&&r.push(...this.encodeSegment(u,!c)),r}decodeToken(e){return e<this.vocab.length&&this.vocab[e]?this.vocab[e].replace(/\u2581/g," "):`<unk:${e}>`}decodeTokens(e){let r="";for(let t of e)r+=this.decodeToken(t);return r}};function T(_,e){let r="",t=null;try{if(e){let n=JSON.parse(e);Array.isArray(n)&&n.length>0&&(t=n)}}catch{t=null}if(t){let n="";for(let c of t){if(n+=`<start_function_declaration>declaration:${c.name}{`,n+=`description:<escape>${c.description}<escape>`,c.parameters){n+=",parameters:{properties:{";let u=Object.entries(c.parameters.properties||{});n+=u.map(([f,o])=>{let a=`${f}:{description:<escape>${o.description}<escape>,type:<escape>${o.type}<escape>`;return o.enum&&(a+=`,enum:[${o.enum.map(p=>`<escape>${p}<escape>`).join(",")}]`),a+="}",a}).join(","),n+="}",c.parameters.required&&(n+=`,required:[${c.parameters.required.map(f=>`<escape>${f}<escape>`).join(",")}]`),n+=`,type:<escape>${c.parameters.type}<escape>`,n+="}"}n+="}<end_function_declaration>"}r=`<start_of_turn>developer
You are a model that can do function calling with the following functions
${n}
<end_of_turn>
`}let s=r;for(let n of _)s+=`<start_of_turn>${n.role}
${n.text}<end_of_turn>
`;return s+=`<start_of_turn>model
`,s}function S(){return{hidden_size:640,q_dim:1024,kv_dim:256,num_q_heads:4,num_kv_heads:1,head_dim:256,intermediate_size:2048,vocab_size:262144,num_layers:18,context_length:2048,rms_norm_eps:1e-6,rope_theta_global:1e6,rope_theta_swa:1e6,swa_period:6}}function O(_,e){return _%e.swa_period<e.swa_period-1}function A(_,e){return O(_,e)?e.rope_theta_swa:e.rope_theta_global}var Q={"1b":"https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q8_0.gguf","270m":"https://huggingface.co/unsloth/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q8_0.gguf",func:"https://huggingface.co/unsloth/functiongemma-270m-it-GGUF/resolve/main/functiongemma-270m-it-Q8_0.gguf"},x=class{config;device;pipelines;modelBuffers;workBuffers;uniformBuffers;kvCaches;bindGroupCache;tokenizer;conversationHistory=[];kvPosition=0;onProgress;deviceLost=!1;constructor(e){this.config=S(),e.contextLength&&(this.config.context_length=e.contextLength),this.onProgress=e.onProgress}async init(e){await this.initWebGPU();let r=e.model||"1b",t=Q[r]||r;this.reportProgress(0,1,"Downloading header...");let s=20*1024*1024,n=await fetch(t,{headers:{Range:`bytes=0-${s-1}`}});if(n.status===206){let u=new Uint8Array(await n.arrayBuffer()),o=new w(u).parse();z(o,this.config,e.contextLength),this.tokenizer=new U,this.tokenizer.extractFromGGUF(o);let a=o.tensors,p=o.dataOffset;this.createPipelines(),this.createUniformBuffers(),await this.uploadWeightsStreaming(t,a,p)}else{let u=n.headers.get("content-length"),f=u?parseInt(u):11e8,o=n.body.getReader(),a=new Uint8Array(f),p=0;for(;;){let{done:m,value:i}=await o.read();if(m)break;a.set(i,p),p+=i.length,this.reportProgress(p,f,"Downloading model...")}let d=new w(a),l=d.parse();z(l,this.config,e.contextLength),this.tokenizer=new U,this.tokenizer.extractFromGGUF(l),this.createPipelines(),this.createUniformBuffers(),this.uploadWeightsFromBuffer(d,l),await this.device.queue.onSubmittedWorkDone()}this.createWorkBuffers(),this.createBindGroups()}wb(e,r,t){this.device.queue.writeBuffer(e,r,t)}reportProgress(e,r,t){this.onProgress&&this.onProgress({loaded:e,total:r,status:t})}async initWebGPU(){if(!navigator.gpu)throw new Error("WebGPU not supported");let e=await navigator.gpu.requestAdapter({powerPreference:"high-performance"});if(!e)throw new Error("No WebGPU adapter found");this.device=await e.requestDevice({requiredLimits:{maxStorageBufferBindingSize:e.limits.maxStorageBufferBindingSize,maxBufferSize:e.limits.maxBufferSize}}),this.device.lost.then(r=>{this.deviceLost=!0,console.error(`WebGPU device lost: ${r.message} (reason: ${r.reason})`)})}createPipelines(){this.pipelines={};for(let[e,r]of Object.entries(C)){let t=this.device.createShaderModule({code:r});this.pipelines[e]=this.device.createComputePipeline({layout:"auto",compute:{module:t,entryPoint:"main"}})}}makeUniformMixed(e){let r=Math.max(e.length*4,16),t=new ArrayBuffer(r),s=new Uint32Array(t),n=new Float32Array(t);for(let u=0;u<e.length;u++){let f=e[u];"u"in f?s[u]=f.u:n[u]=f.f}let c=this.device.createBuffer({size:r,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,mappedAtCreation:!0});return new Uint8Array(c.getMappedRange()).set(new Uint8Array(t)),c.unmap(),c}createUniformBuffers(){let e=this.config,r=e.hidden_size,t=e.q_dim,s=e.kv_dim,n=e.intermediate_size,c=e.context_length,u=e.vocab_size,f=e.head_dim,o=e.num_q_heads,a=e.num_kv_heads;this.uniformBuffers={rmsNorm:this.makeUniformMixed([{u:r},{f:e.rms_norm_eps}]),perHeadRmsNormQ:this.makeUniformMixed([{u:o},{u:f},{f:e.rms_norm_eps},{u:0}]),perHeadRmsNormK:this.makeUniformMixed([{u:a},{u:f},{f:e.rms_norm_eps},{u:0}]),linearQ8_Q_H:this.makeUniformMixed([{u:t},{u:r}]),linearQ8_KV_H:this.makeUniformMixed([{u:s},{u:r}]),linearQ8_H_Q:this.makeUniformMixed([{u:r},{u:t}]),linearQ8_I_H:this.makeUniformMixed([{u:n},{u:r}]),linearQ8_H_I:this.makeUniformMixed([{u:r},{u:n}]),sizeH:this.makeUniformMixed([{u:r}]),sizeI:this.makeUniformMixed([{u:n}]),embeddingLookup:this.makeUniformMixed([{u:r},{u:0}]),ropeQ:[],ropeK:[],kvCacheStore:this.makeUniformMixed([{u:a},{u:f},{u:0},{u:c}]),attnScore:this.makeUniformMixed([{u:o},{u:a},{u:f},{u:0},{f:1/Math.sqrt(f)},{u:0}]),softmax:this.makeUniformMixed([{u:o},{u:0}]),attnOutput:this.makeUniformMixed([{u:o},{u:a},{u:f},{u:0}]),linearQ8_V_H:this.makeUniformMixed([{u},{u:r}]),argmaxSize:this.makeUniformMixed([{u}]),fusedNormRopeQ:[],fusedNormRopeK:[]};for(let p=0;p<e.num_layers;p++){let d=A(p,e);this.uniformBuffers.ropeQ.push(this.makeUniformMixed([{u:o},{u:f},{u:0},{f:d}])),this.uniformBuffers.ropeK.push(this.makeUniformMixed([{u:a},{u:f},{u:0},{f:d}])),this.uniformBuffers.fusedNormRopeQ.push(this.makeUniformMixed([{u:o},{u:f},{f:e.rms_norm_eps},{u:0},{f:d},{u:0}])),this.uniformBuffers.fusedNormRopeK.push(this.makeUniformMixed([{u:a},{u:f},{f:e.rms_norm_eps},{u:0},{f:d},{u:0}]))}}uploadWeightsFromBuffer(e,r){let t=r.tensors,s=r.dataOffset;this.modelBuffers={embeddingQ8:null,layers:[],finalNorm:null};let n=t.find(o=>o.name==="token_embd.weight");if(n)if(n.type===8){let o=e.getTensorDataQ8Packed(n,s);this.modelBuffers.embeddingQ8=this.device.createBuffer({size:o.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.wb(this.modelBuffers.embeddingQ8,0,o)}else{let o=e.getTensorData(n,s);this.modelBuffers.embeddingQ8=this.device.createBuffer({size:o.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.wb(this.modelBuffers.embeddingQ8,0,o)}let c=t.find(o=>o.name==="output_norm.weight");if(c){let o=e.getTensorData(c,s);this.modelBuffers.finalNorm=this.device.createBuffer({size:o.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.wb(this.modelBuffers.finalNorm,0,o)}let u=["attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down"],f=["attn_norm","ffn_norm","attn_q_norm","attn_k_norm","post_attention_norm","post_ffw_norm"];for(let o=0;o<this.config.num_layers;o++){let a=`blk.${o}.`,p={};for(let d of f){let l=t.find(m=>m.name===a+d+".weight");if(l){let m=e.getTensorData(l,s);p[d]=this.device.createBuffer({size:m.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.wb(p[d],0,m)}}for(let d of u){let l=t.find(m=>m.name===a+d+".weight");if(l){let m=e.getTensorDataQ8Packed(l,s);p[d]=this.device.createBuffer({size:m.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),this.wb(p[d],0,m)}}this.modelBuffers.layers.push(p)}}async uploadWeightsStreaming(e,r,t){this.modelBuffers={embeddingQ8:null,layers:[],finalNorm:null};let s=async(d,l)=>{let m=await fetch(e,{headers:{Range:`bytes=${d}-${d+l-1}`}});return new Uint8Array(await m.arrayBuffer())},n=(d,l,m)=>{let i=new w(d),b=Number(m.dims.reduce((g,v)=>g*v,1n));if(m.type===8){let g=i.packQ8_0ForGPU(l,b),v=this.device.createBuffer({size:g.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});return this.wb(v,0,g),{buf:v,packed:!0,byteLength:g.byteLength,numElements:b}}else{let g=b,v;if(m.type===0){v=new Float32Array(g);let B=new Uint8Array(d.buffer,d.byteOffset+l,g*4);new Uint8Array(v.buffer).set(B)}else v=i.dequantizeF16(l,g);let h=this.device.createBuffer({size:v.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});return this.wb(h,0,v),{buf:h,packed:!1,byteLength:v.byteLength,numElements:b}}},c=0,u=r.reduce((d,l)=>d+P(l),0),f=["attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down"],o=["attn_norm","ffn_norm","attn_q_norm","attn_k_norm","post_attention_norm","post_ffw_norm"],a=r.find(d=>d.name==="token_embd.weight");if(a){let d=t+Number(a.offset),l=P(a),m=await s(d,l),i=n(m,0,a);this.modelBuffers.embeddingQ8=i.buf,c+=l,this.reportProgress(c,u,"Streaming weights to GPU...")}let p=r.find(d=>d.name==="output_norm.weight");if(p){let d=t+Number(p.offset),l=P(p),m=await s(d,l),i=n(m,0,p);this.modelBuffers.finalNorm=i.buf,c+=l}for(let d=0;d<this.config.num_layers;d++){let l=`blk.${d}.`,m=r.filter(k=>k.name.startsWith(l));if(m.length===0)continue;let i=1/0,b=0;for(let k of m){let G=Number(k.offset),y=G+P(k);G<i&&(i=G),y>b&&(b=y)}let g=t+i,v=b-i,h=await s(g,v),B={};for(let k of o){let G=m.find(y=>y.name===l+k+".weight");if(G){let y=Number(G.offset)-i,N=n(h,y,G);B[k]=N.buf}}for(let k of f){let G=m.find(y=>y.name===l+k+".weight");if(G){let y=Number(G.offset)-i,N=n(h,y,G);B[k]=N.buf}}this.modelBuffers.layers.push(B),c+=v,this.reportProgress(c,u,`Layer ${d+1}/${this.config.num_layers}`)}}createWorkBuffers(){let e=this.config,r=e.hidden_size,t=e.q_dim,s=e.kv_dim,n=e.intermediate_size,c=e.context_length,u=e.vocab_size,f=e.head_dim,o=e.num_kv_heads,a=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,p=a|GPUBufferUsage.COPY_DST;this.workBuffers={hidden:this.device.createBuffer({size:r*4,usage:p}),hiddenReadback:this.device.createBuffer({size:r*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),residual:this.device.createBuffer({size:r*4,usage:a}),normed:this.device.createBuffer({size:r*4,usage:a}),q:this.device.createBuffer({size:t*4,usage:a}),k:this.device.createBuffer({size:s*4,usage:a}),v:this.device.createBuffer({size:s*4,usage:a}),attnOut:this.device.createBuffer({size:t*4,usage:a}),attnProj:this.device.createBuffer({size:r*4,usage:a}),postAttnNormed:this.device.createBuffer({size:r*4,usage:a}),attnScores:this.device.createBuffer({size:e.num_q_heads*c*4,usage:a}),ffnGate:this.device.createBuffer({size:n*4,usage:a}),ffnUp:this.device.createBuffer({size:n*4,usage:a}),ffnMul:this.device.createBuffer({size:n*4,usage:a}),ffnDown:this.device.createBuffer({size:r*4,usage:a}),postFfnNormed:this.device.createBuffer({size:r*4,usage:a}),logits:this.device.createBuffer({size:u*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),logitsReadback:this.device.createBuffer({size:u*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),argmaxResult:this.device.createBuffer({size:4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),argmaxReadback:this.device.createBuffer({size:4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),topk256Result:this.device.createBuffer({size:256*2*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),topk256Readback:this.device.createBuffer({size:256*2*4,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST})};let d=c*o*f*4;this.kvCaches=[];for(let l=0;l<e.num_layers;l++)this.kvCaches.push({k:this.device.createBuffer({size:d,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC}),v:this.device.createBuffer({size:d,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC})})}createBindGroups(){let e={embeddingLookup:this.device.createBindGroup({layout:this.pipelines.embeddingLookup.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.modelBuffers.embeddingQ8}},{binding:1,resource:{buffer:this.workBuffers.hidden}},{binding:2,resource:{buffer:this.uniformBuffers.embeddingLookup}}]}),finalNorm:this.device.createBindGroup({layout:this.pipelines.rmsNorm.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.hidden}},{binding:1,resource:{buffer:this.modelBuffers.finalNorm}},{binding:2,resource:{buffer:this.workBuffers.normed}},{binding:3,resource:{buffer:this.uniformBuffers.rmsNorm}}]}),lmHead:this.device.createBindGroup({layout:this.pipelines.linearQ8.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.normed}},{binding:1,resource:{buffer:this.modelBuffers.embeddingQ8}},{binding:2,resource:{buffer:this.workBuffers.logits}},{binding:3,resource:{buffer:this.uniformBuffers.linearQ8_V_H}}]}),argmax:this.device.createBindGroup({layout:this.pipelines.argmax.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.logits}},{binding:1,resource:{buffer:this.workBuffers.argmaxResult}},{binding:2,resource:{buffer:this.uniformBuffers.argmaxSize}}]}),topk256:this.device.createBindGroup({layout:this.pipelines.topk256.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.logits}},{binding:1,resource:{buffer:this.workBuffers.topk256Result}},{binding:2,resource:{buffer:this.uniformBuffers.argmaxSize}}]}),layers:[]};for(let r=0;r<this.config.num_layers;r++){let t=this.modelBuffers.layers[r],s=this.kvCaches[r],n={attnNorm:this.device.createBindGroup({layout:this.pipelines.rmsNorm.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.hidden}},{binding:1,resource:{buffer:t.attn_norm}},{binding:2,resource:{buffer:this.workBuffers.normed}},{binding:3,resource:{buffer:this.uniformBuffers.rmsNorm}}]}),linearQ:this.device.createBindGroup({layout:this.pipelines.linearQ8.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.normed}},{binding:1,resource:{buffer:t.attn_q}},{binding:2,resource:{buffer:this.workBuffers.q}},{binding:3,resource:{buffer:this.uniformBuffers.linearQ8_Q_H}}]}),linearK:this.device.createBindGroup({layout:this.pipelines.linearQ8.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.normed}},{binding:1,resource:{buffer:t.attn_k}},{binding:2,resource:{buffer:this.workBuffers.k}},{binding:3,resource:{buffer:this.uniformBuffers.linearQ8_KV_H}}]}),linearV:this.device.createBindGroup({layout:this.pipelines.linearQ8.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.normed}},{binding:1,resource:{buffer:t.attn_v}},{binding:2,resource:{buffer:this.workBuffers.v}},{binding:3,resource:{buffer:this.uniformBuffers.linearQ8_KV_H}}]}),ropeQ:this.device.createBindGroup({layout:this.pipelines.rope.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.q}},{binding:1,resource:{buffer:this.uniformBuffers.ropeQ[r]}}]}),ropeK:this.device.createBindGroup({layout:this.pipelines.rope.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.k}},{binding:1,resource:{buffer:this.uniformBuffers.ropeK[r]}}]}),qNorm:this.device.createBindGroup({layout:this.pipelines.perHeadRmsNorm.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.q}},{binding:1,resource:{buffer:t.attn_q_norm}},{binding:2,resource:{buffer:this.uniformBuffers.perHeadRmsNormQ}}]}),kNorm:this.device.createBindGroup({layout:this.pipelines.perHeadRmsNorm.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.k}},{binding:1,resource:{buffer:t.attn_k_norm}},{binding:2,resource:{buffer:this.uniformBuffers.perHeadRmsNormK}}]}),fusedNormRopeQ:this.device.createBindGroup({layout:this.pipelines.fusedPerHeadNormRope.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.q}},{binding:1,resource:{buffer:t.attn_q_norm}},{binding:2,resource:{buffer:this.uniformBuffers.fusedNormRopeQ[r]}}]}),fusedNormRopeK:this.device.createBindGroup({layout:this.pipelines.fusedPerHeadNormRope.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.k}},{binding:1,resource:{buffer:t.attn_k_norm}},{binding:2,resource:{buffer:this.uniformBuffers.fusedNormRopeK[r]}}]}),kvStore:this.device.createBindGroup({layout:this.pipelines.kvCacheStore.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.k}},{binding:1,resource:{buffer:this.workBuffers.v}},{binding:2,resource:{buffer:s.k}},{binding:3,resource:{buffer:s.v}},{binding:4,resource:{buffer:this.uniformBuffers.kvCacheStore}}]}),attnScore:this.device.createBindGroup({layout:this.pipelines.attnScore.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.q}},{binding:1,resource:{buffer:s.k}},{binding:2,resource:{buffer:this.workBuffers.attnScores}},{binding:3,resource:{buffer:this.uniformBuffers.attnScore}}]}),softmax:this.device.createBindGroup({layout:this.pipelines.softmax.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.attnScores}},{binding:1,resource:{buffer:this.uniformBuffers.softmax}}]}),attnOutput:this.device.createBindGroup({layout:this.pipelines.attnOutput.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.attnScores}},{binding:1,resource:{buffer:s.v}},{binding:2,resource:{buffer:this.workBuffers.attnOut}},{binding:3,resource:{buffer:this.uniformBuffers.attnOutput}}]}),linearAttnOut:this.device.createBindGroup({layout:this.pipelines.linearQ8.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.attnOut}},{binding:1,resource:{buffer:t.attn_output}},{binding:2,resource:{buffer:this.workBuffers.attnProj}},{binding:3,resource:{buffer:this.uniformBuffers.linearQ8_H_Q}}]}),postAttnNorm:this.device.createBindGroup({layout:this.pipelines.rmsNorm.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.attnProj}},{binding:1,resource:{buffer:t.post_attention_norm}},{binding:2,resource:{buffer:this.workBuffers.postAttnNormed}},{binding:3,resource:{buffer:this.uniformBuffers.rmsNorm}}]}),residualAdd1:this.device.createBindGroup({layout:this.pipelines.add.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.hidden}},{binding:1,resource:{buffer:this.workBuffers.postAttnNormed}},{binding:2,resource:{buffer:this.workBuffers.residual}},{binding:3,resource:{buffer:this.uniformBuffers.sizeH}}]}),ffnNorm:this.device.createBindGroup({layout:this.pipelines.rmsNorm.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.residual}},{binding:1,resource:{buffer:t.ffn_norm}},{binding:2,resource:{buffer:this.workBuffers.normed}},{binding:3,resource:{buffer:this.uniformBuffers.rmsNorm}}]}),ffnGate:this.device.createBindGroup({layout:this.pipelines.linearQ8.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.normed}},{binding:1,resource:{buffer:t.ffn_gate}},{binding:2,resource:{buffer:this.workBuffers.ffnGate}},{binding:3,resource:{buffer:this.uniformBuffers.linearQ8_I_H}}]}),ffnUp:this.device.createBindGroup({layout:this.pipelines.linearQ8.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.normed}},{binding:1,resource:{buffer:t.ffn_up}},{binding:2,resource:{buffer:this.workBuffers.ffnUp}},{binding:3,resource:{buffer:this.uniformBuffers.linearQ8_I_H}}]}),geluMul:this.device.createBindGroup({layout:this.pipelines.geluMul.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.ffnGate}},{binding:1,resource:{buffer:this.workBuffers.ffnUp}},{binding:2,resource:{buffer:this.workBuffers.ffnMul}},{binding:3,resource:{buffer:this.uniformBuffers.sizeI}}]}),ffnDown:this.device.createBindGroup({layout:this.pipelines.linearQ8.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.ffnMul}},{binding:1,resource:{buffer:t.ffn_down}},{binding:2,resource:{buffer:this.workBuffers.ffnDown}},{binding:3,resource:{buffer:this.uniformBuffers.linearQ8_H_I}}]}),postFfnNorm:this.device.createBindGroup({layout:this.pipelines.rmsNorm.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.ffnDown}},{binding:1,resource:{buffer:t.post_ffw_norm}},{binding:2,resource:{buffer:this.workBuffers.postFfnNormed}},{binding:3,resource:{buffer:this.uniformBuffers.rmsNorm}}]}),residualAdd2:this.device.createBindGroup({layout:this.pipelines.add.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.residual}},{binding:1,resource:{buffer:this.workBuffers.postFfnNormed}},{binding:2,resource:{buffer:this.workBuffers.hidden}},{binding:3,resource:{buffer:this.uniformBuffers.sizeH}}]}),fusedPostAttnNormAdd:this.device.createBindGroup({layout:this.pipelines.fusedNormAdd.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.attnProj}},{binding:1,resource:{buffer:t.post_attention_norm}},{binding:2,resource:{buffer:this.workBuffers.hidden}},{binding:3,resource:{buffer:this.workBuffers.residual}},{binding:4,resource:{buffer:this.uniformBuffers.rmsNorm}}]}),fusedPostFfnNormAdd:this.device.createBindGroup({layout:this.pipelines.fusedNormAdd.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.workBuffers.ffnDown}},{binding:1,resource:{buffer:t.post_ffw_norm}},{binding:2,resource:{buffer:this.workBuffers.residual}},{binding:3,resource:{buffer:this.workBuffers.hidden}},{binding:4,resource:{buffer:this.uniformBuffers.rmsNorm}}]})};e.layers.push(n)}this.bindGroupCache=e}encodeTransformerPass(e,r,t){let s=this.config,n=s.hidden_size,c=s.q_dim,u=s.kv_dim,f=s.intermediate_size,o=s.head_dim,a=s.num_q_heads,p=s.num_kv_heads,d=t+1;this.wb(this.uniformBuffers.embeddingLookup,4,new Uint32Array([r]));let l=new Uint32Array([t]);for(let b=0;b<s.num_layers;b++)this.wb(this.uniformBuffers.fusedNormRopeQ[b],12,l),this.wb(this.uniformBuffers.fusedNormRopeK[b],12,l);this.wb(this.uniformBuffers.kvCacheStore,8,l);let m=new Uint32Array([d]);this.wb(this.uniformBuffers.attnScore,12,m),this.wb(this.uniformBuffers.softmax,4,m),this.wb(this.uniformBuffers.attnOutput,12,m);let i;i=e.beginComputePass(),i.setPipeline(this.pipelines.embeddingLookup),i.setBindGroup(0,this.bindGroupCache.embeddingLookup),i.dispatchWorkgroups(Math.ceil(n/256)),i.end();for(let b=0;b<s.num_layers;b++){let g=this.bindGroupCache.layers[b];i=e.beginComputePass(),i.setPipeline(this.pipelines.rmsNorm),i.setBindGroup(0,g.attnNorm),i.dispatchWorkgroups(1),i.end(),i=e.beginComputePass(),i.setPipeline(this.pipelines.linearQ8),i.setBindGroup(0,g.linearQ),i.dispatchWorkgroups(Math.ceil(c/256)),i.setBindGroup(0,g.linearK),i.dispatchWorkgroups(Math.ceil(u/256)),i.setBindGroup(0,g.linearV),i.dispatchWorkgroups(Math.ceil(u/256)),i.end(),i=e.beginComputePass(),i.setPipeline(this.pipelines.fusedPerHeadNormRope),i.setBindGroup(0,g.fusedNormRopeQ),i.dispatchWorkgroups(a),i.setBindGroup(0,g.fusedNormRopeK),i.dispatchWorkgroups(p),i.end(),i=e.beginComputePass(),i.setPipeline(this.pipelines.kvCacheStore),i.setBindGroup(0,g.kvStore),i.dispatchWorkgroups(Math.ceil(u/256)),i.end(),i=e.beginComputePass(),i.setPipeline(this.pipelines.attnScore),i.setBindGroup(0,g.attnScore),i.dispatchWorkgroups(Math.ceil(a*d/256)),i.end(),i=e.beginComputePass(),i.setPipeline(this.pipelines.softmax),i.setBindGroup(0,g.softmax),i.dispatchWorkgroups(a),i.end(),i=e.beginComputePass(),i.setPipeline(this.pipelines.attnOutput),i.setBindGroup(0,g.attnOutput),i.dispatchWorkgroups(Math.ceil(a*o/256)),i.end(),i=e.beginComputePass(),i.setPipeline(this.pipelines.linearQ8),i.setBindGroup(0,g.linearAttnOut),i.dispatchWorkgroups(Math.ceil(n/256)),i.end(),i=e.beginComputePass(),i.setPipeline(this.pipelines.fusedNormAdd),i.setBindGroup(0,g.fusedPostAttnNormAdd),i.dispatchWorkgroups(1),i.end(),i=e.beginComputePass(),i.setPipeline(this.pipelines.rmsNorm),i.setBindGroup(0,g.ffnNorm),i.dispatchWorkgroups(1),i.end(),i=e.beginComputePass(),i.setPipeline(this.pipelines.linearQ8),i.setBindGroup(0,g.ffnGate),i.dispatchWorkgroups(Math.ceil(f/256)),i.setBindGroup(0,g.ffnUp),i.dispatchWorkgroups(Math.ceil(f/256)),i.end(),i=e.beginComputePass(),i.setPipeline(this.pipelines.geluMul),i.setBindGroup(0,g.geluMul),i.dispatchWorkgroups(Math.ceil(f/256)),i.end(),i=e.beginComputePass(),i.setPipeline(this.pipelines.linearQ8),i.setBindGroup(0,g.ffnDown),i.dispatchWorkgroups(Math.ceil(n/256)),i.end(),i=e.beginComputePass(),i.setPipeline(this.pipelines.fusedNormAdd),i.setBindGroup(0,g.fusedPostFfnNormAdd),i.dispatchWorkgroups(1),i.end()}i=e.beginComputePass(),i.setPipeline(this.pipelines.rmsNorm),i.setBindGroup(0,this.bindGroupCache.finalNorm),i.dispatchWorkgroups(1),i.end()}async sampleNextToken(e,r,t,s,n){if(this.deviceLost)throw new Error("WebGPU device lost");let c=this.config.vocab_size,u;u=e.beginComputePass(),u.setPipeline(this.pipelines.linearQ8),u.setBindGroup(0,this.bindGroupCache.lmHead),u.dispatchWorkgroups(Math.ceil(c/256)),u.end();let f=r===0&&s<=1;if(f?(u=e.beginComputePass(),u.setPipeline(this.pipelines.argmax),u.setBindGroup(0,this.bindGroupCache.argmax),u.dispatchWorkgroups(1),u.end(),e.copyBufferToBuffer(this.workBuffers.argmaxResult,0,this.workBuffers.argmaxReadback,0,4)):(u=e.beginComputePass(),u.setPipeline(this.pipelines.topk256),u.setBindGroup(0,this.bindGroupCache.topk256),u.dispatchWorkgroups(1),u.end(),e.copyBufferToBuffer(this.workBuffers.topk256Result,0,this.workBuffers.topk256Readback,0,256*2*4)),this.device.queue.submit([e.finish()]),f){try{await this.workBuffers.argmaxReadback.mapAsync(GPUMapMode.READ)}catch(k){throw new Error(`GPU readback failed (device lost?): ${k}`)}let B=new Uint32Array(this.workBuffers.argmaxReadback.getMappedRange())[0];return this.workBuffers.argmaxReadback.unmap(),B}try{await this.workBuffers.topk256Readback.mapAsync(GPUMapMode.READ)}catch(h){throw new Error(`GPU readback failed (device lost?): ${h}`)}let o=new Float32Array(this.workBuffers.topk256Readback.getMappedRange().slice(0));this.workBuffers.topk256Readback.unmap();let a=new Array(256),p=new Uint32Array(o.buffer.slice(0));for(let h=0;h<256;h++)a[h]={val:o[h*2],id:p[h*2+1]};if(s>1&&n.length>0){let h=new Set(n);for(let B=0;B<256;B++)h.has(a[B].id)&&(a[B].val>0?a[B].val/=s:a[B].val*=s)}if(a.sort((h,B)=>B.val-h.val),r===0)return a[0].id;let d=a[0].val,l=0,m=new Float32Array(256);for(let h=0;h<256;h++)m[h]=Math.exp((a[h].val-d)/r),l+=m[h];let i=0,b=256;for(let h=0;h<256;h++)if(i+=m[h]/l,i>=t){b=h+1;break}let g=0;for(let h=0;h<b;h++)g+=m[h];let v=Math.random()*g;for(let h=0;h<b;h++)if(v-=m[h],v<=0)return a[h].id;return a[b-1].id}async forwardPassAndGetToken(e,r,t=0,s=.9,n=1,c=[]){let u=this.device.createCommandEncoder();return this.encodeTransformerPass(u,e,r),this.sampleNextToken(u,t,s,n,c)}forwardPassOnly(e,r){let t=this.device.createCommandEncoder();this.encodeTransformerPass(t,e,r),this.device.queue.submit([t.finish()])}async prefillBatched(e,r=0){for(let t=0;t<e.length;t++)this.forwardPassOnly(e[t],r+t);await this.device.queue.onSubmittedWorkDone()}resetKVCaches(){let e=this.config.head_dim,r=this.config.num_kv_heads,t=this.config.context_length,s=new Float32Array(t*r*e);for(let n=0;n<this.config.num_layers;n++)this.wb(this.kvCaches[n].k,0,s),this.wb(this.kvCaches[n].v,0,s)}async getFirstTokenAfterPrefill(e,r,t,s){let n=this.device.createCommandEncoder();return this.sampleNextToken(n,e,r,t,s)}addUserMessage(e){this.conversationHistory.push({role:"user",text:e})}async*generate(e={}){if(this.deviceLost)throw new Error("WebGPU device lost \u2014 call dispose() and recreate the engine");let r=e.temperature??.7,t=e.topP??.9,s=e.repPenalty??1.2,n=e.maxTokens??32768,c=e.toolsJson??"[]",u=e.signal,f;if(this.kvPosition===0){let i=T(this.conversationHistory,c);f=this.tokenizer.encode(i)}else{let b=`<end_of_turn>
<start_of_turn>user
${this.conversationHistory[this.conversationHistory.length-1].text}<end_of_turn>
<start_of_turn>model
`;f=this.tokenizer.encode(b).slice(1)}if(this.kvPosition+f.length>=this.config.context_length-10){let i=this.conversationHistory[this.conversationHistory.length-1];this.conversationHistory=[{role:"user",text:i.text}];let b=T(this.conversationHistory,c);f=this.tokenizer.encode(b),this.resetKVCaches(),this.kvPosition=0}await this.prefillBatched(f,this.kvPosition);let o=[...f];this.kvPosition+=f.length;let a=await this.getFirstTokenAfterPrefill(r,t,s,o);o.push(a);let p=this.tokenizer.funcTokens["<end_function_call>"],d=[a],l=0;yield this.tokenizer.decodeToken(a);for(let i=1;i<n&&!(a===1||a===0||a===106||p&&a===p||u?.aborted);i++){let b=this.kvPosition+l;if(b>=this.config.context_length-1||(a=await this.forwardPassAndGetToken(a,b,r,t,s,o),l++,a===1||a===0||a===106))break;if(p&&a===p){o.push(a),d.push(a);break}o.push(a),d.push(a),yield this.tokenizer.decodeToken(a)}let m=this.tokenizer.decodeTokens(d);this.conversationHistory.push({role:"model",text:m}),this.kvPosition+=l}resetConversation(){this.conversationHistory=[],this.kvPosition=0,this.resetKVCaches()}dispose(){let e=r=>{r&&r.destroy()};if(e(this.modelBuffers?.embeddingQ8),e(this.modelBuffers?.finalNorm),this.modelBuffers?.layers)for(let r of this.modelBuffers.layers)for(let t of Object.values(r))e(t);if(this.workBuffers)for(let r of Object.values(this.workBuffers))e(r);if(this.kvCaches)for(let r of this.kvCaches)e(r.k),e(r.v);if(this.uniformBuffers)for(let r of Object.values(this.uniformBuffers))if(Array.isArray(r))for(let t of r)e(t);else e(r);this.device?.destroy()}};async function Y(_={}){let e=new x(_);return await e.init(_),e}export{Y as createGemmaEngine};
