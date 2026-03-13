export const SHADERS: Record<string, string> = {
  embeddingLookup: `
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
}`,

  rmsNorm: `
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
}`,

  perHeadRmsNorm: `
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
}`,

  linearQ8: `
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
}`,

  geluMul: `
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
}`,

  add: `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> size: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= size) { return; }
  output[i] = a[i] + b[i];
}`,

  rope: `
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
}`,

  kvCacheStore: `
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
}`,

  attnScore: `
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
}`,

  softmax: `
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
}`,

  attnOutput: `
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
}`,

  argmax: `
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
}`,

  topk256: `
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
}`,

  fusedNormAdd: `
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
}`,

  fusedPerHeadNormRope: `
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
}`,
};
