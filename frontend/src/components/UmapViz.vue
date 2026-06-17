<template>
  <div class="umap-container">
    <div class="umap-card">
      <h3 class="viz-title">UMAP: Clean Embedding (z)</h3>
      <div class="canvas-wrap">
        <div v-if="!hasData" class="empty-overlay">Collecting embeddings…</div>
        <canvas ref="cleanCanvas"></canvas>
      </div>
    </div>
    <div class="umap-card">
      <h3 class="viz-title">UMAP: Noisy Embedding (z<sub>t</sub>)</h3>
      <div class="canvas-wrap">
        <div v-if="!hasData" class="empty-overlay">Collecting embeddings…</div>
        <canvas ref="noisyCanvas"></canvas>
      </div>
    </div>
    <div class="umap-card">
      <h3 class="viz-title">UMAP: Denoised Embedding (ẑ)</h3>
      <div class="canvas-wrap">
        <div v-if="!hasData" class="empty-overlay">Collecting embeddings…</div>
        <canvas ref="denoisedCanvas"></canvas>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, onMounted, onUnmounted, toRaw } from 'vue'

const genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
const genreColors = [
  '#3b82f6', '#64748b', '#14b8a6', '#ec4899', '#ef4444',
  '#eab308', '#f97316', '#06b6d4', '#22c55e', '#8b5cf6'
]

function hexToRgb(hex) {
  return [
    parseInt(hex.slice(1, 3), 16) / 255,
    parseInt(hex.slice(3, 5), 16) / 255,
    parseInt(hex.slice(5, 7), 16) / 255
  ]
}
const genreColorsRgb = genreColors.map(hexToRgb)

const props = defineProps({
  modelType: { type: String, default: 'cnn' },
  cleanEmbHistory: { type: Array, default: () => [] },
  noisyEmbHistory: { type: Array, default: () => [] },
  denoisedEmbHistory: { type: Array, default: () => [] },
  genreIndexHistory: { type: Array, default: () => [] }
})

const flushInterval = 3

const cleanCanvas = ref(null)
const noisyCanvas = ref(null)
const denoisedCanvas = ref(null)

const hasData = ref(false)

// ── Pre-allocated flat buffers for zero-GC animation ────────────────
// Positions are stored as flat [x0,y0,x1,y1,...] to avoid nested array allocations.
const MAX_PTS = 64

const targetCleanFlat = new Float32Array(MAX_PTS * 2)
const targetNoisyFlat = new Float32Array(MAX_PTS * 2)
const targetDenoisedFlat = new Float32Array(MAX_PTS * 2)

const currentCleanFlat = new Float32Array(MAX_PTS * 2)
const currentNoisyFlat = new Float32Array(MAX_PTS * 2)
const currentDenoisedFlat = new Float32Array(MAX_PTS * 2)

// Shared temp buffer for computing visual positions during transitions
const _tempVisual = new Float32Array(MAX_PTS * 2)

let cachedGenreSlice = null
let pointCount = 0
let currentSliceStart = 0

let animFrameId = null
let currentProgress = 1.0
let animStartTime = 0
const ANIM_DURATION = 600 // ms

// WebGL Instances
let glClean = null
let glNoisy = null
let glDenoised = null

// Worker state
let worker = null
let workerBusy = false
let pendingRequest = false

// ── Flat buffer helpers (zero allocation) ───────────────────────────

/** Flatten number[][] from worker into a pre-allocated Float32Array */
function flattenInto(src, dst) {
  for (let i = 0; i < src.length; i++) {
    dst[i * 2] = src[i][0]
    dst[i * 2 + 1] = src[i][1]
  }
}

/**
 * Compute visual (on-screen) positions at current animation progress.
 * Writes into the shared _tempVisual buffer. Zero allocation.
 */
function computeVisual(currentFlat, targetFlat, count, progress) {
  const easeOut = 1.0 - Math.pow(1.0 - progress, 3.0)
  for (let i = 0; i < count * 2; i++) {
    _tempVisual[i] = currentFlat[i] + (targetFlat[i] - currentFlat[i]) * easeOut
  }
}

/**
 * Sync old visual positions → new current buffer based on slice index alignment.
 * Points that existed in both old and new windows keep their visual position;
 * new points spawn from the last known position or the target position.
 * Zero allocation.
 */
function syncFlat(visualFlat, oldCount, oldSliceStart, newCurrentFlat, newCount, newSliceStart, newWorkerData) {
  for (let i = 0; i < newCount; i++) {
    const absIdx = newSliceStart + i
    const oldIdx = absIdx - oldSliceStart
    if (oldIdx >= 0 && oldIdx < oldCount) {
      newCurrentFlat[i * 2] = visualFlat[oldIdx * 2]
      newCurrentFlat[i * 2 + 1] = visualFlat[oldIdx * 2 + 1]
    } else if (oldCount > 0) {
      // New point outside old window: spawn from last known position
      newCurrentFlat[i * 2] = visualFlat[(oldCount - 1) * 2]
      newCurrentFlat[i * 2 + 1] = visualFlat[(oldCount - 1) * 2 + 1]
    } else {
      // First result: snap to target (no animation from origin)
      newCurrentFlat[i * 2] = newWorkerData[i][0]
      newCurrentFlat[i * 2 + 1] = newWorkerData[i][1]
    }
  }
}

/**
 * Process one embedding lane when a worker result arrives.
 * All operations are in-place on pre-allocated buffers — zero GC pressure.
 */
function processLane(currentFlat, targetFlat, newWorkerData, newCount, newSliceStart) {
  // Step 1: Capture visual positions at the moment of interruption → _tempVisual
  computeVisual(currentFlat, targetFlat, pointCount, currentProgress)
  // Step 2: Remap old visual positions → new current positions (index alignment)
  syncFlat(_tempVisual, pointCount, currentSliceStart, currentFlat, newCount, newSliceStart, newWorkerData)
  // Step 3: Flatten new UMAP output → target buffer
  flattenInto(newWorkerData, targetFlat)
}

// ── Worker ──────────────────────────────────────────────────────────

function initWorker() {
  worker = new Worker(
    new URL('../workers/umap-worker.js', import.meta.url),
    { type: 'module' }
  )
  worker.onmessage = (e) => {
    if (e.data.type === 'result') {
      const { clean2d, noisy2d, denoised2d, genreIndices, sliceStart } = e.data
      const N = clean2d ? clean2d.length : 0

      if (N > 0) {
        // Process each lane sequentially (safe: shares _tempVisual, but no concurrency)
        processLane(currentCleanFlat, targetCleanFlat, clean2d, N, sliceStart)
        processLane(currentNoisyFlat, targetNoisyFlat, noisy2d, N, sliceStart)
        processLane(currentDenoisedFlat, targetDenoisedFlat, denoised2d, N, sliceStart)
      }

      cachedGenreSlice = genreIndices
      pointCount = N
      currentSliceStart = sliceStart

      // Upload to GPU — once per worker payload
      glClean?.updateBuffers(currentCleanFlat, targetCleanFlat, N, cachedGenreSlice)
      glNoisy?.updateBuffers(currentNoisyFlat, targetNoisyFlat, N, cachedGenreSlice)
      glDenoised?.updateBuffers(currentDenoisedFlat, targetDenoisedFlat, N, cachedGenreSlice)

      hasData.value = true
      currentProgress = 0.0
      animStartTime = 0

      if (!animFrameId) animFrameId = requestAnimationFrame(animateLoop)

      workerBusy = false
      if (pendingRequest) {
        pendingRequest = false
        requestUMAP()
      }
    } else if (e.data.type === 'error') {
      console.error('UMAP worker error:', e.data.message)
      workerBusy = false
    }
  }
}

function requestUMAP() {
  if (!worker || props.cleanEmbHistory.length < flushInterval) return
  if (workerBusy) { pendingRequest = true; return }
  workerBusy = true
  worker.postMessage({
    type: 'compute',
    clean: toRaw(props.cleanEmbHistory),
    noisy: toRaw(props.noisyEmbHistory),
    denoised: toRaw(props.denoisedEmbHistory),
    genreIndices: toRaw(props.genreIndexHistory)
  })
}

// ── WebGL Renderer Class ────────────────────────────────────────────

const vsSource = `#version 300 es
in vec2 a_start_position;
in vec2 a_target_position;
in vec3 a_color;
in float a_isLast;

uniform vec2 u_min_start;
uniform vec2 u_range_start;
uniform vec2 u_min_target;
uniform vec2 u_range_target;
uniform float u_progress;
uniform vec2 u_resolution;
uniform float u_dpr;

out vec3 v_color;
out float v_isLast;

void main() {
    float easeOut = 1.0 - pow(1.0 - u_progress, 3.0);

    vec2 pos = mix(a_start_position, a_target_position, easeOut);
    vec2 current_min = mix(u_min_start, u_min_target, easeOut);
    vec2 current_range = mix(u_range_start, u_range_target, easeOut);

    vec2 norm = (pos - current_min) / current_range;
    
    // Convert 18px logical padding to NDC ratio based on dynamic resolution
    vec2 padding = (18.0 * u_dpr) / u_resolution;

    vec2 screenNorm = vec2(
        mix(padding.x, 1.0 - padding.x, norm.x),
        mix(padding.y, 1.0 - padding.y, norm.y)
    );
    screenNorm.y = 1.0 - screenNorm.y; // Match Canvas2D orientation

    gl_Position = vec4(screenNorm * 2.0 - 1.0, 0.0, 1.0);
    gl_PointSize = ((a_isLast > 0.5) ? 24.0 : 9.0) * u_dpr;

    v_color = a_color;
    v_isLast = a_isLast;
}
`

const fsSource = `#version 300 es
precision highp float;
in vec3 v_color;
in float v_isLast;
out vec4 outColor;

void main() {
    vec2 p = gl_PointCoord * 2.0 - vec2(1.0);
    float dist = length(p);
    if (dist > 1.0) discard;

    if (v_isLast > 0.5) {
        if (dist < 0.25) {
            float mixEdge = smoothstep(0.20, 0.25, dist);
            outColor = mix(vec4(1.0), vec4(v_color, 1.0), mixEdge);
        } else if (dist < 0.58) {
            outColor = vec4(v_color, 1.0);
        } else {
            float alpha = mix(0.25, 0.0, smoothstep(0.58, 1.0, dist));
            outColor = vec4(v_color, alpha);
        }
    } else {
        float alpha = mix(0.8, 0.0, smoothstep(0.8, 1.0, dist));
        outColor = vec4(v_color, alpha);
    }
}
`

class WebGLUmap {
  constructor(canvas) {
    this.canvas = canvas
    this.gl = canvas.getContext('webgl2', { antialias: false, premultipliedAlpha: false })
    const gl = this.gl
    if (!gl) return

    // Compile shaders with error checking
    const vs = this._compileShader(gl.VERTEX_SHADER, vsSource)
    const fs = this._compileShader(gl.FRAGMENT_SHADER, fsSource)
    if (!vs || !fs) return

    this.program = gl.createProgram()
    gl.attachShader(this.program, vs)
    gl.attachShader(this.program, fs)
    gl.linkProgram(this.program)

    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      console.error('UMAP shader link error:', gl.getProgramInfoLog(this.program))
      return
    }

    // Detach & delete shader objects after successful linking (free driver memory)
    gl.detachShader(this.program, vs)
    gl.detachShader(this.program, fs)
    gl.deleteShader(vs)
    gl.deleteShader(fs)

    // Cache uniform locations
    this.uMinStartLoc = gl.getUniformLocation(this.program, 'u_min_start')
    this.uRangeStartLoc = gl.getUniformLocation(this.program, 'u_range_start')
    this.uMinTargetLoc = gl.getUniformLocation(this.program, 'u_min_target')
    this.uRangeTargetLoc = gl.getUniformLocation(this.program, 'u_range_target')
    this.uProgressLoc = gl.getUniformLocation(this.program, 'u_progress')
    this.uResolutionLoc = gl.getUniformLocation(this.program, 'u_resolution')
    this.uDprLoc = gl.getUniformLocation(this.program, 'u_dpr')

    // Create VBO
    this.vbo = gl.createBuffer()

    // Create VAO — bind vertex layout ONCE, reuse every frame
    const STRIDE = 32 // 8 floats × 4 bytes
    const aStartLoc = gl.getAttribLocation(this.program, 'a_start_position')
    const aTargetLoc = gl.getAttribLocation(this.program, 'a_target_position')
    const aColLoc = gl.getAttribLocation(this.program, 'a_color')
    const aIsLastLoc = gl.getAttribLocation(this.program, 'a_isLast')

    this.vao = gl.createVertexArray()
    gl.bindVertexArray(this.vao)
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo)

    gl.enableVertexAttribArray(aStartLoc)
    gl.vertexAttribPointer(aStartLoc, 2, gl.FLOAT, false, STRIDE, 0)
    gl.enableVertexAttribArray(aTargetLoc)
    gl.vertexAttribPointer(aTargetLoc, 2, gl.FLOAT, false, STRIDE, 8)
    gl.enableVertexAttribArray(aColLoc)
    gl.vertexAttribPointer(aColLoc, 3, gl.FLOAT, false, STRIDE, 16)
    gl.enableVertexAttribArray(aIsLastLoc)
    gl.vertexAttribPointer(aIsLastLoc, 1, gl.FLOAT, false, STRIDE, 28)

    gl.bindVertexArray(null)

    gl.enable(gl.BLEND)
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)

    this.N = 0
    this.bufferData = null
    this.minStart = [0, 0]
    this.rangeStart = [1, 1]
    this.minTarget = [0, 0]
    this.rangeTarget = [1, 1]
  }

  _compileShader(type, source) {
    const gl = this.gl
    const shader = gl.createShader(type)
    gl.shaderSource(shader, source)
    gl.compileShader(shader)
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error('UMAP shader compile error:', gl.getShaderInfoLog(shader))
      gl.deleteShader(shader)
      return null
    }
    return shader
  }

  resizeToDisplaySize() {
    if (!this.gl) return false
    const gl = this.gl
    const dpr = window.devicePixelRatio || 1
    const displayWidth = Math.floor(gl.canvas.clientWidth * dpr)
    const displayHeight = Math.floor(gl.canvas.clientHeight * dpr)

    if (gl.canvas.width !== displayWidth || gl.canvas.height !== displayHeight) {
      gl.canvas.width = Math.max(1, displayWidth)
      gl.canvas.height = Math.max(1, displayHeight)
      return true
    }
    return false
  }

  clear() {
    if (!this.gl) return
    const gl = this.gl
    this.resizeToDisplaySize()
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height)
    gl.clearColor(0.972, 0.980, 0.988, 1.0)
    gl.clear(gl.COLOR_BUFFER_BIT)
  }

  /**
   * Upload vertex data from flat position buffers.
   * startFlat/targetFlat are Float32Array in [x0,y0,x1,y1,...] format.
   */
  updateBuffers(startFlat, targetFlat, N, genreIdx) {
    if (!this.gl) return
    const gl = this.gl
    this.N = N

    if (N === 0) { this.clear(); return }

    // Compute AABB bounds for start and target (used for viewport normalization in shader)
    let xMinS = Infinity, xMaxS = -Infinity, yMinS = Infinity, yMaxS = -Infinity
    let xMinT = Infinity, xMaxT = -Infinity, yMinT = Infinity, yMaxT = -Infinity

    for (let i = 0; i < N; i++) {
      const sx = startFlat[i * 2], sy = startFlat[i * 2 + 1]
      if (sx < xMinS) xMinS = sx; if (sx > xMaxS) xMaxS = sx
      if (sy < yMinS) yMinS = sy; if (sy > yMaxS) yMaxS = sy

      const tx = targetFlat[i * 2], ty = targetFlat[i * 2 + 1]
      if (tx < xMinT) xMinT = tx; if (tx > xMaxT) xMaxT = tx
      if (ty < yMinT) yMinT = ty; if (ty > yMaxT) yMaxT = ty
    }

    this.minStart = [xMinS, yMinS]
    this.rangeStart = [(xMaxS - xMinS) || 1, (yMaxS - yMinS) || 1]
    this.minTarget = [xMinT, yMinT]
    this.rangeTarget = [(xMaxT - xMinT) || 1, (yMaxT - yMinT) || 1]

    // Build interleaved vertex data: [startXY(2) | targetXY(2) | color(3) | isLast(1)] × N
    const STRIDE_FLOATS = 8
    const neededSize = N * STRIDE_FLOATS
    if (!this.bufferData || this.bufferData.length < neededSize) {
      this.bufferData = new Float32Array(neededSize * 2) // 2× headroom to avoid frequent realloc
    }

    for (let i = 0; i < N; i++) {
      const isLast = (i === N - 1) ? 1.0 : 0.0
      const gIdx = (genreIdx && genreIdx[i] != null) ? genreIdx[i] : 0
      const color = genreColorsRgb[gIdx % 10]

      const off = i * STRIDE_FLOATS
      this.bufferData[off] = startFlat[i * 2]
      this.bufferData[off + 1] = startFlat[i * 2 + 1]
      this.bufferData[off + 2] = targetFlat[i * 2]
      this.bufferData[off + 3] = targetFlat[i * 2 + 1]
      this.bufferData[off + 4] = color[0]
      this.bufferData[off + 5] = color[1]
      this.bufferData[off + 6] = color[2]
      this.bufferData[off + 7] = isLast
    }

    // Upload to GPU (no need to bind VAO — just update the VBO data)
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo)
    gl.bufferData(gl.ARRAY_BUFFER, this.bufferData.subarray(0, neededSize), gl.DYNAMIC_DRAW)
  }

  render(progress) {
    if (!this.gl || this.N === 0) return
    const gl = this.gl
    this.clear()

    gl.useProgram(this.program)
    gl.uniform2f(this.uMinStartLoc, this.minStart[0], this.minStart[1])
    gl.uniform2f(this.uRangeStartLoc, this.rangeStart[0], this.rangeStart[1])
    gl.uniform2f(this.uMinTargetLoc, this.minTarget[0], this.minTarget[1])
    gl.uniform2f(this.uRangeTargetLoc, this.rangeTarget[0], this.rangeTarget[1])
    gl.uniform1f(this.uProgressLoc, progress)
    
    gl.uniform2f(this.uResolutionLoc, gl.canvas.width, gl.canvas.height)
    gl.uniform1f(this.uDprLoc, window.devicePixelRatio || 1)

    // VAO binds all vertex attributes in a single call
    gl.bindVertexArray(this.vao)
    gl.drawArrays(gl.POINTS, 0, this.N)
    gl.bindVertexArray(null)
  }

  destroy() {
    if (this.gl) {
      if (this.vao) this.gl.deleteVertexArray(this.vao)
      if (this.program) this.gl.deleteProgram(this.program)
      if (this.vbo) this.gl.deleteBuffer(this.vbo)
    }
  }
}

// ── Animation ──────────────────────────────────────────────

function animateLoop(time) {
  if (!animStartTime) animStartTime = time
  const elapsed = time - animStartTime
  currentProgress = Math.min(elapsed / ANIM_DURATION, 1.0)

  glClean?.render(currentProgress)
  glNoisy?.render(currentProgress)
  glDenoised?.render(currentProgress)

  if (currentProgress < 1.0) {
    animFrameId = requestAnimationFrame(animateLoop)
  } else {
    animFrameId = null
  }
}

// ── Lifecycle ───────────────────────────────────────────────────────

onMounted(() => {
  initWorker()
  glClean = new WebGLUmap(cleanCanvas.value)
  glNoisy = new WebGLUmap(noisyCanvas.value)
  glDenoised = new WebGLUmap(denoisedCanvas.value)
  glClean.clear()
  glNoisy.clear()
  glDenoised.clear()
})

onUnmounted(() => {
  if (worker) { worker.terminate(); worker = null }
  if (animFrameId) cancelAnimationFrame(animFrameId)
  glClean?.destroy()
  glNoisy?.destroy()
  glDenoised?.destroy()
})

let frameCount = 0
watch(() => props.cleanEmbHistory, (newVal) => {
  if (!newVal || newVal.length === 0) {
    frameCount = 0
    return
  }
  frameCount++
  if (newVal.length >= flushInterval && frameCount % flushInterval === 0) {
    requestUMAP()
  }
})

watch(() => props.modelType, () => {
  // Zero out all flat buffers instead of nulling references
  targetCleanFlat.fill(0); targetNoisyFlat.fill(0); targetDenoisedFlat.fill(0)
  currentCleanFlat.fill(0); currentNoisyFlat.fill(0); currentDenoisedFlat.fill(0)
  cachedGenreSlice = null
  pointCount = 0
  currentSliceStart = 0
  hasData.value = false
  currentProgress = 1.0
  animStartTime = 0
  if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null }
  glClean?.clear(); glNoisy?.clear(); glDenoised?.clear()
})
</script>

<style scoped>
.umap-container {
  display: flex;
  gap: 20px;
  margin-top: 24px;
  flex-wrap: wrap;
}

.umap-card {
  flex: 1;
  min-width: 260px;
  background: #ffffff;
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
  display: flex;
  flex-direction: column;
}

.viz-title {
  text-align: center;
  font-size: 1rem;
  font-weight: 700;
  margin-top: 0;
  margin-bottom: 8px;
  color: #111827;
}

.canvas-wrap {
  flex: 1;
  background: #f8fafc;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid rgba(0, 0, 0, 0.05);
  position: relative;
}

.canvas-wrap canvas {
  width: 100%;
  height: 100%;
  display: block;
  object-fit: fill;
}

.empty-overlay {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  color: #6b7280;
  font-size: 13px;
  font-family: system-ui, sans-serif;
  pointer-events: none;
}

@media (max-width: 900px) {
  .umap-container { flex-direction: column; }
}
</style>
