<template>
  <div class="umap-container">
    <div class="umap-card">
      <h3 class="viz-title">UMAP: Clean Embedding (z)</h3>
      <div class="canvas-wrap">
        <div v-if="!hasData" class="empty-overlay">Collecting embeddings…</div>
        <canvas ref="cleanCanvas" width="360" height="240"></canvas>
      </div>
    </div>
    <div class="umap-card">
      <h3 class="viz-title">UMAP: Noisy Embedding (z<sub>t</sub>)</h3>
      <div class="canvas-wrap">
        <div v-if="!hasData" class="empty-overlay">Collecting embeddings…</div>
        <canvas ref="noisyCanvas" width="360" height="240"></canvas>
      </div>
    </div>
    <div class="umap-card">
      <h3 class="viz-title">UMAP: Denoised Embedding (ẑ)</h3>
      <div class="canvas-wrap">
        <div v-if="!hasData" class="empty-overlay">Collecting embeddings…</div>
        <canvas ref="denoisedCanvas" width="360" height="240"></canvas>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, onMounted, onUnmounted, toRaw } from 'vue'

const genres = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
const genreColors = [
  '#3b82f6', '#64748b', '#14b8a6', '#ec4899', '#ef4444',
  '#eab308', '#f97316', '#06b6d4', '#22c55e', '#8b5cf6'
]

function hexToRgb(hex) {
  return [
    parseInt(hex.slice(1,3), 16)/255,
    parseInt(hex.slice(3,5), 16)/255,
    parseInt(hex.slice(5,7), 16)/255
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

// Target results from UMAP worker
let targetClean2d = null
let targetNoisy2d = null
let targetDenoised2d = null
let cachedGenreSlice = null

// Current animated positions (not reactive for performance)
let currentClean2d = null
let currentNoisy2d = null
let currentDenoised2d = null
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

function initWorker() {
  worker = new Worker(
    new URL('../workers/umap-worker.js', import.meta.url),
    { type: 'module' }
  )
  worker.onmessage = (e) => {
    if (e.data.type === 'result') {
      const targetSliceStart = e.data.sliceStart
      
      // Calculate exact visual positions at the time of interruption
      const visualClean2d = interpolateArray(currentClean2d, targetClean2d, currentProgress)
      const visualNoisy2d = interpolateArray(currentNoisy2d, targetNoisy2d, currentProgress)
      const visualDenoised2d = interpolateArray(currentDenoised2d, targetDenoised2d, currentProgress)
      
      // Align visual positions to the new target indices
      currentClean2d = syncArrays(e.data.clean2d, visualClean2d, targetSliceStart, currentSliceStart)
      currentNoisy2d = syncArrays(e.data.noisy2d, visualNoisy2d, targetSliceStart, currentSliceStart)
      currentDenoised2d = syncArrays(e.data.denoised2d, visualDenoised2d, targetSliceStart, currentSliceStart)
      
      targetClean2d = e.data.clean2d
      targetNoisy2d = e.data.noisy2d
      targetDenoised2d = e.data.denoised2d
      cachedGenreSlice = e.data.genreIndices
      
      currentSliceStart = targetSliceStart
      
      // Upload ONLY once per worker payload
      glClean?.updateBuffers(currentClean2d, targetClean2d, cachedGenreSlice)
      glNoisy?.updateBuffers(currentNoisy2d, targetNoisy2d, cachedGenreSlice)
      glDenoised?.updateBuffers(currentDenoised2d, targetDenoised2d, cachedGenreSlice)
      
      hasData.value = true
      currentProgress = 0.0
      animStartTime = performance.now()
      
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

// ── WebGL Renderer Class ─────────────────────────────────────────────
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

out vec3 v_color;
out float v_isLast;

void main() {
    float easeOut = 1.0 - pow(1.0 - u_progress, 3.0);
    
    vec2 pos = mix(a_start_position, a_target_position, easeOut);
    vec2 current_min = mix(u_min_start, u_min_target, easeOut);
    vec2 current_range = mix(u_range_start, u_range_target, easeOut);
    
    vec2 norm = (pos - current_min) / current_range;
    float padX = 18.0 / 360.0;
    float padY = 18.0 / 240.0;
    
    vec2 screenNorm = vec2(
        mix(padX, 1.0 - padX, norm.x),
        mix(padY, 1.0 - padY, norm.y)
    );
    screenNorm.y = 1.0 - screenNorm.y; // Match Canvas2D orientation

    gl_Position = vec4(screenNorm * 2.0 - 1.0, 0.0, 1.0);
    gl_PointSize = (a_isLast > 0.5) ? 24.0 : 9.0;
    
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
    
    const vs = gl.createShader(gl.VERTEX_SHADER); gl.shaderSource(vs, vsSource); gl.compileShader(vs)
    const fs = gl.createShader(gl.FRAGMENT_SHADER); gl.shaderSource(fs, fsSource); gl.compileShader(fs)
    
    this.program = gl.createProgram()
    gl.attachShader(this.program, vs)
    gl.attachShader(this.program, fs)
    gl.linkProgram(this.program)
    
    this.vbo = gl.createBuffer()
    this.uMinStartLoc = gl.getUniformLocation(this.program, "u_min_start")
    this.uRangeStartLoc = gl.getUniformLocation(this.program, "u_range_start")
    this.uMinTargetLoc = gl.getUniformLocation(this.program, "u_min_target")
    this.uRangeTargetLoc = gl.getUniformLocation(this.program, "u_range_target")
    this.uProgressLoc = gl.getUniformLocation(this.program, "u_progress")
    
    this.aStartLoc = gl.getAttribLocation(this.program, "a_start_position")
    this.aTargetLoc = gl.getAttribLocation(this.program, "a_target_position")
    this.aColLoc = gl.getAttribLocation(this.program, "a_color")
    this.aIsLastLoc = gl.getAttribLocation(this.program, "a_isLast")
    
    gl.enable(gl.BLEND)
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
    
    this.N = 0;
  }
  
  clear() {
    if (!this.gl) return
    const gl = this.gl
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height)
    gl.clearColor(0.972, 0.980, 0.988, 1.0)
    gl.clear(gl.COLOR_BUFFER_BIT)
  }
  
  updateBuffers(start2d, target2d, genreIdx) {
    if (!this.gl) return
    const gl = this.gl
    const N = target2d ? target2d.length : 0
    this.N = N;
    
    if (N === 0) {
      this.clear()
      return
    }
    
    let xMinStart = Infinity, xMaxStart = -Infinity, yMinStart = Infinity, yMaxStart = -Infinity
    let xMinTarget = Infinity, xMaxTarget = -Infinity, yMinTarget = Infinity, yMaxTarget = -Infinity
    
    for (let i = 0; i < N; i++) {
      const sx = start2d[i][0], sy = start2d[i][1]
      if (sx < xMinStart) xMinStart = sx; if (sx > xMaxStart) xMaxStart = sx
      if (sy < yMinStart) yMinStart = sy; if (sy > yMaxStart) yMaxStart = sy
      
      const tx = target2d[i][0], ty = target2d[i][1]
      if (tx < xMinTarget) xMinTarget = tx; if (tx > xMaxTarget) xMaxTarget = tx
      if (ty < yMinTarget) yMinTarget = ty; if (ty > yMaxTarget) yMaxTarget = ty
    }
    
    this.minStart = [xMinStart, yMinStart]
    this.rangeStart = [(xMaxStart - xMinStart) || 1, (yMaxStart - yMinStart) || 1]
    this.minTarget = [xMinTarget, yMinTarget]
    this.rangeTarget = [(xMaxTarget - xMinTarget) || 1, (yMaxTarget - yMinTarget) || 1]
    
    const neededSize = N * 8
    if (!this.bufferData || this.bufferData.length < neededSize) {
      this.bufferData = new Float32Array(neededSize * 2)
    }
    
    for(let i=0; i<N; i++) {
       const isLast = (i === N - 1) ? 1.0 : 0.0
       const gIdx = (genreIdx && genreIdx[i] != null) ? genreIdx[i] : 0
       const color = genreColorsRgb[gIdx % 10]
       
       const off = i * 8
       this.bufferData[off+0] = start2d[i][0]
       this.bufferData[off+1] = start2d[i][1]
       this.bufferData[off+2] = target2d[i][0]
       this.bufferData[off+3] = target2d[i][1]
       this.bufferData[off+4] = color[0]
       this.bufferData[off+5] = color[1]
       this.bufferData[off+6] = color[2]
       this.bufferData[off+7] = isLast
    }
    
    gl.useProgram(this.program)
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
    
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo)
    
    gl.enableVertexAttribArray(this.aStartLoc)
    gl.enableVertexAttribArray(this.aTargetLoc)
    gl.enableVertexAttribArray(this.aColLoc)
    gl.enableVertexAttribArray(this.aIsLastLoc)
    
    gl.vertexAttribPointer(this.aStartLoc, 2, gl.FLOAT, false, 32, 0)
    gl.vertexAttribPointer(this.aTargetLoc, 2, gl.FLOAT, false, 32, 8)
    gl.vertexAttribPointer(this.aColLoc, 3, gl.FLOAT, false, 32, 16)
    gl.vertexAttribPointer(this.aIsLastLoc, 1, gl.FLOAT, false, 32, 28)
    
    gl.drawArrays(gl.POINTS, 0, this.N)
  }
  
  destroy() {
    if (this.gl) {
      this.gl.deleteProgram(this.program)
      this.gl.deleteBuffer(this.vbo)
    }
  }
}

// ── Animation ──────────────────────────────────────────────

function interpolateArray(start2d, target2d, progress) {
  if (!start2d || !target2d) return target2d ? target2d.map(p => [p[0], p[1]]) : null;
  const easeOut = 1.0 - Math.pow(1.0 - progress, 3.0);
  const curr = new Array(target2d.length);
  for (let i = 0; i < target2d.length; i++) {
    const s = start2d[i] || target2d[i];
    const t = target2d[i];
    curr[i] = [
      s[0] + (t[0] - s[0]) * easeOut,
      s[1] + (t[1] - s[1]) * easeOut
    ];
  }
  return curr;
}

function syncArrays(target, current, targetStart, currentStart) {
  if (!target) return null
  if (!current || current.length === 0) return target.map(p => [p[0], p[1]])
  
  const next = new Array(target.length)
  for (let i = 0; i < target.length; i++) {
    const absIdx = targetStart + i
    const oldIdx = absIdx - currentStart
    if (oldIdx >= 0 && oldIdx < current.length) {
      next[i] = current[oldIdx]
    } else {
      next[i] = current.length > 0 ? [current[current.length - 1][0], current[current.length - 1][1]] : [target[i][0], target[i][1]]
    }
  }
  return next
}

function animateLoop(time) {
  if (!animStartTime) animStartTime = time;
  let elapsed = time - animStartTime;
  currentProgress = Math.min(elapsed / ANIM_DURATION, 1.0);
  
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

watch(() => props.cleanEmbHistory.length, (newLen) => {
  if (newLen >= flushInterval && newLen % flushInterval === 0) {
    requestUMAP()
  }
})

watch(() => props.modelType, () => {
  targetClean2d = null; targetNoisy2d = null; targetDenoised2d = null
  currentClean2d = null; currentNoisy2d = null; currentDenoised2d = null
  cachedGenreSlice = null
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
  box-shadow: 0 4px 20px rgba(0,0,0,0.03);
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
  border: 1px solid rgba(0,0,0,0.05);
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
