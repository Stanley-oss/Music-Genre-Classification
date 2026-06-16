<template>
  <div class="viz-card">
    <h3 class="viz-title">Feature Maps</h3>

    <!-- Legend -->
    <div class="legend">
      <span class="legend-item">
        <span class="dot dot-green"></span>
        Freq Branch (Timbre)
      </span>
      <span class="legend-item">
        <span class="dot dot-red"></span>
        Time Branch (Rhythm)
      </span>
    </div>

    <div class="canvas-wrapper">
      <div v-if="!mel || mel.length === 0" class="empty-overlay">Waiting for audio stream…</div>
      <canvas ref="canvasRef" width="400" height="220"></canvas>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, onMounted, onUnmounted } from 'vue'

const props = defineProps({
  mel:          { type: Float32Array, default: null },
  shallowMap:   { type: Float32Array, default: null },
  deepMap:      { type: Float32Array, default: null },
  shallowShape: { type: Array, default: null },
  deepShape:    { type: Array, default: null }
})

const canvasRef = ref(null)
const N_MELS = 128

let gl = null
let program = null
let positionBuffer = null
let texMel = null
let texShallow = null
let texDeep = null

const vsSource = `#version 300 es
in vec2 a_position;
out vec2 v_texCoord;
void main() {
    v_texCoord = vec2(a_position.x * 0.5 + 0.5, a_position.y * 0.5 + 0.5);
    gl_Position = vec4(a_position, 0.0, 1.0);
}
`

const fsSource = `#version 300 es
precision highp float;
in vec2 v_texCoord;

uniform sampler2D u_mel;
uniform sampler2D u_shallow;
uniform sampler2D u_deep;

uniform bool u_hasShallow;
uniform bool u_hasDeep;

out vec4 outColor;

void main() {
    float melVal = texture(u_mel, v_texCoord).r;
    
    // Mel grayscale base
    float n = clamp((melVal + 3.0) / 9.0, 0.0, 1.0);
    n = pow(n, 1.2);
    float r = n * 0.85;
    float g = n * 0.95;
    float b = n;

    // Freq branch overlay (neon green)
    if (u_hasShallow) {
        float act = texture(u_shallow, v_texCoord).r;
        if (act > 0.25) {
            float a = 0.85 * act;
            float inv = 1.0 - a;
            r = r * inv;
            g = g * inv + 1.0 * a;
            b = b * inv + 0.47 * a; // 120/255
        }
    }

    // Time branch overlay (hot red/orange)
    if (u_hasDeep) {
        float act = texture(u_deep, v_texCoord).r;
        if (act > 0.3) {
            float a = 0.9 * act;
            float inv = 1.0 - a;
            r = r * inv + 1.0 * a;
            g = g * inv + 0.31 * a; // 80/255
            b = b * inv + 0.08 * a; // 20/255
        }
    }

    outColor = vec4(r, g, b, 1.0);
}
`

function createShader(type, source) {
  const shader = gl.createShader(type)
  gl.shaderSource(shader, source)
  gl.compileShader(shader)
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error(gl.getShaderInfoLog(shader))
    gl.deleteShader(shader)
    return null
  }
  return shader
}

function initWebGL() {
  const canvas = canvasRef.value
  gl = canvas.getContext('webgl2', { antialias: false })
  if (!gl) return

  gl.getExtension('EXT_color_buffer_float')

  const vertexShader = createShader(gl.VERTEX_SHADER, vsSource)
  const fragmentShader = createShader(gl.FRAGMENT_SHADER, fsSource)

  program = gl.createProgram()
  gl.attachShader(program, vertexShader)
  gl.attachShader(program, fragmentShader)
  gl.linkProgram(program)

  positionBuffer = gl.createBuffer()
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer)
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1,  1, -1,  -1, 1,
    -1,  1,  1, -1,   1, 1
  ]), gl.STATIC_DRAW)

  const setupTex = () => {
    const t = gl.createTexture()
    gl.bindTexture(gl.TEXTURE_2D, t)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    return t
  }

  texMel = setupTex()
  texShallow = setupTex()
  texDeep = setupTex()
}

let shallowNormBuffer = null;
let deepNormBuffer = null;

function normalizeMap(flatMap, isDeep) {
  let out;
  if (isDeep) {
    if (!deepNormBuffer || deepNormBuffer.length < flatMap.length) deepNormBuffer = new Float32Array(flatMap.length);
    out = deepNormBuffer;
  } else {
    if (!shallowNormBuffer || shallowNormBuffer.length < flatMap.length) shallowNormBuffer = new Float32Array(flatMap.length);
    out = shallowNormBuffer;
  }
  
  let mn = Infinity, mx = -Infinity
  for (let i = 0; i < flatMap.length; i++) {
    const v = flatMap[i]; if (v < mn) mn = v; if (v > mx) mx = v
  }
  const range = mx - mn || 1
  for (let i = 0; i < flatMap.length; i++) out[i] = (flatMap[i] - mn) / range
  return out.subarray(0, flatMap.length)
}

function draw() {
  if (!gl || !program) return
  if (!props.mel || props.mel.length === 0 || props.mel.length % N_MELS !== 0) {
    gl.clearColor(0.97, 0.98, 0.98, 1.0)
    gl.clear(gl.COLOR_BUFFER_BIT)
    return
  }

  const frames = props.mel.length / N_MELS

  gl.viewport(0, 0, gl.canvas.width, gl.canvas.height)
  gl.useProgram(program)

  // 1. Upload Mel
  gl.activeTexture(gl.TEXTURE0)
  gl.bindTexture(gl.TEXTURE_2D, texMel)
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, frames, N_MELS, 0, gl.RED, gl.FLOAT, props.mel)
  gl.uniform1i(gl.getUniformLocation(program, "u_mel"), 0)

  // 2. Upload Shallow
  let hasShallow = false
  if (props.shallowMap?.length > 0 && props.shallowShape?.length >= 2) {
    const freq = props.shallowShape[props.shallowShape.length - 2]
    const time = props.shallowShape[props.shallowShape.length - 1]
    const norm = normalizeMap(props.shallowMap, false)
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, texShallow)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, time, freq, 0, gl.RED, gl.FLOAT, norm)
    gl.uniform1i(gl.getUniformLocation(program, "u_shallow"), 1)
    hasShallow = true
  }
  gl.uniform1i(gl.getUniformLocation(program, "u_hasShallow"), hasShallow ? 1 : 0)

  // 3. Upload Deep
  let hasDeep = false
  if (props.deepMap?.length > 0 && props.deepShape?.length >= 2) {
    const freq = props.deepShape[props.deepShape.length - 2]
    const time = props.deepShape[props.deepShape.length - 1]
    const norm = normalizeMap(props.deepMap, true)
    gl.activeTexture(gl.TEXTURE2)
    gl.bindTexture(gl.TEXTURE_2D, texDeep)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, time, freq, 0, gl.RED, gl.FLOAT, norm)
    gl.uniform1i(gl.getUniformLocation(program, "u_deep"), 2)
    hasDeep = true
  }
  gl.uniform1i(gl.getUniformLocation(program, "u_hasDeep"), hasDeep ? 1 : 0)

  // Draw quad
  const posLoc = gl.getAttribLocation(program, "a_position")
  gl.enableVertexAttribArray(posLoc)
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer)
  gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0)
  gl.drawArrays(gl.TRIANGLES, 0, 6)
}

let rafId = 0
function scheduleDraw() {
  cancelAnimationFrame(rafId)
  rafId = requestAnimationFrame(draw)
}

watch(
  () => [props.mel, props.shallowMap, props.deepMap, props.shallowShape, props.deepShape],
  scheduleDraw,
  { deep: true }
)

onMounted(() => {
  initWebGL()
  draw()
})

onUnmounted(() => {
  cancelAnimationFrame(rafId)
  if (gl) {
    gl.deleteProgram(program)
    gl.deleteTexture(texMel)
    gl.deleteTexture(texShallow)
    gl.deleteTexture(texDeep)
    gl.deleteBuffer(positionBuffer)
  }
})
</script>

<style scoped>
.viz-card {
  background: #ffffff;
  border-radius: 16px;
  padding: 20px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.03);
  display: flex;
  flex-direction: column;
  height: 100%;
  width: 100%;
}

.viz-title {
  text-align: center;
  font-size: 1.1rem;
  font-weight: 700;
  margin-top: 0;
  margin-bottom: 10px;
  color: #111827;
}

.legend {
  display: flex;
  justify-content: center;
  gap: 20px;
  margin-bottom: 14px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.85rem;
  color: #4b5563;
  font-weight: 500;
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}
.dot-green { background: #00ff78; box-shadow: 0 0 6px rgba(0, 255, 120, 0.6); }
.dot-red { background: #ff5014; box-shadow: 0 0 6px rgba(255, 80, 20, 0.6); }

.canvas-wrapper {
  flex: 1;
  display: flex;
  align-items: stretch;
  justify-content: stretch;
  background: #f3f4f6;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid rgba(0, 0, 0, 0.05);
  position: relative;
}

canvas {
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
</style>
