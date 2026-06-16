<template>
  <div class="viz-card">
    <h3 class="viz-title">Mel Spectrogram</h3>
    <div class="canvas-wrapper">
      <div v-if="!data || data.length === 0" class="empty-overlay">Waiting for audio stream…</div>
      <canvas ref="canvasRef" width="400" height="220"></canvas>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, onMounted, onUnmounted } from 'vue'

const props = defineProps({
  data: {
    type: Float32Array,
    default: null
  }
})

const canvasRef = ref(null)
let gl = null
let program = null
let tex = null
let positionBuffer = null

const N_MELS = 128

const vsSource = `#version 300 es
in vec2 a_position;
out vec2 v_texCoord;
void main() {
    v_texCoord = vec2(a_position.x * 0.5 + 0.5, a_position.y * 0.5 + 0.5);
    gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const fsSource = `#version 300 es
precision highp float;
in vec2 v_texCoord;
uniform sampler2D u_tex;
out vec4 outColor;

void main() {
    float val = texture(u_tex, v_texCoord).r;
    float n = clamp((val + 3.0) / 9.0, 0.0, 1.0);
    n = pow(n, 1.5);
    float r = clamp(1.5 - abs(4.0 * n - 3.0), 0.0, 1.0);
    float g = clamp(1.5 - abs(4.0 * n - 2.0), 0.0, 1.0);
    float b = clamp(1.5 - abs(4.0 * n - 1.0), 0.0, 1.0);
    outColor = vec4(r, g, b, 1.0);
}
`;

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
  if (!gl) {
    console.warn('WebGL2 not supported, visualization will not render.')
    return
  }

  // Float texture extension check (WebGL2 usually has it built-in, but good to check)
  gl.getExtension('EXT_color_buffer_float')

  const vertexShader = createShader(gl.VERTEX_SHADER, vsSource)
  const fragmentShader = createShader(gl.FRAGMENT_SHADER, fsSource)

  program = gl.createProgram()
  gl.attachShader(program, vertexShader)
  gl.attachShader(program, fragmentShader)
  gl.linkProgram(program)

  positionBuffer = gl.createBuffer()
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer)
  // Full screen quad
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1,  1, -1,  -1, 1,
    -1,  1,  1, -1,   1, 1
  ]), gl.STATIC_DRAW)

  tex = gl.createTexture()
  gl.bindTexture(gl.TEXTURE_2D, tex)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
}

function draw() {
  if (!gl || !program) return
  if (!props.data || props.data.length === 0 || props.data.length % N_MELS !== 0) {
    gl.clearColor(0.97, 0.98, 0.98, 1.0) // #f8fafc empty state
    gl.clear(gl.COLOR_BUFFER_BIT)
    return
  }

  const frames = props.data.length / N_MELS

  gl.viewport(0, 0, gl.canvas.width, gl.canvas.height)
  gl.useProgram(program)

  // Upload data
  gl.bindTexture(gl.TEXTURE_2D, tex)
  // R32F texture to read Float32Array directly
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, frames, N_MELS, 0, gl.RED, gl.FLOAT, props.data)

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

watch(() => props.data, scheduleDraw)

onMounted(() => {
  initWebGL()
  draw()
})

onUnmounted(() => {
  cancelAnimationFrame(rafId)
  if (gl) {
    gl.deleteProgram(program)
    gl.deleteTexture(tex)
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
}
.viz-title {
  text-align: center;
  font-size: 1.1rem;
  font-weight: 700;
  margin-top: 0;
  margin-bottom: 16px;
  color: #111827;
}
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
  object-fit: fill;
  display: block;
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
