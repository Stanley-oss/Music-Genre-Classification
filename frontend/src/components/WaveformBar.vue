// File: ./frontend/src/components/WaveformBar.vue
<template>
  <div 
    class="wave-wrap" 
    :class="{ 'is-interactive': !!audioData }"
    ref="wrapRef"
    @mousedown="onMouseDown"
    @mousemove="onMouseMove"
    @mouseup="onMouseUp"
    @mouseleave="onMouseUp"
  >
    <canvas ref="cvs"></canvas>
    <div v-if="!active" class="placeholder">Waiting for audio stream…</div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'

const props = defineProps({ 
  active: Boolean,
  audioData: Float32Array,
  progress: Number
})

const emit = defineEmits(['seek', 'scrub', 'scrubStart', 'scrubEnd'])

const wrapRef = ref(null)
const cvs = ref(null)
let raf
let timeOffset = 0

let peaks =[]
let cachedWidth = 0

let isDragging = false
const scrubProgress = ref(null) 

function getMouseProgress(e) {
  const rect = wrapRef.value.getBoundingClientRect()
  let x = e.clientX - rect.left
  return Math.max(0, Math.min(x, rect.width)) / rect.width
}

function onMouseDown(e) {
  if (!props.audioData) return
  isDragging = true
  emit('scrubStart')
  const p = getMouseProgress(e)
  scrubProgress.value = p
  emit('scrub', p)
}

function onMouseMove(e) {
  if (!isDragging) return
  const p = getMouseProgress(e)
  scrubProgress.value = p
  emit('scrub', p)
}

function onMouseUp(e) {
  if (isDragging) {
    isDragging = false
    emit('seek', scrubProgress.value)
    emit('scrubEnd')
    scrubProgress.value = null
  }
}

// ...下方保留原有的 computePeaks、draw、各种 Watch/Lifecycle 原代码...
function computePeaks(data, w) {
  peaks = new Float32Array(w * 2)
  const step = Math.ceil(data.length / w)
  for (let i = 0; i < w; i++) {
    let min = 0, max = 0
    const start = i * step
    const end = Math.min(start + step, data.length)
    for (let j = start; j < end; j++) {
      const val = data[j]
      if (val > max) max = val
      if (val < min) min = val
    }
    peaks[i * 2] = min
    peaks[i * 2 + 1] = max
  }
}

function draw() {
  const el = cvs.value
  if (!el || !el.parentElement) return

  const dpr = window.devicePixelRatio || 1
  const w = el.parentElement.clientWidth
  const h = 100
  
  if (el.width !== w * dpr || el.height !== h * dpr) {
    el.width = w * dpr
    el.height = h * dpr
    cachedWidth = 0
  }

  const ctx = el.getContext('2d')
  ctx.save()
  ctx.scale(dpr, dpr)
  ctx.clearRect(0, 0, w, h)

  if (!props.active && !props.audioData) {
    ctx.fillStyle = '#e5e7eb'
    ctx.fillRect(0, h / 2 - 1, w, 2)
    ctx.restore()
    return
  }

  if (props.audioData && props.audioData.length > 0) {
    if (cachedWidth !== w) {
      computePeaks(props.audioData, w)
      cachedWidth = w
    }

    const midY = h / 2
    const currentP = scrubProgress.value !== null ? scrubProgress.value : (props.progress || 0)
    const splitIndex = Math.floor(w * currentP)

    ctx.fillStyle = '#bfdbfe' 
    for (let i = 0; i < splitIndex; i++) {
      const min = peaks[i * 2]
      const max = peaks[i * 2 + 1]
      const y = midY + min * midY
      const height = Math.max(1, (max - min) * midY)
      ctx.fillRect(i, y, 1, height)
    }

    ctx.fillStyle = '#3b82f6'
    for (let i = splitIndex; i < w; i++) {
      const min = peaks[i * 2]
      const max = peaks[i * 2 + 1]
      const y = midY + min * midY
      const height = Math.max(1, (max - min) * midY)
      ctx.fillRect(i, y, 1, height)
    }
  } else {
    ctx.fillStyle = '#a855f7'
    const barWidth = 4
    const gap = 2
    const bars = Math.floor(w / (barWidth + gap))
    timeOffset += 0.1

    for (let i = 0; i < bars; i++) {
      let noise = Math.sin(i * 0.15 + timeOffset) * Math.cos(i * 0.27 - timeOffset)
      let base = Math.sin(i * 0.05) * 0.5 + 0.5
      let amplitude = Math.abs(noise * base) * (h / 2) * 0.8 + 2
      amplitude += Math.random() * 3

      const x = i * (barWidth + gap)
      const y = h / 2 - amplitude
      
      ctx.beginPath()
      ctx.roundRect(x, y, barWidth, amplitude * 2, 2)
      ctx.fill()
    }
  }

  ctx.restore()

  if (props.active || isDragging) {
    raf = requestAnimationFrame(draw)
  }
}

onMounted(() => {
  draw()
  window.addEventListener('resize', draw)
})

watch(() => props.active, (v) => {
  if (v) draw()
  else cancelAnimationFrame(raf)
})

watch(() => props.audioData, () => {
  cachedWidth = 0
  draw()
})

onUnmounted(() => {
  cancelAnimationFrame(raf)
  window.removeEventListener('resize', draw)
})
</script>

<style scoped>
.wave-wrap {
  position: relative;
  background: transparent;
  width: 100%;
  height: 100px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.wave-wrap.is-interactive {
  cursor: pointer;
}
canvas {
  display: block;
  width: 100%;
  height: 100%;
}
.placeholder {
  position: absolute;
  color: #9ca3af;
  font-size: 0.9rem;
  pointer-events: none;
}
</style>