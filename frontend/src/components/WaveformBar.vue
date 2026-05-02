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
  progress: Number,
  analyser: Object
})

const emit = defineEmits(['seek'])

const wrapRef = ref(null)
const cvs = ref(null)
let raf

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
  const p = getMouseProgress(e)
  scrubProgress.value = p
  emit('seek', p)
  if (!props.active) draw() // 暂停时强制重绘
}

function onMouseMove(e) {
  if (!isDragging) return
  scrubProgress.value = getMouseProgress(e)
  if (!props.active) draw() // 暂停时拖拽强制重绘
}

function onMouseUp(e) {
  if (isDragging) {
    isDragging = false
    emit('seek', scrubProgress.value)
    scrubProgress.value = null
    if (!props.active) draw()
  }
}

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

  if (!props.active && !props.audioData && !props.analyser) {
    ctx.fillStyle = '#e5e7eb'
    ctx.fillRect(0, h / 2 - 1, w, 2)
    ctx.restore()
    return
  }

  const midY = h / 2

  if (props.audioData && props.audioData.length > 0) {
    if (cachedWidth !== w) {
      computePeaks(props.audioData, w)
      cachedWidth = w
    }

    const currentP = scrubProgress.value !== null ? scrubProgress.value : (props.progress || 0)
    const splitIndex = Math.floor(w * currentP)

    ctx.fillStyle = '#bfdbfe' 
    for (let i = 0; i < splitIndex; i++) {
      const min = peaks[i * 2], max = peaks[i * 2 + 1]
      const y = midY + min * midY
      ctx.fillRect(i, y, 1, Math.max(1, (max - min) * midY))
    }

    ctx.fillStyle = '#3b82f6'
    for (let i = splitIndex; i < w; i++) {
      const min = peaks[i * 2], max = peaks[i * 2 + 1]
      const y = midY + min * midY
      ctx.fillRect(i, y, 1, Math.max(1, (max - min) * midY))
    }
  } 
  else if (props.analyser) {
    const bufferLength = props.analyser.frequencyBinCount
    const dataArray = new Uint8Array(bufferLength)
    props.analyser.getByteFrequencyData(dataArray)

    const barWidth = 4
    const gap = 2
    const bars = Math.floor(w / (barWidth + gap))

    const minBin = 1
    const maxBin = Math.floor(bufferLength * 0.45) 
    const logMin = Math.log(minBin)
    const logMax = Math.log(maxBin)

    for (let i = 0; i < bars; i++) {
      const ratio = i / bars
      const bin = Math.floor(Math.exp(logMin + ratio * (logMax - logMin)))
      const value = dataArray[bin] || 0
      const amplitude = Math.max((value / 255) * midY * 0.9, 2)
      const x = i * (barWidth + gap)
      const y = midY - amplitude
      
      ctx.fillStyle = `hsl(${260 + ratio * 60}, 80%, 65%)`
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

// 当外部只改变进度时（比如外部点按或快进），如果是暂停状态也要重绘
watch(() => props.progress, () => {
  if (!props.active && !isDragging) draw()
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