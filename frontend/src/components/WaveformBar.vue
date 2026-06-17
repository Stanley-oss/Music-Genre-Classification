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
  getProgress: Function,
  analyser: Object
})

const emit = defineEmits(['seek'])

const wrapRef = ref(null)
const cvs = ref(null)
let raf
let resizeRafId = null

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

let offscreenLight = null;
let offscreenDark = null;

function renderWaveformToOffscreen(w, h, midY) {
  const dpr = window.devicePixelRatio || 1;
  const cw = w * dpr;
  const ch = h * dpr;
  
  if (!offscreenLight) offscreenLight = document.createElement('canvas');
  if (!offscreenDark) offscreenDark = document.createElement('canvas');
  
  offscreenLight.width = cw; offscreenLight.height = ch;
  offscreenDark.width = cw; offscreenDark.height = ch;
  
  const ctxLight = offscreenLight.getContext('2d', { alpha: false });
  const ctxDark = offscreenDark.getContext('2d', { alpha: false });
  
  // Fill background
  ctxLight.fillStyle = '#f9fafb'; // matching container background
  ctxLight.fillRect(0, 0, cw, ch);
  ctxDark.fillStyle = '#f9fafb';
  ctxDark.fillRect(0, 0, cw, ch);
  
  ctxLight.scale(dpr, dpr);
  ctxDark.scale(dpr, dpr);
  
  ctxLight.fillStyle = '#bfdbfe';
  ctxDark.fillStyle = '#3b82f6';
  
  for (let i = 0; i < w; i++) {
    const min = peaks[i * 2], max = peaks[i * 2 + 1];
    const y = midY + min * midY;
    const barH = Math.max(1, (max - min) * midY);
    ctxLight.fillRect(i, y, 1, barH);
    ctxDark.fillRect(i, y, 1, barH);
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

  const ctx = el.getContext('2d', { alpha: false })
  ctx.save()
  
  ctx.scale(dpr, dpr)
  
  // Only clear if we are not going to fully overwrite with opaque offscreen images
  if (!props.audioData || props.audioData.length === 0) {
    ctx.fillStyle = '#f9fafb'
    ctx.fillRect(0, 0, w, h)
  }

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
      renderWaveformToOffscreen(w, h, midY)
      cachedWidth = w
    }

    const currentP = scrubProgress.value !== null ? scrubProgress.value : (props.getProgress ? props.getProgress() : 0)
    const splitIndex = Math.floor(w * currentP)

    // Draw dark blue (played portion)
    if (splitIndex > 0) {
      ctx.drawImage(offscreenDark, 0, 0, splitIndex * dpr, h * dpr, 0, 0, splitIndex, h)
    }
    
    // Draw light blue (unplayed portion)
    if (splitIndex < w) {
      const remain = w - splitIndex
      ctx.drawImage(offscreenLight, splitIndex * dpr, 0, remain * dpr, h * dpr, splitIndex, 0, remain, h)
    }
    
    // Draw play cursor
    ctx.fillStyle = '#ef4444' // red cursor
    ctx.fillRect(splitIndex, 0, 2, h)
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

function onResize() {
  cachedWidth = 0 // Force peak recomputation on next draw
  if (!props.active && !isDragging) {
    // Not in active rAF loop: schedule a single debounced redraw
    if (resizeRafId) cancelAnimationFrame(resizeRafId)
    resizeRafId = requestAnimationFrame(() => {
      resizeRafId = null
      draw()
    })
  }
  // If active, the running rAF loop will pick up the change on next frame
}

onMounted(() => {
  draw()
  window.addEventListener('resize', onResize)
})

watch(() => props.active, (v) => {
  if (v) draw()
  else cancelAnimationFrame(raf)
})

watch(() => props.audioData, () => {
  cachedWidth = 0
  draw()
})

// 暴露一个方法供外部必要时手动触发重绘
defineExpose({
  redraw: () => { if (!props.active && !isDragging) draw() }
})

onUnmounted(() => {
  cancelAnimationFrame(raf)
  if (resizeRafId) cancelAnimationFrame(resizeRafId)
  window.removeEventListener('resize', onResize)
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