<!-- File: ./frontend/src/App.vue -->
<template>
  <div class="app-container">
    <header class="header">
      <h1>Music Genre Finder</h1>
      <p>Find Genre of any audio for free</p>
    </header>

    <div class="upload-container">
      <p class="upload-text">
        Drag and drop your audio files anywhere on the screen<br>
        or use the designated upload area below.
      </p>

      <div
        class="upload-box"
        @dragover.prevent
        @drop.prevent="onDrop"
        @click="$refs.fileInput.click()"
      >
        <input ref="fileInput" type="file" accept="audio/*" hidden @change="onFileSelect" />
        <span>Click here to upload files</span>
      </div>

      <div v-if="currentFile" class="file-selected">
        Track: <strong>{{ currentFile.name }}</strong>
      </div>

      <div class="controls">
        <select v-model="selectedModel" :disabled="isRunning">
          <option value="onnx">Browser ONNX (WASM) — ResNet18</option>
          <option value="backend">Server GPU — ResNet18</option>
        </select>

        <button :disabled="!modelReady || isRunning" @click="startMic" class="btn btn-primary">
          🎤 Microphone
        </button>
        <button :disabled="!modelReady || !currentFile || isRunning" @click="startFile" class="btn btn-primary">
          ▶️ Play File
        </button>
        <button :disabled="!isRunning" @click="stop" class="btn btn-danger">
          ⏹ Stop
        </button>
      </div>

      <div class="status" :class="statusClass">
        <span v-if="modelLoading" class="loader"></span>
        {{ statusText }}
      </div>
    </div>

    <div class="content-grid">
      <!-- 新增事件，精准反馈拖拽悬停行为 -->
      <WaveformBar
        :active="isRunning"
        :audioData="currentAudioData"
        :progress="playbackProgress"
        @seek="onSeek"
        @scrub="onScrub"
        @scrubStart="isScrubbing = true"
        @scrubEnd="isScrubbing = false"
      />

      <div class="chart-container">
        <!-- 传入新增 playhead，即图表红线标记秒数 -->
        <GenreChart
          :genres="engine?.genres ||[]"
          :history="patchHistory"
          :time="currentTime"
          :playhead="playbackProgress * trackDuration"
          :duration="trackDuration"
        />
      </div>

      <TopGenres :top5="finalTop5.length ? finalTop5 : currentTop5" />
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, shallowRef, watch, onUnmounted } from 'vue'
import { OnnxEngine } from './engine/onnx-engine'
import { BackendEngine } from './engine/backend-engine'
import { MicrophoneCapture, FileCapture } from './audio/capture'
import { AudioRingBuffer } from './audio/ring-buffer'
import WaveformBar from './components/WaveformBar.vue'
import GenreChart from './components/GenreChart.vue'
import TopGenres from './components/TopGenres.vue'

const selectedModel = ref('onnx')
const engine = shallowRef(null)
const ringBuffer = shallowRef(null)

const modelReady = ref(false)
const modelLoading = ref(false)
const isRunning = ref(false)
const currentFile = ref(null)

const lastFile = ref(null)         // 记录上一首播放的文件
const savedTimeSec = ref(0)        // 记录断点时间

const currentAudioData = ref(null)
const playbackProgress = ref(0)
const trackDuration = ref(0) 
const isScrubbing = ref(false)     // 记录用户是否正按住鼠标拖拽

const patchHistory = ref([])
const currentTop5 = ref([])
const finalTop5 = ref([])
const currentTime = ref(0)

let capture = null
let inferTimer = null
let progressRaf = null
let isInferencing = false

function createEngine(type) {
  if (type === 'onnx') return new OnnxEngine()
  if (type === 'backend') return new BackendEngine()
  throw new Error('Unknown engine type')
}

function resetRingBuffer() {
  if (!engine.value) return
  ringBuffer.value = new AudioRingBuffer(
    engine.value.sampleRate,
    engine.value.patchSamples
  )
}

async function bootEngine() {
  if (engine.value) {
    engine.value.dispose?.()
    engine.value = null
  }
  modelReady.value = false
  const type = selectedModel.value
  modelLoading.value = true
  try {
    const newEngine = createEngine(type)
    await newEngine.init()
    engine.value = newEngine
    resetRingBuffer()
    modelReady.value = true
  } catch (e) {
    console.error(`[Engine] Failed to init ${type}:`, e)
    modelReady.value = false
  } finally {
    modelLoading.value = false
  }
}

watch(selectedModel, () => {
  if (isRunning.value) return
  bootEngine()
})

const statusText = computed(() => {
  if (modelLoading.value) return 'Initializing inference engine…'
  if (!modelReady.value) return 'Engine failed to load. Check console.'
  if (isRunning.value) {
    return engine.value?.mode === 'local'
      ? 'Inferencing locally in browser WASM…'
      : 'Streaming audio to server GPU…'
  }
  return 'Ready. Select an audio source or use Microphone.'
})

const statusClass = computed(() => ({
  running: isRunning.value,
  idle: !isRunning.value && modelReady.value,
  error: !modelReady.value
}))

function onDrop(e) {
  const f = e.dataTransfer.files[0]
  if (f) handleFile(f)
}
function onFileSelect(e) {
  const f = e.target.files[0]
  if (f) handleFile(f)
}
function handleFile(file) {
  currentFile.value = file
}

function onSeek(percentage) {
  const targetSec = percentage * trackDuration.value
  if (capture && capture.isActive && typeof capture.seek === 'function') {
    capture.seek(percentage)
    ringBuffer.value?.clear()
  } else if (trackDuration.value > 0) {
    savedTimeSec.value = targetSec
    playbackProgress.value = percentage
  }
}

// 供拖拽悬停时使用，仅视觉效果变动，不发请求
function onScrub(percentage) {
  if (trackDuration.value > 0) {
    playbackProgress.value = percentage
  }
}

function loopProgress() {
  if (capture && capture.isActive && capture instanceof FileCapture && !isScrubbing.value) {
    playbackProgress.value = capture.currentTime / capture.duration
  }
  progressRaf = requestAnimationFrame(loopProgress)
}

onMounted(() => {
  bootEngine()
  loopProgress()
})

onUnmounted(() => {
  cancelAnimationFrame(progressRaf)
})

async function inferenceLoop() {
  if (!capture?.isActive || !engine.value) return
  if (isInferencing) return

  isInferencing = true
  const snapshot = ringBuffer.value.getSnapshot()
  
  const exactTime = (capture && capture.duration) 
    ? capture.currentTime 
    : patchHistory.value.length * 0.5;

  // 【核心覆写逻辑】根据精确时间构建 0.5秒跨度的“时间桶”
  const roundedT = Math.round(exactTime * 2) / 2;

  try {
    const probs = await engine.value.predict(snapshot, engine.value.sampleRate)
    const probsArr = Array.from(probs)

    // 如果该时间点已有数据（比如用户拖回重播），直接覆写原点位，解决重影连线！
    const existingIdx = patchHistory.value.findIndex(h => h.t === roundedT);
    if (existingIdx !== -1) {
      patchHistory.value[existingIdx] = { t: roundedT, probs: probsArr };
    } else {
      patchHistory.value.push({ t: roundedT, probs: probsArr });
    }

    currentTime.value = exactTime

    // 对所有处理过的不重复时间点的数据求均值
    const genresLen = engine.value.genres.length;
    const mean = new Float32Array(genresLen);
    for (const h of patchHistory.value) {
      for (let i = 0; i < genresLen; i++) mean[i] += h.probs[i];
    }
    const historyLen = patchHistory.value.length;
    const indexed =[];
    for (let i = 0; i < genresLen; i++) {
      indexed.push({
        genre: engine.value.genres[i],
        probability: mean[i] / historyLen
      });
    }

    indexed.sort((a, b) => b.probability - a.probability);
    currentTop5.value = indexed.slice(0, 5);
  } catch (e) {
    console.error('Inference error:', e)
  } finally {
    isInferencing = false
  }
}

function startMic() {
  reset(true) // 麦克风永远是全新会话
  lastFile.value = null 
  capture = new MicrophoneCapture()
  capture.start((chunk, sr) => ringBuffer.value.append(chunk, sr))
  isRunning.value = true
  inferTimer = setInterval(inferenceLoop, 500)
}

async function startFile() {
  if (!currentFile.value) return
  
  // 判断是否为同一首歌，决定是否保留历史图表
  const isNewFile = currentFile.value !== lastFile.value
  reset(isNewFile)
  if (isNewFile) {
    lastFile.value = currentFile.value
  } else {
    // 若即将到结尾才重启，重头开始放
    if (savedTimeSec.value >= trackDuration.value - 0.5) savedTimeSec.value = 0;
  }

  capture = new FileCapture()
  try {
    await capture.start(
      currentFile.value,
      (chunk, sr) => ringBuffer.value.append(chunk, sr),
      (fullData) => { 
        if (isNewFile) {
          currentAudioData.value = fullData 
          trackDuration.value = capture.duration 
        }
      },
      savedTimeSec.value // 从指定记忆断点处开始加载播放
    )
    isRunning.value = true
    inferTimer = setInterval(() => {
      inferenceLoop()
      if (!capture.isActive) stop()
    }, 500)
  } catch (e) {
    console.error('Failed to play file:', e)
  }
}

function reset(full = true) {
  ringBuffer.value?.clear()
  if (full) {
    patchHistory.value =[]
    currentTop5.value = []
    finalTop5.value =[]
    currentTime.value = 0
    playbackProgress.value = 0
    trackDuration.value = 0
    currentAudioData.value = null
    savedTimeSec.value = 0
  }
}

async function stop() {
  // 记录当下断点，以便下次重新 Play 时接续
  if (capture && capture instanceof FileCapture) {
    savedTimeSec.value = capture.currentTime
  }

  capture?.stop()
  clearInterval(inferTimer)
  isRunning.value = false

  if (patchHistory.value.length === 0 || !engine.value) return

  const probs = patchHistory.value.map(h => new Float32Array(h.probs))
  try {
    const result = await engine.value.finalize(probs)
    finalTop5.value = result.top5
  } catch (e) {
    console.error('Finalize error:', e)
  }
}
</script>

<style>
body {
  margin: 0;
  background-color: #f9fafb;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}
.app-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 20px;
  color: #333;
}
.header {
  text-align: center;
  margin-bottom: 40px;
}
.header h1 {
  font-size: 2.5rem;
  margin: 0;
  background: -webkit-linear-gradient(45deg, #6b21a8, #3b82f6);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 800;
}
.header p {
  color: #6b7280;
  font-size: 1.1rem;
  margin-top: 8px;
}
.upload-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-bottom: 40px;
}
.upload-text {
  font-weight: 600;
  color: #374151;
  text-align: center;
  margin-bottom: 16px;
  line-height: 1.5;
}
.upload-box {
  border: 2px dashed #60a5fa;
  background-color: #eff6ff;
  border-radius: 12px;
  padding: 30px;
  width: 100%;
  max-width: 500px;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s ease;
}
.upload-box:hover {
  background-color: #dbeafe;
  border-color: #3b82f6;
}
.upload-box span {
  color: #2563eb;
  font-weight: 600;
}
.controls {
  margin-top: 20px;
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
  justify-content: center;
}
.controls select {
  padding: 10px 16px;
  border-radius: 8px;
  border: 1px solid #d1d5db;
  outline: none;
  font-size: 0.95rem;
}
.btn {
  padding: 10px 20px;
  border-radius: 8px;
  border: none;
  font-weight: 600;
  cursor: pointer;
  color: #fff;
  transition: opacity 0.2s;
}
.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.btn-primary {
  background-color: #3b82f6;
}
.btn-danger {
  background-color: #ef4444;
}
.file-selected {
  margin-top: 12px;
  font-size: 0.9rem;
  color: #4b5563;
  background: #f3f4f6;
  padding: 6px 16px;
  border-radius: 16px;
}
.content-grid {
  display: flex;
  flex-direction: column;
  gap: 30px;
}
.chart-container {
  background: #fff;
  border-radius: 16px;
  padding: 20px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.03);
}
.status {
  text-align: center;
  margin-top: 16px;
  font-size: 0.9rem;
  color: #6b7280;
  min-height: 1.4em;
}
.status.running { color: #10b981; font-weight: 700; }
.status.error { color: #ef4444; }
.loader {
  display: inline-block;
  width: 14px;
  height: 14px;
  border: 2px solid #ccc;
  border-top-color: #4f46e5;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  vertical-align: middle;
  margin-right: 6px;
}
@keyframes spin { to { transform: rotate(360deg); } }
</style>