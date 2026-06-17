<template>
  <div class="app-container">
    <header class="header">
      <h1>Music Genre Finder</h1>
      <p class="upload-text">
        Drag and drop your audio files anywhere on the screen<br>
        or use the designated upload area below.
      </p>
    </header>

    <div class="upload-container">
      <div class="upload-box" @dragover.prevent @drop.prevent="onDrop" @click="onUploadClick">
        <input ref="fileInput" type="file" accept="audio/*" hidden @change="onFileSelect" />
        <span>Click here to upload files</span>
      </div>

      <div v-if="currentFile" class="file-selected">
        Track: <strong>{{ currentFile.name }}</strong>
      </div>

      <div class="controls">
        <select v-model="selectedModel" :disabled="isRunning || isDecoding">
          <option value="cnn">CNN</option>
          <option value="lstm">LSTM</option>
          <option value="resnet">ResNet</option>
        </select>

        <button :disabled="!modelReady || isRunning || isDecoding" @click="startMic" class="btn btn-primary">
          🎤 Microphone
        </button>
        <button :disabled="!modelReady || !currentFile || isRunning || isDecoding" @click="startFile" class="btn btn-primary">
          ▶️ Play
        </button>
        <button :disabled="!isRunning" @click="stop" class="btn btn-danger">
          ⏹ Pause
        </button>
      </div>

      <div class="status" :class="statusClass">
        <span v-if="modelLoading || isDecoding" class="loader"></span>
        {{ statusText }}
      </div>
    </div>

    <div class="content-grid">
      <WaveformBar
        :active="isRunning"
        :audioData="currentAudioData"
        :getProgress="() => (capture && capture.duration) ? capture.currentTime / capture.duration : 0"
        :analyser="micAnalyser"
        @seek="onSeek"
      />

      <div class="chart-container">
        <GenreChart
          :genres="engine?.genres ||[]"
          :history="patchHistory"
          :time="currentTime"
          :duration="audioDuration"
        />
      </div>

      <!-- Real-time visualizations row -->
      <div class="dashboard-row">
        <div class="dash-col dash-col-mel">
          <MelSpectrogram :data="currentMel" />
        </div>
        <div class="dash-col dash-col-act">
          <CnnFeatureViz 
            :mel="currentMel" :shallowMap="freqMap" :deepMap="timeMap"
            :shallowShape="freqShape" :deepShape="timeShape" />
        </div>
        <div class="dash-col dash-col-top5">
          <TopGenres :top5="finalTop5.length ? finalTop5 : currentTop5" />
        </div>
      </div>

      <!-- UMAP Plot Row -->
      <UmapViz 
        :modelType="selectedModel"
        :cleanEmbHistory="cleanEmbHistory"
        :noisyEmbHistory="noisyEmbHistory"
        :denoisedEmbHistory="denoisedEmbHistory"
        :genreIndexHistory="genreIndexHistory"
      />
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
import MelSpectrogram from './components/MelSpectrogram.vue'
import CnnFeatureViz from './components/CnnFeatureViz.vue'
import UmapViz from './components/UmapViz.vue'

const selectedModel = ref('cnn')
const engine = shallowRef(null)
const ringBuffer = shallowRef(null)

const modelReady = ref(false)
const modelLoading = ref(false)
const isRunning = ref(false)
const isDecoding = ref(false) // 增加解码中的状态
const currentFile = shallowRef(null)
const fileInput = ref(null)

const currentAudioData = shallowRef(null)
const audioDuration = ref(0) 
const micAnalyser = shallowRef(null)

const patchHistory = shallowRef([])
const currentTop5 = shallowRef([])
const finalTop5 = shallowRef([])
const currentTime = ref(0)
const currentMel = shallowRef(null)
const freqMap = shallowRef(null)
const timeMap = shallowRef(null)
const freqShape = shallowRef(null)
const timeShape = shallowRef(null)
const currentCleanProbs = shallowRef(null)
const currentNoisyProbs = shallowRef(null)
const currentDenoisedProbs = shallowRef(null)
const cleanEmbHistory = shallowRef([])
const noisyEmbHistory = shallowRef([])
const denoisedEmbHistory = shallowRef([])
const genreIndexHistory = shallowRef([])

let capture = null
let inferTimer = null
let isInferencing = false

function createEngine(type) {
  if (['cnn', 'lstm', 'resnet'].includes(type)) return new OnnxEngine(type)
  if (type === 'backend') return new BackendEngine()
  throw new Error('Unknown engine type')
}

function resetRingBuffer() {
  if (!engine.value) return
  ringBuffer.value = new AudioRingBuffer(engine.value.sampleRate, engine.value.patchSamples)
}

async function bootEngine() {
  if (engine.value) { engine.value.dispose?.(); engine.value = null }
  modelReady.value = false; modelLoading.value = true
  try {
    const newEngine = createEngine(selectedModel.value)
    await newEngine.init()
    engine.value = newEngine
    resetRingBuffer()
    modelReady.value = true
  } catch (e) {
    console.error('bootEngine Error:', e)
    modelReady.value = false
  } finally {
    modelLoading.value = false
  }
}

// 监听下拉框模型切换
watch(selectedModel, () => { 
  if (!isRunning.value) {
    bootEngine()
    
    // 切换模型时，清空上一个模型的全部推理记忆
    // 但故意保留 currentAudioData、currentTime 和 playbackProgress
    // 这样波形图和播放进度能保持原样，只是排行榜和折线图重新开始记录
    patchHistory.value = []
    currentProbsSum = null
    currentProbsCount = 0
    currentTop5.value = []
    finalTop5.value = []
    currentMel.value = null
    freqMap.value = null; timeMap.value = null;
    currentCleanProbs.value = null; currentNoisyProbs.value = null; currentDenoisedProbs.value = null;
  }
})

const statusText = computed(() => {
  if (modelLoading.value) return 'Initializing engine…'
  if (isDecoding.value) return 'Decoding full audio track (this may take a few seconds)…'
  if (!modelReady.value) return 'Engine failed to load.'
  if (isRunning.value) return engine.value?.mode === 'local' ? 'Inferencing locally…' : 'Streaming to GPU…'
  return 'Ready. Select an audio source or use Microphone.'
})

const statusClass = computed(() => ({
  running: isRunning.value,
  idle: !isRunning.value && modelReady.value && !isDecoding.value,
  error: !modelReady.value
}))

function onUploadClick() {
  if (isRunning.value) stop()
  fileInput.value?.click()
}
function onDrop(e) { const f = e.dataTransfer.files[0]; if (f) handleFile(f) }
function onFileSelect(e) { const f = e.target.files[0]; if (f) handleFile(f); e.target.value = null }
function handleFile(file) {
  if (isRunning.value) stop()
  if (capture) { capture.stop(); capture = null }
  currentFile.value = file; reset() 
}

function onSeek(percentage) {
  if (capture && typeof capture.seek === 'function') {
    const timeSec = percentage * capture.duration
    capture.seek(percentage)
    ringBuffer.value?.clear()
    currentTime.value = timeSec
  }
}

let currentProbsSum = null
let currentProbsCount = 0

function recalculateTop5() {
  if (!engine.value || currentProbsCount === 0 || !currentProbsSum) { currentTop5.value = []; return }
  const genresLen = engine.value.genres.length
  const indexed = []
  for (let i = 0; i < genresLen; i++) {
    indexed.push({ genre: engine.value.genres[i], probability: currentProbsSum[i] / currentProbsCount })
  }
  indexed.sort((a, b) => b.probability - a.probability)
  currentTop5.value = indexed.slice(0, 5)
}

onMounted(() => { bootEngine() })
onUnmounted(() => {
  if (inferTimer) clearInterval(inferTimer)
  if (capture) { capture.stop(); capture = null }
  if (engine.value && typeof engine.value.dispose === 'function') engine.value.dispose()
})

async function inferenceLoop() {
  if (!capture?.isActive || !engine.value) return
  if (isInferencing) return
  isInferencing = true
  const snapshot = ringBuffer.value.getSnapshot()
  
  let rawTime = capture instanceof FileCapture ? capture.currentTime : patchHistory.value.length * 0.5
  let quantizedT = Math.round(rawTime * 2) / 2

  try {
    const result = await engine.value.predict(snapshot, engine.value.sampleRate)
    const probsArr = Array.from(result.probs)
    
    // Update visualizations
    if (result.mel && result.mel.length > 0) currentMel.value = result.mel
    
    if (result.freqMap && result.timeMap) {
      freqMap.value = result.freqMap
      timeMap.value = result.timeMap
      freqShape.value = result.freqShape
      timeShape.value = result.timeShape
    }

    if (result.cleanProbs) {
      currentCleanProbs.value = result.cleanProbs
      currentNoisyProbs.value = result.noisyProbs
      currentDenoisedProbs.value = result.denoisedProbs
    }

    if (result.cleanEmb) {
      const MAX_EMB = 64 // Match UMAP worker's maxPts cap
      const cleanArr = [...cleanEmbHistory.value, Array.from(result.cleanEmb)]
      const noisyArr = [...noisyEmbHistory.value, Array.from(result.noisyEmb)]
      const denoisedArr = [...denoisedEmbHistory.value, Array.from(result.denoisedEmb)]
      cleanEmbHistory.value = cleanArr.length > MAX_EMB ? cleanArr.slice(-MAX_EMB) : cleanArr
      noisyEmbHistory.value = noisyArr.length > MAX_EMB ? noisyArr.slice(-MAX_EMB) : noisyArr
      denoisedEmbHistory.value = denoisedArr.length > MAX_EMB ? denoisedArr.slice(-MAX_EMB) : denoisedArr

      const cp = result.cleanProbs || result.probs
      let maxIdx = 0
      for (let i = 1; i < cp.length; i++) { if (cp[i] > cp[maxIdx]) maxIdx = i }
      const genreArr = [...genreIndexHistory.value, maxIdx]
      genreIndexHistory.value = genreArr.length > MAX_EMB ? genreArr.slice(-MAX_EMB) : genreArr
    }
    
    let existingIdx = patchHistory.value.findIndex(h => h.t === quantizedT)
    if (!currentProbsSum && engine.value) currentProbsSum = new Float32Array(engine.value.genres.length)
    
    if (existingIdx !== -1) {
      const oldProbs = patchHistory.value[existingIdx].probs
      for (let i = 0; i < probsArr.length; i++) {
        currentProbsSum[i] += probsArr[i] - oldProbs[i]
      }
      patchHistory.value[existingIdx] = { t: quantizedT, probs: probsArr }
      patchHistory.value = [...patchHistory.value] // Trigger reactivity
    } else {
      for (let i = 0; i < probsArr.length; i++) {
        currentProbsSum[i] += probsArr[i]
      }
      currentProbsCount++
      const h = patchHistory.value
      if (h.length === 0 || h[h.length - 1].t < quantizedT) {
        patchHistory.value = [...h, { t: quantizedT, probs: probsArr }]
      } else {
        const newHistory = [...h, { t: quantizedT, probs: probsArr }]
        newHistory.sort((a, b) => a.t - b.t)
        patchHistory.value = newHistory
      }
    }
    
    currentTime.value = rawTime 
    recalculateTop5()
  } finally {
    isInferencing = false
  }
}

async function startMic() {
  reset()
  capture = new MicrophoneCapture()
  try {
    await capture.start((chunk, sr) => ringBuffer.value.append(chunk, sr))
    micAnalyser.value = capture.analyser
    isRunning.value = true
    inferTimer = setInterval(inferenceLoop, 500)
  } catch (e) {
    console.error('Mic Error:', e)
  }
}

async function startFile() {
  if (!currentFile.value) return

  if (capture && capture instanceof FileCapture && capture.buffer) {
    finalTop5.value = [] // 解除排行榜锁定
    capture.resume(); isRunning.value = true
    inferTimer = setInterval(() => { inferenceLoop(); if (!capture.isActive) stop() }, 500)
    return
  }

  reset()
  capture = new FileCapture()
  
  try {
    isDecoding.value = true // 显示解码加载中状态
    await capture.start(currentFile.value, (chunk, sr) => ringBuffer.value.append(chunk, sr),
      (fullData, duration) => { 
        currentAudioData.value = fullData
        audioDuration.value = duration 
      }
    )
    isRunning.value = true
    inferTimer = setInterval(() => { inferenceLoop(); if (!capture.isActive) stop() }, 500)
  } catch (e) {
    console.error('File Error:', e)
  } finally {
    isDecoding.value = false
  }
}

function reset() {
  ringBuffer.value?.clear()
  patchHistory.value = []; currentTop5.value = []; finalTop5.value = []
  currentProbsSum = null; currentProbsCount = 0
  currentTime.value = 0; audioDuration.value = 0
  currentAudioData.value = null; micAnalyser.value = null
  currentMel.value = null; freqMap.value = null; timeMap.value = null
  currentCleanProbs.value = null; currentNoisyProbs.value = null; currentDenoisedProbs.value = null
  cleanEmbHistory.value = []; noisyEmbHistory.value = []; denoisedEmbHistory.value = []; genreIndexHistory.value = []
}

async function stop() {
  if (capture instanceof FileCapture) capture.pause()
  else capture?.stop()
  clearInterval(inferTimer); isRunning.value = false
  if (patchHistory.value.length === 0 || !engine.value) return
  const probs = patchHistory.value.map(h => new Float32Array(h.probs))
  try {
    const result = await engine.value.finalize(probs)
    finalTop5.value = result.top5
  } catch (e) {
    console.warn('finalize error:', e)
  }
}
</script>

<style>
*, *::before, *::after {
  box-sizing: border-box;
}
body {
  margin: 0;
  background-color: #f9fafb;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}
.app-container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 40px 20px;
  color: #333;
}
.header {
  text-align: center;
  margin-bottom: 24px;
}
.header h1 {
  font-size: 2.5rem;
  margin: 0;
  background: -webkit-linear-gradient(45deg, #6b21a8, #3b82f6);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-weight: 800;
}
.upload-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-bottom: 40px;
}
.upload-text {
  color: #6b7280;
  font-size: 1.05rem;
  text-align: center;
  margin-top: 12px;
  margin-bottom: 0;
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

/* New Dashboard Layout */
.dashboard-row {
  display: flex;
  gap: 20px;
  flex-wrap: nowrap;
  align-items: stretch;
  min-height: 360px;
}
.dash-col {
  display: flex;
  flex-direction: column;
  min-width: 0;
}
.dash-col-mel {
  flex: 1.2;
}
.dash-col-act {
  flex: 1.2;
}
.dash-col-top5 {
  flex: 0.8;
  min-width: 250px;
}
@media (max-width: 900px) {
  .dashboard-row {
    flex-direction: column;
  }
  .dash-col {
    flex: 1;
  }
}
</style>