<template>
  <div class="top5-wrapper">
    <h3 class="title">Top 5 Genres</h3>
    <div class="list">
      <div v-for="(item, idx) in top5" :key="item.genre" class="genre-row">
        <div class="left">
          <svg class="icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7v8a2 2 0 002 2h6M8 7V5a2 2 0 012-2h4.586a1 1 0 01.707.293l4.414 4.414a1 1 0 01.293.707V15a2 2 0 01-2 2h-2M8 7H6a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2v-2"></path>
          </svg>
          <span class="genre-name">{{ item.genre }}</span>
        </div>
        <div class="right">
          <span class="pct">{{ pct(item.probability) }}</span>
        </div>
        <div class="progress-bg">
           <div class="progress-fill" :style="{ width: pct(item.probability) }"></div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
const props = defineProps({ top5: Array })
const pct = (p) => (p * 100).toFixed(1) + '%'
</script>

<style scoped>
.top5-wrapper {
  width: 100%;
  max-width: 600px;
  margin: 0 auto;
}
.title {
  text-align: center;
  font-size: 1.25rem;
  font-weight: 700;
  margin-bottom: 20px;
  color: #111827;
}
.list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.genre-row {
  background: #ffffff;
  border-radius: 8px;
  padding: 16px 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 1px 4px rgba(0,0,0,0.04);
  border: 1px solid #f3f4f6;
  position: relative;
  overflow: hidden;
}
.left {
  display: flex;
  align-items: center;
  gap: 12px;
  z-index: 1;
}
.icon {
  width: 18px;
  height: 18px;
  color: #9ca3af;
}
.genre-name {
  font-weight: 600;
  color: #374151;
  text-transform: capitalize;
  font-size: 1.05rem;
}
.right {
  z-index: 1;
}
.pct {
  font-weight: 700;
  color: #111827;
  font-size: 1.05rem;
}
.progress-bg {
  position: absolute;
  bottom: 0;
  left: 0;
  height: 4px;
  width: 100%;
  background: transparent;
}
.progress-fill {
  height: 100%;
  background: #6366f1;
  transition: width 0.5s ease;
}
</style>