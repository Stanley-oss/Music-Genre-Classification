<!-- File: ./frontend/src/components/GenreChart.vue -->
<template>
  <div class="chart-box">
    <v-chart class="chart" :option="option" autoresize />
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent, DataZoomComponent, MarkLineComponent } from 'echarts/components' // 加入 MarkLine
import VChart from 'vue-echarts'

use([ CanvasRenderer, LineChart, GridComponent, TooltipComponent, LegendComponent, DataZoomComponent, MarkLineComponent ])

const props = defineProps({
  genres: Array,
  history: Array, 
  time: Number,
  playhead: Number, // 新增：红色播放头
  duration: Number 
})

const palette =[
  '#8b5cf6', '#ef4444', '#f59e0b', '#10b981', '#3b82f6',
  '#ec4899', '#6366f1', '#14b8a6', '#f97316', '#64748b'
]

const option = computed(() => {
  const sortedHistory = [...props.history].sort((a, b) => a.t - b.t)

  const series = props.genres.map((g, i) => {
    const s = {
      name: g,
      type: 'line',
      smooth: false, 
      showSymbol: true, 
      symbolSize: 6,
      lineStyle: { width: 2 },
      itemStyle: { color: palette[i % palette.length] },
      data: sortedHistory.map((h) => [h.t, (h.probs[i] * 100).toFixed(1)]),
    }

    // 只有在第一组数据中挂载 markLine 以避免重复渲染红线
    if (i === 0 && props.playhead >= 0 && props.duration > 0) {
      s.markLine = {
        symbol:['none', 'none'],
        animation: false, // 禁用动画使得跟随绝对丝滑
        silent: true,
        data: [{ xAxis: props.playhead }],
        lineStyle: { color: '#ef4444', width: 2, type: 'dashed' },
        label: { show: false }
      }
    }
    return s
  })

  const xMax = props.duration > 0 ? props.duration : Math.max(10, props.time + 2)

  return {
    tooltip: { trigger: 'axis' },
    legend: { data: props.genres, top: 0, itemWidth: 16, itemHeight: 10, icon: 'roundRect', textStyle: { fontSize: 12, color: '#6b7280' } },
    grid: { left: '2%', right: '3%', bottom: '15%', top: '15%', containLabel: true },
    xAxis: { type: 'value', name: 'Time (seconds)', nameLocation: 'middle', nameGap: 25, min: 0, max: xMax, axisLine: { show: false }, axisTick: { show: false }, splitLine: { show: false } },
    yAxis: { type: 'value', name: '%', min: 0, max: 100, splitLine: { lineStyle: { color: '#f3f4f6' } } },
    dataZoom:[
      { type: 'inside', xAxisIndex: 0, filterMode: 'none' },
      { type: 'slider', xAxisIndex: 0, filterMode: 'none', height: 24, bottom: 8 }
    ],
    series,
  }
})
</script>

<style scoped>
.chart-box {
  width: 100%;
  height: 400px;
}
.chart {
  width: 100%;
  height: 100%;
}
</style>