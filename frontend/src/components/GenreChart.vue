<template>
  <div class="chart-box">
    <v-chart class="chart" :option="option" autoresize @datazoom="onDataZoom" />
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent, DataZoomComponent } from 'echarts/components'
import VChart from 'vue-echarts'

use([CanvasRenderer, LineChart, GridComponent, TooltipComponent, LegendComponent, DataZoomComponent])

const props = defineProps({ genres: Array, history: Array, time: Number, duration: Number })

const palette =['#8b5cf6', '#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#ec4899', '#6366f1', '#14b8a6', '#f97316', '#64748b']

const zoomStart = ref(0)
const zoomEnd = ref(100)

function onDataZoom(e) {
  if (e.batch && e.batch.length > 0) {
    if (e.batch[0].start != null) zoomStart.value = e.batch[0].start
    if (e.batch[0].end != null) zoomEnd.value = e.batch[0].end
  } else {
    if (e.start != null) zoomStart.value = e.start
    if (e.end != null) zoomEnd.value = e.end
  }
}

watch(() => props.duration, (newVal, oldVal) => {
  if (newVal !== oldVal && newVal > 0) { zoomStart.value = 0; zoomEnd.value = 100 }
})

watch(() => props.time, (newTime) => {
  const xMax = props.duration && props.duration > 0 ? Math.ceil(props.duration) : Math.max(10, props.time + 2)
  const currentPct = (newTime / xMax) * 100
  if (currentPct > zoomEnd.value) {
    const range = zoomEnd.value - zoomStart.value
    zoomEnd.value = Math.min(100, currentPct)
    zoomStart.value = zoomEnd.value - range
  } else if (currentPct < zoomStart.value) {
    const range = zoomEnd.value - zoomStart.value
    zoomStart.value = Math.max(0, currentPct)
    zoomEnd.value = zoomStart.value + range
  }
})

const option = computed(() => {
  const series = props.genres.map((g, i) => {
    const data =[];
    for (let j = 0; j < props.history.length; j++) {
      const current = props.history[j];
      const probValue = current.probs[i] * 100;
      const displayProb = probValue >= 1.0 ? probValue.toFixed(1) : null;
      data.push([current.t, displayProb]);
      
      if (j < props.history.length - 1) {
        const next = props.history[j + 1];
        if (next.t - current.t > 1.0) data.push([current.t + 0.1, null]);
      }
    }
    return {
      name: g, type: 'line', smooth: false, showSymbol: true, symbolSize: 6,
      connectNulls: false, lineStyle: { width: 2 },
      itemStyle: { color: palette[i % palette.length] }, data: data,
    }
  })

  const xMax = props.duration && props.duration > 0 ? Math.ceil(props.duration) : Math.max(10, props.time + 2)

  return {
    // 启用 ECharts 全局自动动画
    animation: true,
    animationDurationUpdate: 300,
    animationEasingUpdate: 'cubicOut',

    tooltip: { 
      trigger: 'axis', backgroundColor: 'rgba(255, 255, 255, 0.95)',
      extraCssText: 'box-shadow: 0 4px 16px rgba(0,0,0,0.08); border-radius: 8px; border: none; padding: 12px;',
      formatter: function (params) {
        const hoveredTime = Number(params[0].axisValue);
        const histItem = props.history.find(h => Math.abs(h.t - hoveredTime) < 0.01);
        if (!histItem) return '';
        let sortedGenres = props.genres.map((g, i) => ({
          name: g, prob: histItem.probs[i] * 100, color: palette[i % palette.length]
        })).sort((a, b) => b.prob - a.prob);
        let timeStr = hoveredTime.toFixed(1);
        let html = `<div style="font-weight:700; color:#374151; margin-bottom:10px; border-bottom:1px solid #e5e7eb; padding-bottom:6px;">Time: ${timeStr}s</div>`;
        sortedGenres.forEach(item => {
          html += `
            <div style="display:flex; justify-content:space-between; align-items:center; gap:24px; margin-bottom:6px;">
              <div style="display:flex; align-items:center;">
                <span style="display:inline-block; width:10px; height:10px; border-radius:50%; background-color:${item.color}; margin-right:8px;"></span>
                <span style="font-size:13px; color:#4b5563; text-transform:capitalize;">${item.name}</span>
              </div>
              <span style="font-weight:700; font-size:13px; color:#111827; font-variant-numeric:tabular-nums;">${item.prob.toFixed(1)}%</span>
            </div>
          `
        })
        return html
      }
    },
    legend: { data: props.genres, top: 0, itemWidth: 16, itemHeight: 10, icon: 'roundRect', textStyle: { fontSize: 12, color: '#6b7280' } },
    grid: { left: '2%', right: '3%', bottom: '15%', top: '15%', containLabel: true },
    xAxis: {
      type: 'value', name: 'Time (seconds)', nameLocation: 'middle', nameGap: 30, min: 0, max: xMax,
      axisLine: { show: false }, axisTick: { show: false }, splitLine: { show: false },
      axisLabel: { formatter: (val) => Number(val).toFixed(1) }
    },
    yAxis: { type: 'value', name: '%', min: 0, max: 100, splitLine: { lineStyle: { color: '#f3f4f6' } } },
    dataZoom:[
      { type: 'inside', xAxisIndex: 0, filterMode: 'none', start: zoomStart.value, end: zoomEnd.value },
      { type: 'slider', xAxisIndex: 0, filterMode: 'none', height: 20, bottom: 5, borderColor: 'transparent', handleStyle: { color: '#3b82f6' }, start: zoomStart.value, end: zoomEnd.value }
    ],
    series,
  }
})
</script>

<style scoped>
.chart-box { width: 100%; height: 420px; }
.chart { width: 100%; height: 100%; }
</style>