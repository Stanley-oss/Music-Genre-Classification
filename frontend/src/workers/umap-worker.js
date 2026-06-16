/**
 * UMAP Web Worker — runs dimensionality reduction off the main thread.
 * 
 * Now imports from the npm-installed umap-js package, which Vite bundles
 * into the worker automatically via `new Worker(new URL(...), { type: 'module' })`.
 *
 * Protocol:
 *   IN:  { type: 'compute', clean: number[][], noisy: number[][], denoised: number[][], genreIndices: number[] }
 *   OUT: { type: 'result', clean2d, noisy2d, denoised2d, genreIndices }
 *   OUT: { type: 'error', message }
 */
import { UMAP } from 'umap-js'

function runOne(data, nNeighbors) {
  if (!data || data.length < 3) return null
  const umap = new UMAP({
    nNeighbors: Math.min(nNeighbors, data.length - 1),
    minDist: 0.1,
    nComponents: 2,
    nEpochs: 200
  })
  return umap.fit(data)
}

self.onmessage = (e) => {
  if (e.data.type !== 'compute') return

  try {
    const { clean, noisy, denoised, genreIndices } = e.data

    // Cap at 64 points for performance
    const maxPts = 64
    const n = Math.min(clean.length, maxPts)
    const sliceStart = clean.length > n ? clean.length - n : 0

    const cSlice = clean.slice(sliceStart)
    const nSlice = noisy.slice(sliceStart)
    const dSlice = denoised.slice(sliceStart)
    const gSlice = genreIndices.slice(sliceStart)

    const nNeighbors = Math.min(15, n - 1)

    const clean2d = runOne(cSlice, nNeighbors)
    const noisy2d = runOne(nSlice, nNeighbors)
    const denoised2d = runOne(dSlice, nNeighbors)

    self.postMessage({
      type: 'result',
      clean2d,
      noisy2d,
      denoised2d,
      genreIndices: gSlice,
      sliceStart
    })
  } catch (err) {
    self.postMessage({ type: 'error', message: err.message })
  }
}
