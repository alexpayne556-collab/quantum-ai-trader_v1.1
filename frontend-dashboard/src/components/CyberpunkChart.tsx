import { useEffect, useRef } from 'react'
import {
  createChart,
  CrosshairMode,
  CandlestickSeriesPartialOptions,
  ISeriesApi,
  Time,
} from 'lightweight-charts'

type Candle = {
  time: Time
  open: number
  high: number
  low: number
  close: number
}

function generateCandles(count = 200): Candle[] {
  const result: Candle[] = []
  let t = Math.floor(Date.now() / 1000) - count * 24 * 3600
  let price = 300
  for (let i = 0; i < count; i++) {
    const drift = (Math.sin(i / 15) + Math.cos(i / 30)) * 0.6
    const vol = 1.2 + Math.abs(Math.sin(i / 10)) * 2.5
    const change = drift + (Math.random() - 0.5) * vol
    const open = price
    const close = Math.max(5, open + change)
    const high = Math.max(open, close) + Math.random() * (vol * 0.8)
    const low = Math.min(open, close) - Math.random() * (vol * 0.8)
    result.push({ time: (t += 24 * 3600) as Time, open, high, low, close })
    price = close
  }
  return result
}

function ema(values: number[], period: number) {
  const k = 2 / (period + 1)
  let ema = values[0]
  const out = [ema]
  for (let i = 1; i < values.length; i++) {
    ema = values[i] * k + ema * (1 - k)
    out.push(ema)
  }
  return out
}

export default function CyberpunkChart() {
  const containerRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!containerRef.current) return

    const chart = createChart(containerRef.current, {
      autoSize: true,
      layout: {
        background: { type: 'solid', color: '#0a0e27' },
        textColor: '#ccd6f6',
      },
      grid: {
        vertLines: { color: '#13203b' },
        horzLines: { color: '#13203b' },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: '#1a2547' },
      timeScale: { borderColor: '#1a2547' },
    })

    // Neon ambient glow using a transparent overlay
    const glow = document.createElement('div')
    glow.style.position = 'absolute'
    glow.style.inset = '0'
    glow.style.pointerEvents = 'none'
    glow.style.boxShadow = '0 0 120px 20px #00d9ff22 inset, 0 0 160px 30px #ff006e11 inset'
    containerRef.current.appendChild(glow)

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#00ff88',
      downColor: '#ff006e',
      borderUpColor: '#00ff88',
      borderDownColor: '#ff006e',
      wickUpColor: '#00ff88',
      wickDownColor: '#ff006e',
    } as CandlestickSeriesPartialOptions)
    
    const attachData = (candles: Candle[]) => {
      candleSeries.setData(candles)

      // EMA ribbon (13 / 21 / 55)
      const closes = candles.map((c) => c.close)
      const ema13 = ema(closes, 13)
      const ema21 = ema(closes, 21)
      const ema55 = ema(closes, 55)

      const ema13Series = chart.addLineSeries({ color: '#00ccff', lineWidth: 1 })
      const ema21Series = chart.addLineSeries({ color: '#0066ff', lineWidth: 1 })
      const ema55Series = chart.addLineSeries({ color: '#6a00ff', lineWidth: 2 })
      ema13Series.setData(candles.map((c, i) => ({ time: c.time, value: ema13[i] })))
      ema21Series.setData(candles.map((c, i) => ({ time: c.time, value: ema21[i] })))
      ema55Series.setData(candles.map((c, i) => ({ time: c.time, value: ema55[i] })))

      // Forecast cone (stylized preview)
      const last = candles[candles.length - 1]
      const horizon = 24
      const forecast: { time: Time; value: number }[] = []
      let p = last.close
      for (let i = 1; i <= horizon; i++) {
        p = p * (1 + 0.0005 * i) + Math.sin(i / 3) * 0.6
        const lastTime = typeof last.time === 'number' ? (last.time as number) : Math.floor(Date.now() / 1000)
        forecast.push({ time: (lastTime + i * 24 * 3600) as Time, value: p })
      }
      const forecastSeries: ISeriesApi<'Line'> = chart.addLineSeries({
        color: '#00d9ff',
        lineWidth: 2,
        lineStyle: 1,
      })
      forecastSeries.setData(forecast)

      // Confidence cone as area around forecast
      const upperSeries = chart.addAreaSeries({
        topColor: 'rgba(0, 217, 255, 0.20)',
        bottomColor: 'rgba(0, 217, 255, 0.02)',
        lineColor: 'rgba(0,0,0,0)',
        lineWidth: 0,
        priceScaleId: 'right',
      })
      upperSeries.setData(
        forecast.map((pt, i) => ({ time: pt.time, value: pt.value * (1 + 0.02 + i * 0.0005) }))
      )
    }

    // Try backend first; fallback to mock
    const apiBase = (import.meta as any).env?.VITE_API_URL || 'http://127.0.0.1:8000'
    fetch(`${apiBase}/api/ohlcv?ticker=SPY&limit=240`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((rows: any[]) => {
        const candles: Candle[] = rows.map((r) => ({
          time: r.time, // YYYY-MM-DD accepted as BusinessDay
          open: Number(r.open),
          high: Number(r.high),
          low: Number(r.low),
          close: Number(r.close),
        }))
        attachData(candles)
      })
      .catch(() => {
        const candles = generateCandles(240)
        attachData(candles)
      })

    const resize = () => chart.timeScale().fitContent()
    window.addEventListener('resize', resize)
    resize()

    return () => {
      window.removeEventListener('resize', resize)
      chart.remove()
    }
  }, [])

  return (
    <div className="cyber-shell">
      <div className="hud">
        <div className="badge">ALGO LIVE</div>
        <div className="title">Quantum Trader • Cyberpunk Preview</div>
        <div className="edge">Edge: 72</div>
      </div>
      <div ref={containerRef} className="chart-host" />
    </div>
  )
}
