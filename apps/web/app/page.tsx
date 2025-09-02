'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { createChart, ISeriesApi, UTCTimestamp } from 'lightweight-charts';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

type Ohlcv = { ts: string; open: number; high: number; low: number; close: number; volume: number; };

export default function HomePage() {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<ReturnType<typeof createChart> | null>(null);
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);

  const [symbol, setSymbol] = useState('2330.TW');
  const [range, setRange] = useState<{start?: string; end?: string}>({});
  const [tf, setTf] = useState<'5m'|'15m'|'1h'|'1d'>('1d');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [minStepSec, setMinStepSec] = useState<number | null>(null);
  const [rangeLabel, setRangeLabel] = useState<string>('');

  const fetchOhlcv = async (sym: string, timeframe: string) => {
    // Choose sensible default ranges per timeframe to avoid short windows
    const tfToRange: Record<string,string> = { '5m':'60d', '15m':'60d', '1h':'1y', '1d':'5y' };
    const rng = tfToRange[timeframe] || '5y';
    const params = new URLSearchParams({ symbol: sym, tf: timeframe, limit: String(2000) });
    params.set('rng', rng);
    const url = `${API}/chart/ohlcv_live?${params.toString()}`;
    const r = await fetch(url, { cache: 'no-store', headers: { 'Accept': 'application/json' } });
    if (!r.ok) {
      const text = await r.text().catch(() => '');
      throw new Error(`GET ${url} failed: HTTP ${r.status}${text ? ` - ${text}` : ''}`);
    }
    const j = await r.json();
    setRangeLabel(j.range || rng);
    const data = (j.items as Ohlcv[]).map(d => {
      const date = new Date(d.ts);
      if (timeframe === '1d') {
        // Use BusinessDay-like time to avoid DST/timezone visual artifacts on daily bars
        return {
          time: {
            year: date.getUTCFullYear(),
            month: (date.getUTCMonth() + 1) as number,
            day: date.getUTCDate() as number,
          } as any,
          open: d.open, high: d.high, low: d.low, close: d.close,
        };
      }
      return {
        time: (date.getTime() / 1000) as UTCTimestamp,
        open: d.open, high: d.high, low: d.low, close: d.close,
      };
    });
    return data;
  };

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, { height: 420 });
    const series = chart.addCandlestickSeries();
    chartRef.current = chart;
    seriesRef.current = series;

    const resize = () => chart.applyOptions({ width: containerRef.current!.clientWidth });
    window.addEventListener('resize', resize);

    setError(null);
    fetchOhlcv(symbol, tf)
      .then(data => {
        series.setData(data);
        chart.timeScale().fitContent();
      })
      .catch((e: unknown) => {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
      });

    // Robust time handling for BusinessDay (daily) and epoch seconds (intraday)
    type ChartTime = number | { year: number; month: number; day: number };
    const toEpochSec = (t: ChartTime | undefined | null): number | null => {
      if (typeof t === 'number' && Number.isFinite(t)) return t;
      if (t && typeof t === 'object' && 'year' in t && 'month' in t && 'day' in t) {
        return Math.floor(Date.UTC(t.year, (t.month as number) - 1, t.day as number) / 1000);
      }
      return null;
    };

    let selecting = false;
    let startT: ChartTime | null = null;
    chart.subscribeClick(param => {
      if (!param.time) return; // clicked on whitespace
      if (!selecting) {
        selecting = true;
        startT = param.time as ChartTime;
      } else {
        selecting = false;
        const t1 = toEpochSec(startT);
        const t2 = toEpochSec(param.time as ChartTime);
        if (t1 == null || t2 == null) return;
        const a = Math.min(t1, t2);
        const b = Math.max(t1, t2);
        const iso = (sec:number) => new Date(sec*1000).toISOString().slice(0,10);
        setRange({ start: iso(a), end: iso(b) });
      }
    });

    return () => { window.removeEventListener('resize', resize); chart.remove(); };
  }, []);

  // Live data: no DB meta; clear any previous state
  useEffect(() => { setMinStepSec(null); }, [symbol]);

  // Auto refresh to keep latest candle updated
  const refreshMs = useMemo(() => {
    switch (tf) {
      case '5m': return 15000;   // 15s
      case '15m': return 60000;  // 60s
      case '1h': return 120000;  // 2min
      case '1d':
      default: return 300000;    // 5min
    }
  }, [tf]);

  useEffect(() => {
    if (!seriesRef.current || !chartRef.current) return;
    let cancelled = false;
    const tick = () => {
      setError(null);
      fetchOhlcv(symbol, tf)
        .then(d => { if (!cancelled) { seriesRef.current?.setData(d); /* do not refit on every refresh */ } })
        .catch((e: unknown)=>{
          if (cancelled) return;
          const msg = e instanceof Error ? e.message : String(e);
          setError(msg);
        });
    };
    tick();
    const id = setInterval(tick, refreshMs);
    return () => { cancelled = true; clearInterval(id); };
  }, [symbol, tf, refreshMs]);

  const onFindSimilar = async () => {
    if (!range.start || !range.end) {
      alert('請先在圖上點兩下選取起訖（第一次點＝起點；第二次點＝終點）');
      return;
    }
    const m = 1;
    // Quick client-side validation to reduce 400s
    try {
      const start = new Date(range.start);
      const end = new Date(range.end);
      const approxDays = Math.max(0, Math.round((+end - +start) / 86400000));
      if (approxDays < m + 2) {
        alert(`選取區間太短，請至少選取約 ${m + 2} 個交易日（目前約 ${approxDays} 天）`);
        return;
      }
    } catch {}
    try {
      // Call same-origin proxy to avoid extension/CORS issues
      const url = `/api/similar/search`;
      const res = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type':'application/json','Accept':'application/json'},
        body: JSON.stringify({ symbol, start: range.start, end: range.end, m, top: 5 })
      });
      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(`POST ${url} failed: HTTP ${res.status}${text ? ` - ${text}` : ''}`);
      }
      const j = await res.json();
      alert('相似片段 Top 5:\n' + j.items.map((x:any, i:number)=>`${i+1}. ${x.symbol} ${x.start_time.slice(0,10)}~${x.end_time.slice(0,10)}  dist=${x.distance.toFixed(3)}`).join('\n'));
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      alert('呼叫相似搜尋失敗：\n' + msg + '\n\n建議：\n1) 確認 API 已啟動：http://localhost:8000/health\n2) 停用可能攔截請求的瀏覽器外掛（或改用無痕視窗）\n3) 若前端非 3000 埠，請調整 API CORS 設定');
    }
  };

  return (
    <main style={{ padding: 20 }}>
      <h1 style={{ fontSize: 22, fontWeight: 700 }}>FinLab Starter (Fixed)</h1>
      <div style={{ display:'flex', gap: 8, alignItems:'center', marginBottom: 12 }}>
        <label>Symbol：</label>
        <select value={symbol} onChange={e=>setSymbol(e.target.value)}>
          <option value="2330.TW">2330.TW</option>
          <option value="AAPL">AAPL</option>
        </select>
        <label style={{marginLeft:8}}>時間尺度：</label>
        <select value={tf} onChange={e=>setTf(e.target.value as any)}>
          <option value="5m">5 分鐘</option>
          <option value="15m">15 分鐘</option>
          <option value="1h">1 小時</option>
          <option value="1d">1 天</option>
        </select>
        <button
          disabled={loading}
          onClick={()=>{
            setError(null);
            setLoading(true);
            fetchOhlcv(symbol, tf)
              .then(d=>{ seriesRef.current?.setData(d); chartRef.current?.timeScale().fitContent(); })
              .catch((e: unknown)=>{ const msg = e instanceof Error ? e.message : String(e); setError(msg); })
              .finally(()=> setLoading(false));
          }}
        >
          {loading ? '載入中…' : '載入'}
        </button>
        <button onClick={onFindSimilar}>找相似</button>
        <span style={{marginLeft:12, opacity:0.7}}>
          小技巧：在圖上點兩下依序選取「起點」與「終點」，再按「找相似」
        </span>
      </div>
      <div ref={containerRef} style={{ width:'100%', height: 420, border:'1px solid #e5e7eb', borderRadius:8 }} />
      {error && (
        <div style={{marginTop:10, padding:10, background:'#fff7ed', color:'#9a3412', border:'1px solid #fdba74', borderRadius:6}}>
          錯誤：{error}
        </div>
      )}
      <div style={{marginTop:10, fontSize:12, opacity:0.7}}>
        API：{API}　|　TF：{tf}　|　範圍：{rangeLabel || '—'}　|　選取區間：{range.start || '—'} ~ {range.end || '—'}
      </div>
    </main>
  );
}
