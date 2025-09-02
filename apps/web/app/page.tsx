'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { createChart, ISeriesApi, UTCTimestamp, ColorType } from 'lightweight-charts';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

type Ohlcv = { ts: string; open: number; high: number; low: number; close: number; volume: number; };

export default function HomePage() {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<ReturnType<typeof createChart> | null>(null);
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);

  const defaultSymbols = (process.env.NEXT_PUBLIC_SYMBOLS || '2330.TW,AAPL')
    .split(',').map(s=>s.trim()).filter(Boolean);
  const [symbols, setSymbols] = useState<string[]>(defaultSymbols);
  const [newSym, setNewSym] = useState('');
  const [symbol, setSymbol] = useState('2330.TW');
  const [range, setRange] = useState<{start?: string; end?: string}>({});
  const [tf, setTf] = useState<'5m'|'15m'|'1h'|'1d'>('1d');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [minStepSec, setMinStepSec] = useState<number | null>(null);
  const [rangeLabel, setRangeLabel] = useState<string>('');
  const [symbolOpen, setSymbolOpen] = useState(false);
  const symbolMenuRef = useRef<HTMLDivElement | null>(null);
  const [tfOpen, setTfOpen] = useState(false);
  const tfMenuRef = useRef<HTMLDivElement | null>(null);
  const tfLabels: Record<string,string> = { '5m':'5 分鐘', '15m':'15 分鐘', '1h':'1 小時', '1d':'1 天' };
  const [sentSym, setSentSym] = useState<{label:string; score:number; rsi14:number; slope_norm:number; vol:number} | null>(null);
  const [sentEnv, setSentEnv] = useState<{label:string; pct_above_ma50:number; pct_above_ma20:number; avg_rsi:number; universe_size:number} | null>(null);
  const [newsSym, setNewsSym] = useState<Array<{title:string; url:string; publisher?:string; published_at?:string}>>([]);
  const [newsMkt, setNewsMkt] = useState<Array<{title:string; url:string; publisher?:string; published_at?:string}>>([]);

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
    const chart = createChart(containerRef.current, {
      height: 560,
      layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: '#334155' },
      grid: { vertLines: { color: '#e2e8f0' }, horzLines: { color: '#e2e8f0' } },
      rightPriceScale: { borderColor: '#cbd5e1' },
      timeScale: { borderColor: '#cbd5e1' },
    });
    const series = chart.addCandlestickSeries({
      upColor: '#16a34a', downColor: '#ef4444',
      wickUpColor: '#16a34a', wickDownColor: '#ef4444',
      borderUpColor: '#16a34a', borderDownColor: '#ef4444',
    });
    chartRef.current = chart;
    seriesRef.current = series;

    const resize = () => chart.applyOptions({ width: containerRef.current!.clientWidth });
    window.addEventListener('resize', resize);

    setError(null);
    if (symbol.trim()) {
      fetchOhlcv(symbol, tf)
        .then(data => {
          series.setData(data);
          chart.timeScale().fitContent();
        })
        .catch((e: unknown) => {
          const msg = e instanceof Error ? e.message : String(e);
          setError(msg);
        });
    }

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

  // Load/save custom symbols list from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem('finlab_symbols');
      if (saved) {
        const arr = JSON.parse(saved);
        if (Array.isArray(arr) && arr.every(x => typeof x === 'string')) {
          setSymbols(Array.from(new Set(arr)) as string[]);
          if (!arr.includes(symbol) && arr.length > 0) setSymbol(arr[0]);
        }
      }
    } catch {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    try { localStorage.setItem('finlab_symbols', JSON.stringify(symbols)); } catch {}
  }, [symbols]);

  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      const node = e.target as Node;
      if (symbolMenuRef.current && !symbolMenuRef.current.contains(node)) setSymbolOpen(false);
      if (tfMenuRef.current && !tfMenuRef.current.contains(node)) setTfOpen(false);
    };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, []);

  // Live data: no DB meta; clear any previous state
  useEffect(() => { setMinStepSec(null); }, [symbol]);

  // Fetch sentiment when symbol or universe changes (daily-based)
  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      if (!symbol.trim()) { setSentSym(null); setSentEnv(null); return; }
      try {
        const r = await fetch('/api/sentiment/summary', {
          method: 'POST', headers: { 'Content-Type':'application/json' },
          body: JSON.stringify({ symbol, universe: symbols })
        });
        if (!r.ok) return;
        const j = await r.json();
        if (cancelled) return;
        setSentSym({
          label: j.symbol_sentiment?.label,
          score: j.symbol_sentiment?.score ?? 0,
          rsi14: j.symbol_sentiment?.rsi14 ?? 0,
          slope_norm: j.symbol_sentiment?.slope_norm ?? 0,
          vol: j.symbol_sentiment?.vol ?? 0,
        });
        setSentEnv({
          label: j.environment_sentiment?.label,
          pct_above_ma50: j.environment_sentiment?.pct_above_ma50 ?? 0,
          pct_above_ma20: j.environment_sentiment?.pct_above_ma20 ?? 0,
          avg_rsi: j.environment_sentiment?.avg_rsi ?? 0,
          universe_size: j.environment_sentiment?.universe_size ?? symbols.length,
        });
      } catch {}
      try {
        const r2 = await fetch('/api/news/summary', {
          method: 'POST', headers: { 'Content-Type':'application/json' },
          body: JSON.stringify({ symbol, universe: symbols })
        });
        if (r2.ok) {
          const j2 = await r2.json();
          if (!cancelled) {
            setNewsSym(Array.isArray(j2.symbol_news) ? j2.symbol_news : []);
            setNewsMkt(Array.isArray(j2.market_news) ? j2.market_news : []);
          }
        }
      } catch {}
    };
    run();
    return () => { cancelled = true; };
  }, [symbol, symbols]);

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
      if (!symbol.trim()) return;
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
    const m = 30;
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
        body: JSON.stringify({ symbol, start: range.start, end: range.end, m, top: 5, universe: symbols })
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

  const onDetectPattern = async () => {
    if (!range.start || !range.end) {
      alert('請先在圖上點兩下選取起訖（第一次點＝起點；第二次點＝終點）');
      return;
    }
    try {
      const url = `/api/pattern/classify`;
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        body: JSON.stringify({ symbol, start: range.start, end: range.end, tf }),
      });
      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(`POST ${url} failed: HTTP ${res.status}${text ? ` - ${text}` : ''}`);
      }
      const j = await res.json();
      const nameMap: Record<string,string> = {
        sym_triangle: '對稱三角形',
        asc_triangle: '上升三角形',
        desc_triangle: '下降三角形',
        unknown: '未辨識/非三角形'
      };
      alert(`型態辨識：${nameMap[j.label] || j.label}\n信心：${(j.confidence*100).toFixed(0)}%\n詳細：\n- 高點斜率: ${j.meta?.slope_high?.toFixed?.(6)}\n- 低點斜率: ${j.meta?.slope_low?.toFixed?.(6)}\n- R2(高/低): ${j.meta?.r2_high?.toFixed?.(2)} / ${j.meta?.r2_low?.toFixed?.(2)}\n- 範圍收斂比: ${j.meta?.contraction?.toFixed?.(2)}`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      alert('型態辨識失敗：\n' + msg);
    }
  };

  return (
    <main className="app" suppressHydrationWarning>
      <div className="header" suppressHydrationWarning>
        <h1 className="title">FinLab Starter</h1>
      </div>
      <div className="toolbar" suppressHydrationWarning>
        <span className="label">Symbol：</span>
        <div className="dropdown" ref={symbolMenuRef}>
          <button className="selectbox" onClick={()=>setSymbolOpen(v=>!v)}>{symbol || '選擇代碼'}</button>
          {symbolOpen && (
            <div className="menu" role="menu">
              {symbols.length === 0 && <div className="menu-item">尚無代碼</div>}
              {symbols.map(s => (
                <div className="menu-item" key={s}>
                  <button
                    type="button"
                    className="menu-choose"
                    onMouseDown={(e)=>{ e.preventDefault(); e.stopPropagation(); setSymbol(s); setSymbolOpen(false); }}
                  >{s}</button>
                  <button
                    type="button"
                    className="menu-del"
                    onMouseDown={(e)=>{ e.preventDefault(); e.stopPropagation(); const next = symbols.filter(x=>x!==s); setSymbols(next); if (symbol === s) setSymbol(next[0] || ''); }}
                  >x</button>
                </div>
              ))}
            </div>
          )}
        </div>
        <input
          className="input"
          placeholder="新增代碼，如 AAPL / 2330.TW"
          value={newSym}
          onChange={e=>setNewSym(e.target.value)}
          onKeyDown={e=>{ if(e.key==='Enter'){ const v=newSym.trim(); if(v && !symbols.includes(v)){ setSymbols([v, ...symbols].slice(0,50)); setNewSym(''); } }}}
          aria-label="新增代碼"
        />
        <button className="btn" onClick={()=>{ const v=newSym.trim(); if(v && !symbols.includes(v)){ setSymbols([v, ...symbols].slice(0,50)); setNewSym(''); } }}>加入</button>
        <span className="label">時間尺度：</span>
        <div className="dropdown" ref={tfMenuRef}>
          <button className="selectbox" onClick={()=>setTfOpen(v=>!v)}>{tfLabels[tf] || tf}</button>
          {tfOpen && (
            <div className="menu" role="menu">
              {(['5m','15m','1h','1d'] as const).map(v => (
                <div className="menu-item" key={v}>
                  <button
                    type="button"
                    className="menu-choose"
                    onMouseDown={(e)=>{ e.preventDefault(); e.stopPropagation(); setTf(v); setTfOpen(false); }}
                  >{tfLabels[v]}</button>
                </div>
              ))}
            </div>
          )}
        </div>
        <button className="btn btn-primary"
          disabled={loading}
          onClick={()=>{
            setError(null);
            if (!symbol.trim()) { alert('請先選擇或新增代碼'); return; }
            setLoading(true);
            fetchOhlcv(symbol, tf)
              .then(d=>{ seriesRef.current?.setData(d); chartRef.current?.timeScale().fitContent(); })
              .catch((e: unknown)=>{ const msg = e instanceof Error ? e.message : String(e); setError(msg); })
              .finally(()=> setLoading(false));
          }}
        >
          {loading ? '載入中…' : '載入'}
        </button>
        <button className="btn" onClick={onFindSimilar}>找相似</button>
        <button className="btn" onClick={onDetectPattern}>辨識型態</button>
        <span className="tip">小技巧：在圖上點兩下依序選取「起點」與「終點」，再按「找相似」</span>
      </div>
      {/* custom dropdown handles deletion; chips removed */}
      <div className="card" suppressHydrationWarning>
        <div ref={containerRef} className="chart" suppressHydrationWarning />
      </div>
      {error && (
        <div className="alert" suppressHydrationWarning>錯誤：{error}</div>
      )}
      <div className="info" suppressHydrationWarning>
        API：{API}　|　TF：{tf}　|　範圍：{rangeLabel || '—'}　|　選取區間：{range.start || '—'} ~ {range.end || '—'}
      </div>
      <div className="sentiment">
        <div className="sent-card">
          <div className="sent-title">個股情緒</div>
          <div className="sent-metrics">
            <div className="sent-kv"><span>狀態</span><span className="badge">{sentSym?.label || '—'}</span></div>
            <div className="sent-kv"><span>分數</span><span>{sentSym ? Math.round(sentSym.score*100) : '—'}%</span></div>
            <div className="sent-kv"><span>RSI(14)</span><span>{sentSym ? sentSym.rsi14.toFixed(1) : '—'}</span></div>
            <div className="sent-kv"><span>趨勢斜率</span><span>{sentSym ? sentSym.slope_norm.toFixed(4) : '—'}</span></div>
            <div className="sent-kv"><span>波動率(近60)</span><span>{sentSym ? (sentSym.vol*100).toFixed(2) : '—'}%</span></div>
          </div>
        </div>
        <div className="sent-card">
          <div className="sent-title">環境情緒</div>
          <div className="sent-metrics">
            <div className="sent-kv"><span>狀態</span><span className="badge">{sentEnv?.label || '—'}</span></div>
            <div className="sent-kv"><span>樣本數</span><span>{sentEnv?.universe_size ?? 0}</span></div>
            <div className="sent-kv"><span>MA50 之上</span><span>{sentEnv ? Math.round(sentEnv.pct_above_ma50*100) : '—'}%</span></div>
            <div className="sent-kv"><span>MA20 之上</span><span>{sentEnv ? Math.round(sentEnv.pct_above_ma20*100) : '—'}%</span></div>
            <div className="sent-kv"><span>平均 RSI</span><span>{sentEnv ? sentEnv.avg_rsi.toFixed(1) : '—'}</span></div>
          </div>
        </div>
      </div>
      <div className="news-card">
        <div className="news-title">新聞快照</div>
        <div className="news-grid">
          <div>
            <div className="sent-title">個股相關</div>
            <ul className="news-list">
              {newsSym.length === 0 && <li className="news-item">—</li>}
              {newsSym.slice(0,6).map((n,i) => (
                <li key={i} className="news-item">
                  <a href={n.url} target="_blank" rel="noopener noreferrer">{n.title}</a>
                  {n.publisher && <span style={{marginLeft:6, color:'#64748b'}}>· {n.publisher}</span>}
                  {n.published_at && <span style={{marginLeft:6, color:'#94a3b8'}}>· {new Date(n.published_at).toLocaleString('zh-TW', { hour12:false })}</span>}
                </li>
              ))}
            </ul>
          </div>
          <div>
            <div className="sent-title">大環境</div>
            <ul className="news-list">
              {newsMkt.length === 0 && <li className="news-item">—</li>}
              {newsMkt.slice(0,6).map((n,i) => (
                <li key={i} className="news-item">
                  <a href={n.url} target="_blank" rel="noopener noreferrer">{n.title}</a>
                  {n.publisher && <span style={{marginLeft:6, color:'#64748b'}}>· {n.publisher}</span>}
                  {n.published_at && <span style={{marginLeft:6, color:'#94a3b8'}}>· {new Date(n.published_at).toLocaleString('zh-TW', { hour12:false })}</span>}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </main>
  );
}
