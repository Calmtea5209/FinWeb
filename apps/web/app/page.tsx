'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { createChart, ISeriesApi, UTCTimestamp, ColorType, TickMarkType } from 'lightweight-charts';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

type Ohlcv = { ts: string; open: number; high: number; low: number; close: number; volume: number; };

export default function HomePage() {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<ReturnType<typeof createChart> | null>(null);
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const ma20Ref = useRef<ISeriesApi<'Line'> | null>(null);
  const ma50Ref = useRef<ISeriesApi<'Line'> | null>(null);
  const volRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const highlightRef = useRef<HTMLDivElement | null>(null);
  const dataRef = useRef<Array<{time:any; open:number; high:number; low:number; close:number}>>([]);
  const ma20DataRef = useRef<Array<{ time:any; value:number }>>([]);
  const ma50DataRef = useRef<Array<{ time:any; value:number }>>([]);
  const volDataRef = useRef<Array<{ time:any; value:number; color?: string }>>([]);

  const defaultSymbols = (process.env.NEXT_PUBLIC_SYMBOLS || '2330.TW,AAPL')
    .split(',').map(s=>s.trim()).filter(Boolean);
  const [symbols, setSymbols] = useState<string[]>(defaultSymbols);
  const [newSym, setNewSym] = useState('');
  const [symbol, setSymbol] = useState<string>('2330.TW');
  const [range, setRange] = useState<{start?: string; end?: string}>({});
  const [tf, setTf] = useState<'5m'|'15m'|'1h'|'1d'>('1d');
  const tfRef = useRef<'5m'|'15m'|'1h'|'1d'>(tf);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [minStepSec, setMinStepSec] = useState<number | null>(null);
  const [rangeLabel, setRangeLabel] = useState<string>('');
  const [symbolOpen, setSymbolOpen] = useState(false);
  const symbolMenuRef = useRef<HTMLDivElement | null>(null);
  const [tfOpen, setTfOpen] = useState(false);
  const tfMenuRef = useRef<HTMLDivElement | null>(null);
  const tfLabels: Record<string,string> = { '5m':'5 分鐘', '15m':'15 分鐘', '1h':'1 小時', '1d':'1 天' };
  const [newsSym, setNewsSym] = useState<Array<{title:string; url:string; publisher?:string; published_at?:string}>>([]);
  const [newsMkt, setNewsMkt] = useState<Array<{title:string; url:string; publisher?:string; published_at?:string}>>([]);
  const [simOpen, setSimOpen] = useState(false);
  const [simItems, setSimItems] = useState<any[]>([]);
  const [simError, setSimError] = useState<string | null>(null);
  const [simIndex, setSimIndex] = useState(0);
  const simChartRef = useRef<ReturnType<typeof createChart> | null>(null);
  const simSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const simContainerRef = useRef<HTMLDivElement | null>(null);
  const [userEmail, setUserEmail] = useState<string | null>(null);
  const [isLoggedIn, setIsLoggedIn] = useState<boolean>(false);
  const [authReady, setAuthReady] = useState(false);
  const [canPersist, setCanPersist] = useState(false);
  // Selection indices (robust for intraday)
  const selIdxRef = useRef<{ i1: number | null; i2: number | null }>({ i1: null, i2: null });
  const [showMA20, setShowMA20] = useState(false);
  const [showMA50, setShowMA50] = useState(false);
  const [showVOL, setShowVOL] = useState(false);
  // Backtest modal state
  const [btOpen, setBtOpen] = useState(false);
  const [btFast, setBtFast] = useState(10);
  const [btSlow, setBtSlow] = useState(30);
  const [btCash, setBtCash] = useState(1000000);
  const [btJob, setBtJob] = useState<string | null>(null);
  const [btStatus, setBtStatus] = useState<string | null>(null);
  const [btReport, setBtReport] = useState<any | null>(null);

  const fetchOhlcv = async (sym: string, timeframe: string) => {
    // Choose sensible default ranges per timeframe to avoid short windows
    const tfToRange: Record<string,string> = { '5m':'60d', '15m':'60d', '1h':'1y', '1d':'5y' };
    const rng = tfToRange[timeframe] || '5y';
    const params = new URLSearchParams({ symbol: sym, tf: timeframe, limit: String(2000) });
    params.set('rng', rng);
    const url = `/api/chart/ohlcv_live?${params.toString()}`;
    const r = await fetch(url, { cache: 'no-store', headers: { 'Accept': 'application/json' } });
    if (!r.ok) {
      const text = await r.text().catch(() => '');
      throw new Error(`GET ${url} failed: HTTP ${r.status}${text ? ` - ${text}` : ''}`);
    }
    const j = await r.json();
    setRangeLabel(j.range || rng);
    const mapTime = (ts: string) => {
      const date = new Date(ts);
      if (timeframe === '1d') {
        return { year: date.getUTCFullYear(), month: (date.getUTCMonth() + 1) as number, day: date.getUTCDate() as number } as any;
      }
      return (date.getTime() / 1000) as UTCTimestamp;
    };
    const items = (j.items as Ohlcv[]);
    const price = items.map(d => ({ time: mapTime(d.ts), open: d.open, high: d.high, low: d.low, close: d.close }));
    // volume histogram, color by candle direction
    const upColor = 'rgba(22,163,74,0.45)';
    const downColor = 'rgba(239,68,68,0.45)';
    const vol = items.map(d => ({ time: mapTime(d.ts), value: d.volume ?? 0, color: (d.close >= d.open) ? upColor : downColor }));
    // SMA helpers
    const sma = (arr: number[], period: number) => {
      const out: Array<{ time:any; value:number }> = [];
      let sum = 0;
      for (let i = 0; i < arr.length; i++) {
        sum += arr[i];
        if (i >= period) sum -= arr[i - period];
        if (i >= period - 1) out.push({ time: price[i].time, value: sum / period });
      }
      return out;
    };
    const closes = items.map(d => d.close);
    const ma20 = sma(closes, 20);
    const ma50 = sma(closes, 50);
    return { price, vol, ma20, ma50 } as const;
  };

  // Fetch a slice for modal preview using current timeframe
  const fetchSlice = async (sym: string, startIso: string, endIso: string, timeframe: '5m'|'15m'|'1h'|'1d') => {
    const params = new URLSearchParams({ symbol: sym, tf: timeframe, limit: String(2000) });
    const tfToRange: Record<string,string> = { '5m':'60d', '15m':'60d', '1h':'1y', '1d':'5y' };
    params.set('rng', tfToRange[timeframe] || '5y');
    const url = `/api/chart/ohlcv_live?${params.toString()}`;
    const r = await fetch(url, { cache: 'no-store', headers: { 'Accept': 'application/json' } });
    if (!r.ok) throw new Error(`GET ${url} failed: HTTP ${r.status}`);
    const j = await r.json();
    const s = new Date(startIso).getTime();
    const e = new Date(endIso).getTime();
    const padMs = 1000 * 60 * 60 * 24 * 10; // +/- 10 days padding for context
    const data = (j.items as Ohlcv[]).map(d => ({
      time: (new Date(d.ts).getTime() / 1000) as UTCTimestamp,
      open: d.open, high: d.high, low: d.low, close: d.close,
    }))
    .filter(bar => {
      const t = (bar.time as number) * 1000;
      return t >= (s - padMs) && t <= (e + padMs);
    });
    return data;
  };

  useEffect(() => {
    tfRef.current = tf;
  }, [tf]);

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = createChart(containerRef.current, {
      height: 460,
      layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: '#334155' },
      grid: { vertLines: { color: '#e2e8f0' }, horzLines: { color: '#e2e8f0' } },
      rightPriceScale: { borderColor: '#cbd5e1' },
      leftPriceScale: { visible: false, borderColor: '#cbd5e1' },
      timeScale: { borderColor: '#cbd5e1' },
      localization: { locale: 'zh-TW' as any },
    });
    // volume histogram first (behind), attach to left scale for visible axis
    volRef.current = chart.addHistogramSeries({ priceScaleId: 'left', priceFormat: { type: 'volume' }, visible: false });
    const series = chart.addCandlestickSeries({
      upColor: '#16a34a', downColor: '#ef4444',
      wickUpColor: '#16a34a', wickDownColor: '#ef4444',
      borderUpColor: '#16a34a', borderDownColor: '#ef4444',
    });
    chartRef.current = chart;
    seriesRef.current = series;
    // MA lines (hidden init)
    ma20Ref.current = chart.addLineSeries({ color: '#f59e0b', lineWidth: 2, priceScaleId: 'right', visible: false });
    ma50Ref.current = chart.addLineSeries({ color: '#0ea5e9', lineWidth: 2, priceScaleId: 'right', visible: false });

    const resize = () => { chart.applyOptions({ width: containerRef.current!.clientWidth }); updateHighlight(); };
    window.addEventListener('resize', resize);

    setError(null);
    if (symbol.trim()) {
    fetchOhlcv(symbol, tf)
      .then(pack => {
        series.setData(pack.price);
        dataRef.current = pack.price as any;
        ma20DataRef.current = pack.ma20 as any;
        ma50DataRef.current = pack.ma50 as any;
        volDataRef.current = pack.vol as any;
        if (volRef.current) volRef.current.setData(pack.vol);
        if (ma20Ref.current) ma20Ref.current.setData(pack.ma20);
        if (ma50Ref.current) ma50Ref.current.setData(pack.ma50);
        chart.timeScale().fitContent();
        updateHighlight();
        // re-apply backtest markers if any
        applyBtMarkers(btReport);
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
        let a = Math.min(t1, t2);
        let b = Math.max(t1, t2);
        // For intraday, snap to nearest bar time to avoid mismatched seconds
        if (tf !== '1d') {
          const bars: any[] = (dataRef.current || []) as any[];
          const nearest = (sec: number): number => {
            let best: number = sec; let bestDiff = Number.POSITIVE_INFINITY;
            for (const d of bars) {
              if (typeof d.time !== 'number') continue;
              const s = d.time as number;
              const diff = Math.abs(s - sec);
              if (diff < bestDiff) { best = s; bestDiff = diff; if (diff === 0) break; }
            }
            return best;
          };
          if (bars.length > 0) { a = nearest(a); b = nearest(b); }
        }
        const iso = (sec:number) => {
          // Truncate to whole minutes for intraday to avoid seconds
          const curTf = tfRef.current;
          if (curTf === '1d') {
            const d = new Date(sec*1000);
            return d.toISOString().slice(0,10); // YYYY-MM-DD
          }
          const sTrunc = Math.floor(sec/60) * 60; // drop seconds
          const d = new Date(sTrunc*1000);
          return d.toISOString().slice(0,16) + ':00Z'; // YYYY-MM-DDTHH:mm:00Z
        };
        const bars = (dataRef.current || []) as any[];
        const toIdx = (sec:number): number => { let best=0, diff=Number.POSITIVE_INFINITY; for (let i=0;i<bars.length;i++){ const s = typeof bars[i].time === 'number' ? (bars[i].time as number) : Math.floor(Date.UTC(bars[i].time.year, bars[i].time.month-1, bars[i].time.day)/1000); const d=Math.abs(s-sec); if (d<diff){best=i; diff=d; if(d===0) break;} } return best; };
        const i1 = toIdx(a); const i2 = toIdx(b);
        selIdxRef.current = { i1: Math.min(i1,i2), i2: Math.max(i1,i2) };
        const barToIso = (bar:any): string => { if (typeof bar.time === 'number') { const sTr=Math.floor((bar.time as number)/60)*60; const d=new Date(sTr*1000); return d.toISOString().slice(0,16)+':00Z'; } const d=new Date(Date.UTC(bar.time.year, bar.time.month-1, bar.time.day)); return d.toISOString().slice(0,10); };
        const rs = barToIso(bars[selIdxRef.current.i1!]);
        const re = barToIso(bars[selIdxRef.current.i2!]);
        setRange({ start: rs, end: re });
        updateHighlight();
      }
    });

    // Build overlay for selected range
    if (containerRef.current && !highlightRef.current) {
      const ov = document.createElement('div');
      ov.style.position = 'absolute';
      // Let vertical bounds be computed dynamically
      ov.style.top = '0';
      ov.style.bottom = '0';
      ov.style.left = '0';
      ov.style.width = '0';
      ov.style.background = 'rgba(37,99,235,0.10)';
      ov.style.border = '1px solid rgba(37,99,235,0.45)';
      ov.style.pointerEvents = 'none';
      ov.style.display = 'none';
      containerRef.current.appendChild(ov);
      highlightRef.current = ov;
    }

    // Update on visible time range change (pan/zoom)
    const visHandler = () => updateHighlight();
    chart.timeScale().subscribeVisibleTimeRangeChange(visHandler);

    return () => {
      window.removeEventListener('resize', resize);
      try { chart.timeScale().unsubscribeVisibleTimeRangeChange(visHandler); } catch {}
      highlightRef.current?.remove();
      highlightRef.current = null;
      chart.remove();
    };
  }, []);

  // Fetch current user
  useEffect(() => {
    let cancelled = false;
    fetch('/api/auth/me', { cache: 'no-store' })
      .then(async r => {
        if (!r.ok) { if (!cancelled) { setUserEmail(null); setIsLoggedIn(false); try { localStorage.removeItem('finlab_logged_in'); } catch {} } return; }
        const j = await r.json();
        if (!cancelled) { setUserEmail(j?.email || null); setIsLoggedIn(true); try { localStorage.setItem('finlab_logged_in','1'); } catch {} }
      })
      .catch(() => {})
      .finally(() => { if (!cancelled) setAuthReady(true); });
    return () => { cancelled = true; };
  }, []);

  // Load/reset preferences based on auth status
  useEffect(() => {
    let cancelled = false;
    const resetDefaults = () => {
      if (cancelled) return;
      setSymbols(defaultSymbols);
      setSymbol(defaultSymbols[0] || '');
      setShowMA20(false);
      setShowMA50(false);
      setShowVOL(false);
      setCanPersist(false);
    };
    if (!authReady) { return; }
    if (!isLoggedIn) { // guest: reset on each visit/reload
      resetDefaults();
      return;
    }
    // logged-in: first apply local cached prefs (warm start), then fetch from server
    try {
      const rawSyms = localStorage.getItem('finlab_symbols');
      if (rawSyms) {
        const arr = JSON.parse(rawSyms);
        if (Array.isArray(arr) && arr.every((x:any)=>typeof x==='string')) {
          const uniq = Array.from(new Set(arr));
          setSymbols(uniq.length ? uniq : defaultSymbols);
          const chosen = localStorage.getItem('finlab_symbol_selected');
          if (chosen && uniq.includes(chosen)) setSymbol(chosen);
          else if (uniq.length && !uniq.includes(symbol)) setSymbol(uniq[0]);
        }
      }
      const rawInd = localStorage.getItem('finlab_indicators');
      if (rawInd) {
        const obj = JSON.parse(rawInd || 'null');
        if (obj && typeof obj === 'object') {
          setShowMA20(!!obj.ma20); setShowMA50(!!obj.ma50); setShowVOL(!!obj.vol);
        }
      }
    } catch {}
    (async () => {
      try {
        const r = await fetch('/api/user/prefs', { cache: 'no-store' });
        if (!r.ok) { /* keep local cached values to avoid flicker */ return; }
        const j = await r.json();
        if (cancelled) return;
        const syms = Array.isArray(j.symbols) ? j.symbols.filter((x:any) => typeof x === 'string') : [];
        const uniq = Array.from(new Set(syms));
        setSymbols(uniq.length ? uniq : defaultSymbols);
        if (uniq.length) {
          if (!uniq.includes(symbol)) setSymbol(uniq[0]);
        } else {
          setSymbol(defaultSymbols[0] || '');
        }
        const ind = (j.indicators || {}) as any;
        setShowMA20(!!ind.ma20);
        setShowMA50(!!ind.ma50);
        setShowVOL(!!ind.vol);
      } catch { /* ignore to keep current state */ }
      finally { if (!cancelled) setCanPersist(true); }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLoggedIn, authReady]);

  // Persist preferences when logged in and after initial load
  useEffect(() => {
    if (!isLoggedIn || !canPersist) return; // do not persist until ready
    const payload = { symbols, indicators: { ma20: showMA20, ma50: showMA50, vol: showVOL } };
    try {
      fetch('/api/user/prefs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }).catch(()=>{});
    } catch {}
  }, [symbols, showMA20, showMA50, showVOL, isLoggedIn]);

  // Also cache indicators locally when logged in (warm start to avoid flicker on next reload)
  useEffect(() => {
    if (!isLoggedIn || !canPersist) return;
    try { localStorage.setItem('finlab_indicators', JSON.stringify({ ma20: showMA20, ma50: showMA50, vol: showVOL })); } catch {}
  }, [showMA20, showMA50, showVOL, isLoggedIn, canPersist]);

  // Cache symbols and selected symbol locally when logged in for warm start
  useEffect(() => {
    if (!isLoggedIn || !canPersist) return;
    try { localStorage.setItem('finlab_symbols', JSON.stringify(symbols)); } catch {}
  }, [symbols, isLoggedIn, canPersist]);

  useEffect(() => {
    if (!isLoggedIn || !canPersist) return;
    try { if (symbol) localStorage.setItem('finlab_symbol_selected', symbol); } catch {}
  }, [symbol, isLoggedIn, canPersist]);

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

  // Fetch news when symbol or universe changes
  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      if (!symbol.trim()) {
        setNewsSym([]);
        setNewsMkt([]);
        return;
      }
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

  // Toggle indicator visibility. For volume, show left axis and pin to bottom via left scale margins
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    if (ma20Ref.current) ma20Ref.current.applyOptions({ visible: showMA20 });
    if (ma50Ref.current) ma50Ref.current.applyOptions({ visible: showMA50 });
    if (volRef.current) volRef.current.applyOptions({ visible: showVOL });
    chart.applyOptions({ leftPriceScale: { visible: showVOL, borderColor: '#cbd5e1' } });
    const left = chart.priceScale('left');
    left?.applyOptions({ scaleMargins: showVOL ? { top: 0.75, bottom: 0 } : { top: 0.1, bottom: 0.1 } });
  }, [showMA20, showMA50, showVOL]);

  useEffect(() => {
    if (!seriesRef.current || !chartRef.current) return;
    let cancelled = false;
    const tick = () => {
      setError(null);
      if (!symbol.trim()) return;
      fetchOhlcv(symbol, tf)
        .then(pack => { if (!cancelled) {
          seriesRef.current?.setData(pack.price);
          dataRef.current = pack.price as any;
          ma20DataRef.current = pack.ma20 as any;
          ma50DataRef.current = pack.ma50 as any;
          volDataRef.current = pack.vol as any;
          if (volRef.current) volRef.current.setData(pack.vol);
          if (ma20Ref.current) ma20Ref.current.setData(pack.ma20);
          if (ma50Ref.current) ma50Ref.current.setData(pack.ma50);
          updateHighlight(); /* do not refit on every refresh */
          applyBtMarkers(btReport);
        } })
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

  // Crosshair/time label: format with Asia/Taipei
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    const isDaily = tf === '1d';
    const fmtDate = new Intl.DateTimeFormat('zh-TW', { timeZone: 'Asia/Taipei', year: 'numeric', month: '2-digit', day: '2-digit' });
    const fmtDateTime = new Intl.DateTimeFormat('zh-TW', { timeZone: 'Asia/Taipei', year: 'numeric', month: '2-digit', day: '2-digit', hour12: false, hour: '2-digit', minute: '2-digit' });
    const timeFormatter = (time: any) => {
      let ms: number | null = null;
      if (typeof time === 'number') ms = time * 1000;
      else if (time && typeof time === 'object' && 'year' in time) ms = Date.UTC(time.year as number, (time.month as number) - 1, time.day as number);
      if (ms == null) return '';
      const d = new Date(ms);
      const s = isDaily ? fmtDate.format(d) : fmtDateTime.format(d);
      // unify separators to YYYY-MM-DD (or YYYY-MM-DD HH:mm)
      return s.replaceAll('/', '-');
    };
    chart.applyOptions({ localization: { locale: 'zh-TW' as any, timeFormatter: timeFormatter as any } });
  }, [tf]);

  // Ensure x-axis shows appropriate units when zooming (Asia/Taipei)
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    const isDaily = tf === '1d';
    const fmtTick = (time: any, markType?: TickMarkType) => {
      // time can be UTCTimestamp (number) or BusinessDay-like object
      let ms: number;
      if (typeof time === 'number') {
        ms = time * 1000;
      } else if (time && typeof time === 'object' && 'year' in time) {
        ms = Date.UTC(time.year as number, (time.month as number) - 1, time.day as number);
      } else {
        return '';
      }
      const d = new Date(ms);
      // Keep intraday labels fixed to M/D HH:mm regardless of zoom
      if (!isDaily) {
        const md = new Intl.DateTimeFormat('zh-TW', { timeZone: 'Asia/Taipei', month: 'numeric', day: 'numeric' }).format(d).replace(/\s/g, '');
        const hm = new Intl.DateTimeFormat('zh-TW', { timeZone: 'Asia/Taipei', hour12: false, hour: '2-digit', minute: '2-digit' }).format(d);
        return `${md} ${hm}`;
      }
      if (markType === TickMarkType.Year) {
        return new Intl.DateTimeFormat('zh-TW', { timeZone: 'Asia/Taipei', year: 'numeric' }).format(d);
      }
      if (markType === TickMarkType.Month) {
        // Avoid Year/Month style to prevent confusion with Month/Day
        return new Intl.DateTimeFormat('zh-TW', { timeZone: 'Asia/Taipei', month: 'numeric' }).format(d);
      }
      return new Intl.DateTimeFormat('zh-TW', { timeZone: 'Asia/Taipei', month: 'numeric', day: 'numeric' }).format(d).replace(/\s/g, '');
    };
    chart.applyOptions({
      timeScale: {
        timeVisible: !isDaily,
        secondsVisible: false, // fixed minutes display for intraday
        tickMarkFormatter: fmtTick as any,
      },
    });
  }, [tf]);

  // Helper: convert ISO to Time based on tf
  const isoToTime = (iso: string): any => {
    // Supports either date-only (YYYY-MM-DD) or full ISO with time
    const hasTime = iso.includes('T');
    const d = hasTime ? new Date(iso) : new Date(iso + 'T00:00:00Z');
    if (tf === '1d' || (!hasTime && tf === '1d')) {
      return { year: d.getUTCFullYear(), month: d.getUTCMonth()+1, day: d.getUTCDate() } as any;
    }
    return Math.floor(d.getTime()/1000) as UTCTimestamp;
  };

  // Update overlay box with vertical bounds from selected data range
  const updateHighlight = (r?: {start?: string; end?: string}) => {
    const ov = highlightRef.current;
    const chart = chartRef.current;
    const series = seriesRef.current;
    if (!ov || !chart || !series) return;
    const bars = (dataRef.current || []) as any[];
    if (bars.length && selIdxRef.current.i1 != null && selIdxRef.current.i2 != null) {
      const i1 = Math.max(0, Math.min(bars.length-1, selIdxRef.current.i1!));
      const i2 = Math.max(0, Math.min(bars.length-1, selIdxRef.current.i2!));
      const a = Math.min(i1,i2), b = Math.max(i1,i2);
      const x1c = chart.timeScale().timeToCoordinate(bars[a].time);
      const x2c = chart.timeScale().timeToCoordinate(bars[b].time);
      if (x1c == null || x2c == null) { ov.style.display='none'; return; }
      let leftOffset = 0; try { const lps:any=(chart as any).priceScale? (chart as any).priceScale('left'):null; const w=lps&&typeof lps.width==='function'? lps.width():0; if(Number.isFinite(w)) leftOffset=w; } catch {}
      const coord = (idx:number)=> chart.timeScale().timeToCoordinate(bars[idx].time)!;
      const half1 = a>0? Math.abs(coord(a)-coord(a-1))/2 : (a+1<bars.length? Math.abs(coord(a+1)-coord(a))/2:2);
      const half2 = b+1<bars.length? Math.abs(coord(b+1)-coord(b))/2 : (b>0? Math.abs(coord(b)-coord(b-1))/2:2);
      // Align box to centers of first and last selected bars
      const left = Math.min(x1c, x2c) + leftOffset;
      const width = Math.max(2, Math.abs(x2c - x1c));
      let hi=-Infinity, lo=Infinity; for(let i=a;i<=b;i++){ const d=bars[i]; if(d.high>hi) hi=d.high; if(d.low<lo) lo=d.low; }
      if (!Number.isFinite(hi) || !Number.isFinite(lo)) { ov.style.display='none'; return; }
      const pad=Math.max((hi-lo)*0.04,(hi+lo)*0.0005); const yTop=series.priceToCoordinate(hi+pad); const yBot=series.priceToCoordinate(lo-pad);
      if (yTop==null||yBot==null) { ov.style.display='none'; return; }
      const top=Math.min(yTop,yBot); const bottomCoord=Math.max(yTop,yBot); const containerH=containerRef.current?.clientHeight ?? 0; const bottomPx=Math.max(0, containerH-bottomCoord);
      ov.style.display='block';
      ov.style.left = `${Math.round(left)}px`;
      ov.style.width = `${Math.round(width)}px`;
      ov.style.top = `${Math.round(top)}px`;
      ov.style.bottom = `${Math.round(bottomPx)}px`;
      return;
    }
    const rs = (r?.start ?? range.start);
    const re = (r?.end ?? range.end);
    if (!rs || !re) { ov.style.display = 'none'; return; }
    try {
      const ts1 = isoToTime(rs);
      const ts2 = isoToTime(re);
      const x1 = chart.timeScale().timeToCoordinate(ts1);
      const x2 = chart.timeScale().timeToCoordinate(ts2);
      if (x1 == null || x2 == null) { ov.style.display = 'none'; return; }
      // Adjust for left price scale width when visible (e.g., volume shown)
      let leftOffset = 0;
      try {
        const lps: any = (chart as any).priceScale ? (chart as any).priceScale('left') : null;
        const w = lps && typeof lps.width === 'function' ? lps.width() : 0;
        if (Number.isFinite(w)) leftOffset = w;
      } catch {}
      // Align box to centers of the two selected times
      const left = Math.min(x1, x2) + leftOffset;
      const width = Math.max(2, Math.abs(x2 - x1));

      // compute vertical bounds
      const parseMs = (s: string|undefined): number => {
        if (!s) return NaN as any;
        return new Date(s.includes('T') ? s : (s + 'T00:00:00Z')).getTime();
      };
      const sMs = parseMs(rs);
      const eMs = parseMs(re);
      let hi = Number.NEGATIVE_INFINITY, lo = Number.POSITIVE_INFINITY;
      for (const d of (dataRef.current || [])) {
        let tms: number | null = null;
        if (typeof d.time === 'number') tms = d.time * 1000;
        else if (d.time && typeof d.time === 'object' && 'year' in d.time) tms = Date.UTC(d.time.year, d.time.month-1, d.time.day);
        if (tms == null) continue;
        if (tms >= sMs && tms <= eMs) {
          if (d.high > hi) hi = d.high;
          if (d.low < lo) lo = d.low;
        }
      }
      if (!Number.isFinite(hi) || !Number.isFinite(lo)) { ov.style.display = 'none'; return; }
      const pad = Math.max((hi - lo) * 0.04, (hi + lo) * 0.0005);
      const yTop = series.priceToCoordinate(hi + pad);
      const yBot = series.priceToCoordinate(lo - pad);
      if (yTop == null || yBot == null) { ov.style.display = 'none'; return; }
      const top = Math.min(yTop, yBot);
      const bottomCoord = Math.max(yTop, yBot);
      const containerH = containerRef.current?.clientHeight ?? 0;
      const bottomPx = Math.max(0, containerH - bottomCoord);

      ov.style.display = 'block';
      ov.style.left = `${Math.round(left)}px`;
      ov.style.width = `${Math.round(width)}px`;
      ov.style.top = `${Math.round(top)}px`;
      ov.style.bottom = `${Math.round(bottomPx)}px`;
    } catch {
      ov.style.display = 'none';
    }
  };

  // Recompute highlight when range or tf changes
  useEffect(() => { updateHighlight(); }, [range.start, range.end, tf]);

  // Render mini chart for selected similar item
  useEffect(() => {
    if (!simOpen) return;
    if (!simContainerRef.current) return;
    if (!simItems || simItems.length === 0) return;
    const item = simItems[Math.min(simIndex, simItems.length-1)];
    if (!item) return;

    // init chart once per mount
    let chart = simChartRef.current;
    if (!chart) {
      chart = createChart(simContainerRef.current, {
        height: 320,
        layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: '#334155' },
        grid: { vertLines: { color: '#e2e8f0' }, horzLines: { color: '#e2e8f0' } },
        rightPriceScale: { borderColor: '#cbd5e1' },
        timeScale: { borderColor: '#cbd5e1' },
      });
      simChartRef.current = chart;
      simSeriesRef.current = chart.addCandlestickSeries({ upColor:'#16a34a', downColor:'#ef4444', wickUpColor:'#16a34a', wickDownColor:'#ef4444', borderUpColor:'#16a34a', borderDownColor:'#ef4444' });
    }

    // apply intraday x-axis formatting similar to main chart
    try {
      const isDaily = tfRef.current === '1d';
      const fmtTick = (time: any, markType?: TickMarkType) => {
        let ms: number;
        if (typeof time === 'number') {
          ms = time * 1000;
        } else if (time && typeof time === 'object' && 'year' in time) {
          ms = Date.UTC(time.year as number, (time.month as number) - 1, time.day as number);
        } else {
          return '';
        }
        const d = new Date(ms);
        if (!isDaily) {
          const md = new Intl.DateTimeFormat('zh-TW', { timeZone: 'Asia/Taipei', month: 'numeric', day: 'numeric' }).format(d).replace(/\s/g, '');
          const hm = new Intl.DateTimeFormat('zh-TW', { timeZone: 'Asia/Taipei', hour12: false, hour: '2-digit', minute: '2-digit' }).format(d);
          return `${md} ${hm}`;
        }
        return new Intl.DateTimeFormat('zh-TW', { timeZone: 'Asia/Taipei', month: 'numeric', day: 'numeric' }).format(d).replace(/\s/g, '');
      };
      chart.applyOptions({
        timeScale: {
          borderColor: '#cbd5e1',
          timeVisible: tfRef.current !== '1d',
          secondsVisible: false,
          tickMarkFormatter: fmtTick as any,
        },
      });
    } catch {}

    // fetch data
    setError(null);
    fetchSlice(item.symbol, item.start_time, item.end_time, tfRef.current)
      .then(data => {
        simSeriesRef.current?.setData(data);
        simChartRef.current?.timeScale().fitContent();
      })
      .catch(() => {/* swallow modal errors */});

    // cleanup when modal closes
    return () => {
      // keep chart for reuse while modal open; destroyed when modal closes by outer effect
    };
  }, [simOpen, simIndex, simItems]);

  // Destroy modal chart when closing
  useEffect(() => {
    if (!simOpen && simChartRef.current) {
      simChartRef.current.remove();
      simChartRef.current = null;
      simSeriesRef.current = null;
    }
  }, [simOpen]);

  // Apply backtest markers to the candlestick series
  const applyBtMarkers = (rep: any | null) => {
    const series = seriesRef.current;
    const chart = chartRef.current;
    if (!series || !chart) return;
    if (!rep || !Array.isArray(rep.events) || rep.events.length === 0) {
      // If no raw events, try to derive from trades_detail
      if (!Array.isArray(rep?.trades_detail) || rep.trades_detail.length === 0) {
        try { series.setMarkers([] as any); } catch {}
        return;
      }
    }
    // Choose source: prefer trades_detail to ensure state-valid entries
    const srcEvents: Array<any> = Array.isArray(rep.trades_detail) && rep.trades_detail.length > 0
      ? rep.trades_detail.flatMap((t:any) => {
          const out: any[] = [];
          if (t.entry_ts) out.push({ ts: t.entry_ts, type: t.carried ? 'carried' : 'buy', price: t.entry_price });
          if (t.exit_ts) out.push({ ts: t.exit_ts, type: 'sell', price: t.exit_price });
          return out;
        })
      : (rep.events || []);

    // helper to get ms from chart bar time
    const barMs = (t: any): number | null => {
      if (typeof t === 'number') return t * 1000;
      if (t && typeof t === 'object' && 'year' in t) return Date.UTC(t.year as number, (t.month as number)-1, t.day as number);
      return null;
    };
    const bars = (dataRef.current || []) as Array<any>;
    const markers = srcEvents.map((ev: any) => {
      const kind = (ev.type || ev.side);
      const isBuy = kind === 'buy' || kind === 'carried';
      const iso = String(ev.ts || '').slice(0,10);
      let timeForMarker: any = null;
      if (tf === '1d') {
        timeForMarker = isoToTime(iso);
      } else {
        const startMs = new Date(iso + 'T00:00:00Z').getTime();
        const endMs = startMs + 24*60*60*1000 - 1;
        // pick last bar within that UTC day (closest to session close)
        let chosen: any = null;
        for (let i = bars.length - 1; i >= 0; i--) {
          const ms = barMs(bars[i].time);
          if (ms != null && ms >= startMs && ms <= endMs) { chosen = bars[i].time; break; }
        }
        timeForMarker = chosen ?? isoToTime(iso);
      }
      return {
        time: timeForMarker,
        position: isBuy ? 'belowBar' as const : 'aboveBar' as const,
        color: isBuy ? '#16a34a' : '#ef4444',
        shape: isBuy ? 'arrowUp' as const : 'arrowDown' as const,
        text: (kind === 'carried' ? '承接 ' : (isBuy ? '買入 ' : '賣出 ')) + (typeof ev.price === 'number' ? String((ev.price as number).toFixed(2)) : ''),
      };
    });
    try { series.setMarkers(markers as any); } catch {}
  };

  // Poll backtest status when a job is created
  useEffect(() => {
    if (!btJob) return;
    let cancelled = false;
    setBtStatus('queued');
    const poll = async () => {
      try {
        const r = await fetch(`/api/backtest/status?job_id=${encodeURIComponent(btJob)}`, { cache: 'no-store' });
        if (!r.ok) return;
        const j = await r.json();
        if (cancelled) return;
        setBtStatus(j.status);
        if (j.status === 'finished' || j.status === 'failed') {
          try {
            const r2 = await fetch(`/api/backtest/report?job_id=${encodeURIComponent(btJob)}`, { cache: 'no-store' });
            const j2 = await r2.json();
            if (!cancelled) { setBtReport(j2.report || { error: 'no report' }); applyBtMarkers(j2.report); }
          } catch {}
          clearInterval(id);
        }
      } catch {}
    };
    poll();
    const id = setInterval(poll, 1000);
    return () => { cancelled = true; clearInterval(id); };
  }, [btJob]);

  // Re-apply markers if timeframe changes (daily uses BusinessDay mapping)
  useEffect(() => { applyBtMarkers(btReport); }, [btReport, tf]);

  const onFindSimilar = async () => {
    if (!range.start || !range.end) {
      alert('請先在圖上點兩下選取起訖（第一次點＝起點；第二次點＝終點）');
      return;
    }
    // 動態決定 m：不限制天數，依選取區間長度自適應（最少3，最多60）
    let m = 30;
    try {
      const start = new Date(range.start);
      const end = new Date(range.end);
      const approxDays = Math.max(0, Math.round((+end - +start) / 86400000));
      if (Number.isFinite(approxDays) && approxDays > 0) {
        m = Math.max(3, Math.min(60, approxDays - 1));
      }
    } catch {}
    try {
      // Call same-origin proxy to avoid extension/CORS issues
      const url = `/api/similar/search`;
      const res = await fetch(url, {
        method: 'POST',
        headers: {'Content-Type':'application/json','Accept':'application/json'},
        body: JSON.stringify({ symbol, start: range.start, end: range.end, m, top: 5, universe: symbols, tf })
      });
      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(`POST ${url} failed: HTTP ${res.status}${text ? ` - ${text}` : ''}`);
      }
      const j = await res.json();
      setSimError(null);
      const items = Array.isArray(j.items) ? j.items : [];
      setSimItems(items);
      setSimIndex(0);
      setSimOpen(true);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setSimItems([]);
      setSimError('呼叫相似搜尋失敗：\n' + msg + '\n\n建議：\n1) 確認 API 已啟動：http://localhost:8000/health\n2) 停用可能攔截請求的瀏覽器外掛（或改用無痕視窗）\n3) 若前端非 3000 埠，請調整 API CORS 設定');
      setSimOpen(true);
    }
  };

  return (
    <main className="app" suppressHydrationWarning>
      <div className="header" suppressHydrationWarning>
        <h1 className="title">FinLab</h1>
        <div style={{marginLeft:'auto', display:'flex', gap:8, alignItems:'center'}}>
          {isLoggedIn ? (
            <>
              <span className="badge" title={userEmail || ''}>{userEmail || '…'}</span>
              <button className="btn" onClick={async()=>{
                try { await fetch('/api/auth/logout',{method:'POST'}); } catch {}
                // Reset local UI state when logging out
                setUserEmail(null);
                setIsLoggedIn(false);
                setSymbols(defaultSymbols);
                setSymbol(defaultSymbols[0] || '');
                setShowMA20(false);
                setShowMA50(false);
                setShowVOL(false);
                // Keep local caches for seamless restore on next login; only drop the logged-in marker
                try { localStorage.removeItem('finlab_logged_in'); } catch {}
              }}>登出</button>
            </>
          ) : (
            <>
              <a className="btn" href="/login">登入</a>
              <a className="btn" href="/register">註冊</a>
            </>
          )}
        </div>
      </div>
      {simOpen && (
        <div className="modal-overlay" onMouseDown={(e)=>{ if (e.target === e.currentTarget) setSimOpen(false); }}>
          <div className="modal" role="dialog" aria-modal="true" aria-label="相似片段結果">
            <div className="modal-header">相似片段 {simError ? '' : `Top ${Math.min(5, simItems.length)}`}</div>
            <div className="modal-body">
              {simError ? (
                <div style={{whiteSpace:'pre-wrap'}}>{simError}</div>
              ) : (
                <div className="modal-grid">
                  <div>
                    <ul className="modal-list">
                      {simItems.slice(0,5).map((x:any, i:number)=> (
                        <li key={i} className={`modal-list-item ${i===simIndex ? 'active':''}`} onMouseDown={(e)=>{e.preventDefault(); setSimIndex(i);}}>
                          <div>
                            <strong style={{marginRight:8}}>{i+1}.</strong>
                            <span style={{fontWeight:600}}>{x.symbol}</span>
                            <span style={{marginLeft:8, color:'#64748b'}}>{(x.start_time||'').slice(0,10)} ~ {(x.end_time||'').slice(0,10)}</span>
                          </div>
                          <div style={{whiteSpace:'nowrap', color:'#334155'}}>d={Number(x.distance).toFixed(3)}</div>
                        </li>
                      ))}
                      {simItems.length === 0 && <li className="modal-list-item">— 無結果 —</li>}
                    </ul>
                  </div>
                  <div>
                    <div ref={simContainerRef} className="mini-chart" />
                  </div>
                </div>
              )}
            </div>
            <div className="modal-footer">
              <button className="btn" onClick={()=>setSimOpen(false)}>關閉</button>
            </div>
          </div>
        </div>
      )}
      <div className="toolbar" suppressHydrationWarning>
        <span className="label">代碼：</span>
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
        <span className="label">指標：</span>
        <label className="check" suppressHydrationWarning>
          <input type="checkbox" checked={showMA20} onChange={e=>setShowMA20(e.target.checked)} /> MA20
        </label>
        <label className="check" suppressHydrationWarning>
          <input type="checkbox" checked={showMA50} onChange={e=>setShowMA50(e.target.checked)} /> MA50
        </label>
        <label className="check" suppressHydrationWarning>
          <input type="checkbox" checked={showVOL} onChange={e=>setShowVOL(e.target.checked)} /> 成交量
        </label>
        <button className="btn" onClick={onFindSimilar}>找相似</button>
        <button className="btn" onClick={()=>{
          if (!range.start || !range.end) { alert('請先在圖上點兩下選取回測期間'); return; }
          setBtOpen(true);
          setBtReport(null);
          setBtStatus(null);
          setBtJob(null);
        }}>回測</button>
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
        {(() => {
          const fmtDate = new Intl.DateTimeFormat('zh-TW', { timeZone: 'Asia/Taipei', year: 'numeric', month: '2-digit', day: '2-digit' });
          const fmtDateTime = new Intl.DateTimeFormat('zh-TW', { timeZone: 'Asia/Taipei', year: 'numeric', month: '2-digit', day: '2-digit', hour12: false, hour: '2-digit', minute: '2-digit' });
          const show = (s?: string | null) => {
            if (!s) return '—';
            if (s.includes('T')) { const d = new Date(s); return fmtDateTime.format(d).replaceAll('/', '-'); }
            const d = new Date(s + 'T00:00:00Z'); return fmtDate.format(d).replaceAll('/', '-');
          };
          return (
            <>
              TF：{tf}　|　範圍：{rangeLabel || '—'}　|　選取區間：{show(range.start)} ~ {show(range.end)}
            </>
          );
        })()}
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
      {btOpen && (
        <div className="modal-overlay" onMouseDown={(e)=>{ if (e.target === e.currentTarget) setBtOpen(false); }}>
          <div className="modal" role="dialog" aria-modal="true" aria-label="回測設定">
            <div className="modal-header">回測設定（{symbol}：{range.start} ~ {range.end}）</div>
            <div className="modal-body">
              <div style={{display:'grid', gridTemplateColumns:'repeat(3, minmax(0,1fr))', gap:12}}>
                <label className="form-label">初始資金
                  <input className="input" type="number" min={1000} step={1000} value={btCash} onChange={e=>setBtCash(Number(e.target.value)||0)} />
                </label>
                <label className="form-label">快線(天)
                  <input className="input" type="number" min={2} step={1} value={btFast} onChange={e=>setBtFast(Number(e.target.value)||0)} />
                </label>
                <label className="form-label">慢線(天)
                  <input className="input" type="number" min={3} step={1} value={btSlow} onChange={e=>setBtSlow(Number(e.target.value)||0)} />
                </label>
              </div>
              {btJob && (
                <div className="info">任務 {btJob} 狀態：{btStatus || '查詢中…'}</div>
              )}
              {btReport && !btReport.error && (
                <div className="sent-card" style={{marginTop:12}}>
                  <div className="sent-title">回測結果</div>
                  <div className="sent-metrics">
                    <div className="sent-kv"><span>最終資金</span><span>{(btReport.final_equity||0).toLocaleString('zh-TW', {maximumFractionDigits:0})}</span></div>
                    <div className="sent-kv"><span>總報酬</span><span>{btReport.total_return!=null ? Math.round(btReport.total_return*100) : '—'}%</span></div>
                    <div className="sent-kv"><span>最大回撤</span><span>{btReport.max_drawdown!=null ? Math.round(btReport.max_drawdown*100) : '—'}%</span></div>
                    <div className="sent-kv"><span>交易次數</span><span>{btReport.trades ?? '—'}</span></div>
                  </div>
                </div>
              )}
              {btReport && btReport.error && (
                <div className="alert">回測失敗：{btReport.error}</div>
              )}
            </div>
            <div className="modal-footer">
              <button className="btn" onClick={()=>setBtOpen(false)}>關閉</button>
              <button className="btn btn-primary" onClick={async()=>{
                if (!symbol.trim() || !range.start || !range.end) { alert('請先選擇代碼與回測期間'); return; }
                if (btFast <= 0 || btSlow <= 0 || btFast >= btSlow) { alert('請確認快線/慢線（快線需小於慢線）'); return; }
                try {
                  setBtReport(null); setBtStatus('queued'); setBtJob(null);
                  const res = await fetch('/api/backtest/submit', {
                    method: 'POST', headers: { 'Content-Type':'application/json' },
                    body: JSON.stringify({ config: { symbol, start: range.start, end: range.end, cash: btCash, strategy: 'sma_cross', params: { fast: btFast, slow: btSlow } } })
                  });
                  if (!res.ok) { const t = await res.text(); throw new Error(t || 'submit failed'); }
                  const j = await res.json();
                  const jid = j.job_id as string;
                  setBtJob(jid);
                } catch (e:any) {
                  setBtReport({ error: e?.message || String(e) });
                }
              }}>開始回測</button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
