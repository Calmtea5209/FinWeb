'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null); setLoading(true);
    try {
      const r = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      if (!r.ok) {
        const txt = await r.text().catch(()=> '');
        throw new Error(txt || '登入失敗');
      }
      // Warm cache: mark logged-in and prefetch user prefs via token to avoid timing on cookie
      let token: string | null = null;
      try { const j = await r.json(); token = j?.access_token || null; } catch {}
      try { localStorage.setItem('finlab_logged_in','1'); } catch {}
      try {
        if (token && typeof token === 'string') {
          const api = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
          const pr = await fetch(`${api}/user/prefs`, { headers: { 'Accept': 'application/json', 'Authorization': `Bearer ${token}` }, cache: 'no-store' });
          if (pr.ok) {
            const pj = await pr.json();
            const syms = Array.isArray(pj.symbols) ? pj.symbols.filter((x:any)=>typeof x==='string') : [];
            const inds = (pj.indicators || {}) as any;
            // If server has nothing but local cache exists, upload local cache to server immediately
            let needUpload = false;
            try {
              const localSymsRaw = localStorage.getItem('finlab_symbols');
              const localIndsRaw = localStorage.getItem('finlab_indicators');
              const localSyms = localSymsRaw ? JSON.parse(localSymsRaw) : [];
              const localInds = localIndsRaw ? JSON.parse(localIndsRaw) : {};
              const serverEmpty = (!syms || syms.length === 0) && (!inds || (!inds.ma20 && !inds.ma50 && !inds.vol));
              const haveLocal = Array.isArray(localSyms) && localSyms.length > 0;
              if (serverEmpty && haveLocal) {
                needUpload = true;
                await fetch(`${api}/user/prefs`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json', 'Accept': 'application/json', 'Authorization': `Bearer ${token}` },
                  body: JSON.stringify({ symbols: localSyms, indicators: { ma20: !!localInds.ma20, ma50: !!localInds.ma50, vol: !!localInds.vol } }),
                }).catch(()=>{});
              }
            } catch {}
            // Update local cache from server (or keep local if we just uploaded)
            const useSyms = needUpload ? (JSON.parse(localStorage.getItem('finlab_symbols')||'[]')||[]) : syms;
            const useInds = needUpload ? (JSON.parse(localStorage.getItem('finlab_indicators')||'{}')||{}) : inds;
            try { localStorage.setItem('finlab_symbols', JSON.stringify(useSyms)); } catch {}
            if (useSyms?.length) { try { localStorage.setItem('finlab_symbol_selected', useSyms[0]); } catch {} }
            try { localStorage.setItem('finlab_indicators', JSON.stringify(useInds || {})); } catch {}
          }
        }
      } catch {}
      router.push('/');
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    } finally { setLoading(false); }
  };

  return (
    <main className="auth-wrap">
      <div className="auth-card">
        <div className="auth-header">
          <div className="auth-brand">
            <span className="auth-logo">💹</span>
            <span className="auth-title">FinLab</span>
          </div>
          <div className="auth-subtitle">歡迎回來，請登入您的帳號</div>
        </div>

        <form onSubmit={onSubmit} className="auth-form">
          <div className="form-row">
            <label className="form-label">電子郵件</label>
            <input className="input" type="email" value={email} onChange={e=>setEmail(e.target.value)} required placeholder="you@example.com" />
          </div>
          <div className="form-row">
            <label className="form-label">密碼</label>
            <input className="input" type="password" value={password} onChange={e=>setPassword(e.target.value)} required minLength={6} />
          </div>
          {error && <div className="alert" role="alert">{error}</div>}
          <button className="btn btn-primary auth-submit" disabled={loading} type="submit">{loading ? '處理中…' : '登入'}</button>
        </form>

        <div className="auth-footer">
          <span className="muted">還沒有帳號？</span>
          <Link href="/register" className="auth-link">去註冊</Link>
        </div>
      </div>
    </main>
  );
}
