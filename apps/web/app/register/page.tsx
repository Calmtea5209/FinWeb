'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';

export default function RegisterPage() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null); setLoading(true);
    try {
      const r = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      if (!r.ok) {
        const txt = await r.text().catch(()=> '');
        throw new Error(txt || '註冊失敗');
      }
      // After register, auto login
      const r2 = await fetch('/api/auth/login', {
        method: 'POST', headers: { 'Content-Type':'application/json' }, body: JSON.stringify({ email, password })
      });
      if (!r2.ok) throw new Error('自動登入失敗，請手動登入');
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
            <span className="auth-title">建立帳號</span>
          </div>
          <div className="auth-subtitle">快速開始，加入 FinWeb</div>
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
          <button className="btn btn-primary auth-submit" disabled={loading} type="submit">{loading ? '處理中…' : '註冊'}</button>
        </form>
        <div className="auth-footer">
          <span className="muted">已經有帳號？</span>
          <Link href="/login" className="auth-link">去登入</Link>
        </div>
      </div>
    </main>
  );
}
