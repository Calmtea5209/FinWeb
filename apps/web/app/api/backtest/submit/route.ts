export const dynamic = 'force-dynamic';

async function postUpstream(path: string, body: any) {
  const envApi = process.env.NEXT_PUBLIC_API_URL || '';
  const port = process.env.API_PORT || '8000';
  const bases = [envApi, `http://api:${port}`, 'http://api:8000', `http://localhost:${port}`, 'http://localhost:8000']
    .filter((x) => typeof x === 'string' && x.length > 0);
  let lastErr: any = null;
  for (const base of bases) {
    try {
      const r = await fetch(`${base}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        body: JSON.stringify(body ?? {}),
        cache: 'no-store',
      });
      return r;
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr || new Error('proxy error');
}

export async function POST(req: Request) {
  let payload: any = null;
  try { payload = await req.json(); } catch {}
  try {
    const upstream = await postUpstream('/backtest/submit', payload);
    const text = await upstream.text();
    return new Response(text, { status: upstream.status, headers: { 'Content-Type': upstream.headers.get('content-type') || 'application/json' } });
  } catch (e: any) {
    return new Response(JSON.stringify({ detail: e?.message || 'proxy error' }), { status: 502, headers: { 'Content-Type': 'application/json' } });
  }
}

