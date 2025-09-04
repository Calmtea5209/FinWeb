export const dynamic = 'force-dynamic';

async function getUpstream(pathWithQs: string) {
  const envApi = process.env.NEXT_PUBLIC_API_URL || '';
  const port = process.env.API_PORT || '8000';
  const bases = [envApi, `http://api:${port}`, 'http://api:8000', `http://localhost:${port}`, 'http://localhost:8000']
    .filter((x) => typeof x === 'string' && x.length > 0);
  let lastErr: any = null;
  for (const base of bases) {
    try {
      const r = await fetch(`${base}${pathWithQs}`, { method: 'GET', headers: { 'Accept': 'application/json' }, cache: 'no-store' });
      return r;
    } catch (e) { lastErr = e; }
  }
  throw lastErr || new Error('proxy error');
}

export async function GET(req: Request) {
  const url = new URL(req.url);
  const qs = url.search || '';
  try {
    const upstream = await getUpstream(`/backtest/report${qs}`);
    const text = await upstream.text();
    return new Response(text, { status: upstream.status, headers: { 'Content-Type': upstream.headers.get('content-type') || 'application/json' } });
  } catch (e: any) {
    return new Response(JSON.stringify({ detail: e?.message || 'proxy error' }), { status: 502, headers: { 'Content-Type': 'application/json' } });
  }
}

