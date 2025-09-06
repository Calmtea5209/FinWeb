export const dynamic = 'force-dynamic';

function readCookie(req: Request, name: string): string | null {
  const cookie = req.headers.get('cookie') || '';
  const parts = cookie.split(/;\s*/g);
  for (const p of parts) {
    const [k, v] = p.split('=');
    if (k === name) return decodeURIComponent(v || '');
  }
  return null;
}

function upstreamCandidates(): string[] {
  const envApi = process.env.NEXT_PUBLIC_API_URL || '';
  const port = process.env.API_PORT || '8000';
  return [
    envApi,
    `http://api:${port}`,
    'http://api:8000',
    `http://localhost:${port}`,
    'http://localhost:8000',
    `http://127.0.0.1:${port}`,
    'http://127.0.0.1:8000',
    `http://host.docker.internal:${port}`,
    'http://host.docker.internal:8000',
  ].filter(x => typeof x === 'string' && x.length > 0);
}

async function proxyUpstream(path: string, init: RequestInit): Promise<Response> {
  const tried: string[] = [];
  let lastErr: any = null;
  for (const base of upstreamCandidates()) {
    tried.push(base);
    try {
      const r = await fetch(`${base}${path}`, init);
      const text = await r.text();
      const headers: Record<string, string> = {
        'Content-Type': r.headers.get('content-type') || 'application/json',
        'X-Proxy-Upstream': base,
      };
      return new Response(text, { status: r.status, headers });
    } catch (e) {
      lastErr = e;
      continue;
    }
  }
  const msg = lastErr instanceof Error ? lastErr.message : String(lastErr || 'proxy error');
  return new Response(JSON.stringify({ detail: `fetch failed: ${msg}`, tried }), { status: 502, headers: { 'Content-Type': 'application/json' } });
}

export async function GET(req: Request) {
  const token = readCookie(req, 'auth_token');
  if (!token) return new Response(JSON.stringify({ detail: 'unauthorized' }), { status: 401, headers: { 'Content-Type': 'application/json' } });
  try {
    const r = await proxyUpstream('/user/prefs', {
      method: 'GET',
      headers: { 'Accept': 'application/json', 'Authorization': `Bearer ${token}` },
      cache: 'no-store',
    });
    return r;
  } catch (e: any) {
    return new Response(JSON.stringify({ detail: e?.message || 'proxy error' }), { status: 502, headers: { 'Content-Type': 'application/json' } });
  }
}

export async function POST(req: Request) {
  const token = readCookie(req, 'auth_token');
  if (!token) return new Response(JSON.stringify({ detail: 'unauthorized' }), { status: 401, headers: { 'Content-Type': 'application/json' } });
  let payload: any = null;
  try { payload = await req.json(); } catch {}
  try {
    const r = await proxyUpstream('/user/prefs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json', 'Authorization': `Bearer ${token}` },
      body: JSON.stringify(payload ?? {}),
      cache: 'no-store',
    });
    return r;
  } catch (e: any) {
    return new Response(JSON.stringify({ detail: e?.message || 'proxy error' }), { status: 502, headers: { 'Content-Type': 'application/json' } });
  }
}
