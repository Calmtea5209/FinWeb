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

export async function GET(req: Request) {
  const api = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  try {
    const token = readCookie(req, 'auth_token');
    if (!token) return new Response(JSON.stringify({ detail: 'unauthorized' }), { status: 401, headers: { 'Content-Type': 'application/json' } });
    const r = await fetch(`${api}/auth/me`, {
      method: 'GET',
      headers: { 'Accept': 'application/json', 'Authorization': `Bearer ${token}` },
      cache: 'no-store',
    });
    const text = await r.text();
    return new Response(text, { status: r.status, headers: { 'Content-Type': r.headers.get('content-type') || 'application/json' } });
  } catch (e: any) {
    return new Response(JSON.stringify({ detail: e?.message || 'proxy error' }), { status: 502, headers: { 'Content-Type': 'application/json' } });
  }
}

