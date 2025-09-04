export const dynamic = 'force-dynamic';

function cookieAttrs(maxAgeSec: number) {
  const attrs = [
    'Path=/',
    'HttpOnly',
    'SameSite=Lax',
    `Max-Age=${Math.max(0, Math.floor(maxAgeSec))}`,
  ];
  if (process.env.NODE_ENV === 'production') attrs.push('Secure');
  return attrs.join('; ');
}

export async function POST(req: Request) {
  const api = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  let payload: any = null;
  try { payload = await req.json(); } catch {}
  try {
    const r = await fetch(`${api}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify(payload ?? {}),
      cache: 'no-store',
    });
    const text = await r.text();
    let headers: Record<string,string> = { 'Content-Type': r.headers.get('content-type') || 'application/json' };
    try {
      if (r.ok) {
        const j = JSON.parse(text);
        const token = j?.access_token;
        // default to 7d if missing
        const maxAge = 60 * 60 * 24 * 7;
        if (token && typeof token === 'string') {
          headers['Set-Cookie'] = `auth_token=${encodeURIComponent(token)}; ${cookieAttrs(maxAge)}`;
        }
      }
    } catch {}
    return new Response(text, { status: r.status, headers });
  } catch (e: any) {
    return new Response(JSON.stringify({ detail: e?.message || 'proxy error' }), { status: 502, headers: { 'Content-Type': 'application/json' } });
  }
}

