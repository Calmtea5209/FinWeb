export const dynamic = 'force-dynamic';

export async function POST(req: Request) {
  const api = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  let payload: any = null;
  try {
    payload = await req.json();
  } catch {}
  try {
    const r = await fetch(`${api}/similar/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify(payload ?? {}),
      // Prevent Next from caching and keep it server-side
      cache: 'no-store',
    });
    const text = await r.text();
    return new Response(text, {
      status: r.status,
      headers: { 'Content-Type': r.headers.get('content-type') || 'application/json' },
    });
  } catch (e: any) {
    const msg = e?.message || 'proxy error';
    return new Response(JSON.stringify({ detail: msg }), { status: 502, headers: { 'Content-Type': 'application/json' } });
  }
}

