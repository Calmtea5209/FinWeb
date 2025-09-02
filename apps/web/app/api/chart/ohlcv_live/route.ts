export const dynamic = 'force-dynamic';

export async function GET(req: Request) {
  const api = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  const url = new URL(req.url);
  const qs = url.search || '';
  try {
    const upstream = await fetch(`${api}/chart/ohlcv_live${qs}`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      cache: 'no-store',
    });
    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: { 'Content-Type': upstream.headers.get('content-type') || 'application/json' },
    });
  } catch (e: any) {
    return new Response(JSON.stringify({ detail: e?.message || 'proxy error' }), {
      status: 502,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}

