export const dynamic = 'force-dynamic';

async function fetchWithFallbacks(pathAndQs: string) {
  const envApi = process.env.NEXT_PUBLIC_API_URL || '';
  const port = process.env.API_PORT || '8000';
  const candidates = [
    envApi,
    `http://api:${port}`,
    'http://api:8000',
    `http://localhost:${port}`,
    'http://localhost:8000',
  ].filter((x) => typeof x === 'string' && x.length > 0);
  let lastErr: any = null;
  for (const base of candidates) {
    try {
      const r = await fetch(`${base}${pathAndQs}`, {
        method: 'GET',
        headers: { Accept: 'application/json' },
        cache: 'no-store',
      });
      // Even非200也直接回傳，因為那是上游的回應
      return r;
    } catch (e) {
      lastErr = e;
      continue;
    }
  }
  const err = lastErr instanceof Error ? lastErr.message : 'proxy error';
  throw new Error(err);
}

export async function GET(req: Request) {
  const url = new URL(req.url);
  const qs = url.search || '';
  try {
    const upstream = await fetchWithFallbacks(`/chart/ohlcv_live${qs}`);
    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: {
        'Content-Type': upstream.headers.get('content-type') || 'application/json',
      },
    });
  } catch (e: any) {
    return new Response(JSON.stringify({ detail: e?.message || 'proxy error' }), {
      status: 502,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}
