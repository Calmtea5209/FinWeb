export const dynamic = 'force-dynamic';

export async function POST() {
  const headers: Record<string,string> = {
    'Content-Type': 'application/json',
    // expire immediately
    'Set-Cookie': `auth_token=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0${process.env.NODE_ENV==='production'?'; Secure':''}`,
  };
  return new Response(JSON.stringify({ ok: true }), { status: 200, headers });
}

