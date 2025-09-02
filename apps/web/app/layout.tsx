import './globals.css';
export const metadata = { title: 'FinLab Starter', description: 'Next.js + FastAPI' };
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-Hant" suppressHydrationWarning>
      <body suppressHydrationWarning style={{fontFamily:'system-ui, -apple-system, Segoe UI, Roboto, Noto Sans, sans-serif'}}>
        {children}
      </body>
    </html>
  );
}
