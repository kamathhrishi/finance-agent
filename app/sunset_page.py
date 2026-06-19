"""
Sunset / farewell landing page.

Served at "/" only when the app is running on Railway (the public deployment).
On localhost the normal React landing page is served unchanged. See
`setup_frontend_routes` in app/routes.py for the branch.

The HTML is fully self-contained (inline CSS, fonts from Google Fonts) so it
has no dependency on the React build output — the build can be absent or stale
and this page still renders. The styling mirrors the React landing page
(frontend/src/pages/LandingPage.tsx) one-to-one:
  - action / brand color is navy #0a1628 (NOT the teal/blue in index.css — the
    landing page uses inline bg-[#0a1628] everywhere)
  - fixed 64px white nav bar with the fa-layer-group logo mark in a navy box
  - hero on warm #faf9f7 with a 64px #e2e8f0 grid overlay
  - Playfair Display serif headline, Inter body, slate-500 supporting text
  - dark navy (#0a1628) footer with a slate-800 top border
Slate values match Tailwind: 200 #e2e8f0 · 300 #cbd5e1 · 400 #94a3b8 ·
500 #64748b · 600 #475569 · 800 #1e293b.
"""

# fa-layer-group icon path, lifted verbatim from frontend StrataLensLogo.tsx so
# the nav/footer mark is pixel-identical to the live site.
_LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" '
    'width="{size}" height="{size}" fill="currentColor">'
    '<path d="M12.41 148.02l232.94 105.67c6.8 3.09 14.49 3.09 21.29 0l232.94-105.67c16.55-7.51 16.55-32.52 0-40.03L266.65 2.31a25.607 25.607 0 0 0-21.29 0L12.41 107.98c-16.55 7.51-16.55 32.53 0 40.04zm487.18 88.28l-58.09-26.33-161.64 73.27c-7.56 3.43-15.59 5.17-23.86 5.17s-16.29-1.74-23.86-5.17L70.51 209.97l-58.1 26.33c-16.55 7.5-16.55 32.5 0 40l232.94 105.59c6.8 3.08 14.49 3.08 21.29 0L499.59 276.3c16.55-7.5 16.55-32.5 0-40zm0 127.8l-57.87-26.23-161.86 73.37c-7.56 3.43-15.59 5.17-23.86 5.17s-16.29-1.74-23.86-5.17L70.29 337.87 12.41 364.1c-16.55 7.5-16.55 32.5 0 40l232.94 105.59c6.8 3.08 14.49 3.08 21.29 0L499.59 404.1c16.55-7.5 16.55-32.5 0-40z"/>'
    "</svg>"
)

SUNSET_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>StrataLens</title>
<link rel="icon" href="/favicon.svg" type="image/svg+xml" />
<meta name="robots" content="noindex" />
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap" rel="stylesheet" />
<style>
  :root {{
    --navy: #0a1628;
    --navy-light: #1e293b;
    --slate-800: #1e293b;
    --slate-600: #475569;
    --slate-500: #64748b;
    --slate-400: #94a3b8;
    --slate-300: #cbd5e1;
    --slate-200: #e2e8f0;
    --slate-100: #f1f5f9;
    --slate-50: #f8fafc;
    --warm: #faf9f7;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html {{ scroll-behavior: smooth; }}
  body {{
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background: #ffffff;
    color: var(--slate-500);
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }}
  a {{ color: inherit; }}

  /* ── Fixed nav (mirrors LandingPage <nav>: h-16, white, slate-200 border) ── */
  nav {{
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 50;
    height: 64px;
    background: #ffffff;
    border-bottom: 1px solid var(--slate-200);
  }}
  .nav-inner {{
    max-width: 72rem;
    height: 100%;
    margin: 0 auto;
    padding: 0 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .brand {{
    display: inline-flex;
    align-items: center;
    gap: 0.625rem;
    text-decoration: none;
  }}
  .brand .mark {{
    width: 36px; height: 36px;
    background: var(--navy);
    border-radius: 0.5rem;
    display: flex; align-items: center; justify-content: center;
    color: #ffffff;
  }}
  .brand .name {{
    font-size: 1.125rem;
    font-weight: 600;
    color: var(--navy);
    letter-spacing: -0.01em;
  }}
  .nav-link {{
    font-size: 0.875rem;
    font-weight: 500;
    color: var(--slate-500);
    text-decoration: none;
    padding: 0.5rem 1.25rem;
    border-radius: 0.5rem;
    transition: background 0.2s, color 0.2s;
  }}
  .nav-link.solid {{
    background: var(--navy);
    color: #ffffff;
  }}
  .nav-link.solid:hover {{ background: var(--navy-light); }}

  /* ── Hero (warm bg + 64px slate grid, matching LandingPage hero) ── */
  .hero {{
    position: relative;
    min-height: 100vh;
    padding-top: 64px;
    display: flex;
    align-items: center;
    overflow: hidden;
    background: var(--warm);
  }}
  .hero .grid {{
    position: absolute;
    inset: 0;
    opacity: 0.4;
    background-image:
      linear-gradient(to right, var(--slate-200) 1px, transparent 1px),
      linear-gradient(to bottom, var(--slate-200) 1px, transparent 1px);
    background-size: 64px 64px;
    pointer-events: none;
  }}
  .hero-inner {{
    position: relative;
    z-index: 10;
    width: 100%;
    max-width: 48rem;
    margin: 0 auto;
    padding: 5rem 1.5rem;
    text-align: center;
  }}
  .eyebrow {{
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: var(--slate-400);
    margin-bottom: 1.5rem;
  }}
  h1 {{
    font-family: 'Playfair Display', Georgia, serif;
    font-weight: 600;
    font-size: clamp(2.25rem, 5vw, 3.5rem);
    line-height: 1.15;
    color: var(--navy);
    margin-bottom: 1.5rem;
  }}
  .lede {{
    font-size: 1.125rem;
    color: var(--slate-500);
    line-height: 1.7;
    max-width: 34rem;
    margin: 0 auto 2.5rem;
  }}

  /* ── Link rows ── */
  .links {{
    max-width: 32rem;
    margin: 0 auto 2.5rem;
    border-top: 1px solid var(--slate-200);
    border-bottom: 1px solid var(--slate-200);
    padding: 0.5rem 0;
  }}
  .link-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.875rem 0.25rem;
    text-decoration: none;
    border-bottom: 1px solid var(--slate-100);
    transition: background 0.15s;
  }}
  .link-row:last-child {{ border-bottom: none; }}
  .link-row:hover {{ background: rgba(255,255,255,0.6); }}
  .link-row .label {{
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--slate-400);
    flex: 0 0 auto;
  }}
  .link-row .value {{
    font-size: 0.95rem;
    font-weight: 500;
    color: var(--navy);
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    word-break: break-word;
    text-align: right;
  }}
  .link-row .value .arrow {{
    color: var(--slate-300);
    transition: transform 0.15s, color 0.15s;
  }}
  .link-row:hover .value .arrow {{
    color: var(--slate-500);
    transform: translateX(2px);
  }}

  /* ── Personal note card ── */
  .note {{
    max-width: 32rem;
    margin: 0 auto;
    text-align: left;
    background: var(--slate-50);
    border: 1px solid var(--slate-200);
    border-left: 2px solid var(--navy);
    border-radius: 0 0.75rem 0.75rem 0;
    padding: 1.5rem 1.75rem;
    font-size: 0.95rem;
    color: var(--slate-600);
    line-height: 1.7;
  }}
  .note .from {{
    display: block;
    margin-top: 1rem;
    font-family: 'Playfair Display', Georgia, serif;
    font-style: italic;
    font-size: 1.05rem;
    color: var(--navy);
  }}
  .note .from a {{
    font-style: normal;
    text-decoration: underline;
    text-underline-offset: 2px;
    text-decoration-color: var(--slate-300);
  }}
  .note .from a:hover {{ text-decoration-color: var(--navy); }}

  /* ── Footer (dark navy, mirrors LandingPage <footer>) ── */
  footer {{
    background: var(--navy);
    border-top: 1px solid var(--slate-800);
    padding: 2.5rem 1.5rem;
  }}
  .footer-inner {{
    max-width: 72rem;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1.5rem;
    text-align: center;
  }}
  .footer-brand {{
    display: inline-flex;
    align-items: center;
    gap: 0.625rem;
  }}
  .footer-brand .mark {{
    width: 32px; height: 32px;
    background: var(--slate-800);
    border-radius: 0.5rem;
    display: flex; align-items: center; justify-content: center;
    color: #ffffff;
  }}
  .footer-brand .name {{
    font-size: 1rem;
    font-weight: 500;
    color: #ffffff;
  }}
  .footer-inner p {{
    font-size: 0.875rem;
    color: var(--slate-500);
  }}

  @media (min-width: 768px) {{
    .footer-inner {{ flex-direction: row; justify-content: space-between; }}
  }}
  @media (max-width: 560px) {{
    .nav-links-secondary {{ display: none; }}
    .link-row {{ flex-direction: column; align-items: flex-start; gap: 0.25rem; }}
    .link-row .value {{ text-align: left; }}
  }}
</style>
</head>
<body>
  <nav>
    <div class="nav-inner">
      <a class="brand" href="/">
        <span class="mark">{logo_nav}</span>
        <span class="name">StrataLens</span>
      </a>
      <a class="nav-link solid" href="https://github.com/kamathhrishi/finance-agent" target="_blank" rel="noopener noreferrer">View on GitHub</a>
    </div>
  </nav>

  <section class="hero">
    <div class="grid"></div>
    <div class="hero-inner">
      <span class="eyebrow">Market Intelligence Platform</span>
      <h1>It was a pleasure building StrataLens AI.</h1>
      <p class="lede">
        This site is no longer maintained. The project is open source — clone
        the repo and run it with your own key.
      </p>

      <div class="links">
        <a class="link-row" href="https://github.com/kamathhrishi/finance-agent" target="_blank" rel="noopener noreferrer">
          <span class="label">Source code</span>
          <span class="value">github.com/kamathhrishi/finance-agent <span class="arrow" aria-hidden>&rarr;</span></span>
        </a>
        <a class="link-row" href="https://substack.com/@kamathhrishi" target="_blank" rel="noopener noreferrer">
          <span class="label">Blog</span>
          <span class="value">substack.com/@kamathhrishi <span class="arrow" aria-hidden>&rarr;</span></span>
        </a>
      </div>

      <div class="note">
        A note from me — thank you to everyone who tried StrataLens and shared
        feedback along the way. It began as an experiment in researching
        companies straight from primary sources, and I learned an enormous
        amount building it. The code is open, so take it, fork it, and make it
        your own.
        <span class="from">
          &mdash; Hrishikesh &middot;
          <a href="https://kamathhrishi.github.io/" target="_blank" rel="noopener noreferrer">kamathhrishi.github.io</a>
        </span>
      </div>
    </div>
  </section>

  <footer>
    <div class="footer-inner">
      <div class="footer-brand">
        <span class="mark">{logo_footer}</span>
        <span class="name">StrataLens</span>
      </div>
      <p>Institutional-grade market intelligence platform</p>
    </div>
  </footer>
</body>
</html>
""".format(
    logo_nav=_LOGO_SVG.format(size=17),
    logo_footer=_LOGO_SVG.format(size=14),
)
