# Keep Astro as the experience shell

The site stays on Astro, not Next.js or a site-wide single-page app.

## Reading wants static HTML

Most of the site is reading: essays, Field Reports, book records. Static HTML
loads fast, works without JavaScript, and needs no hydration. A few features
need browser code, like the living room and its bookcase. Astro runs each as
an island inside a static page.

## Two rules follow

- Keep interactive libraries and framework components inside islands.
- Essential content and navigation work without WebGL, animation, or
  JavaScript. The living room falls back to a poster with real links.
