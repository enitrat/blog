# Keep Astro as the experience shell

The site stays on Astro. We considered Next.js and a site-wide single-page
application, and rejected both.

Most of the site is reading: essays, Field Reports, book records. Static HTML
serves reading best. It loads fast, works without JavaScript, and needs no
hydration. Only a few features need code in the browser, like the 3D living
room and the bookcase on `/bookshelf/`. Astro runs each of those as an island,
a bounded piece of client-side code inside an otherwise static page.

Two rules follow from this decision:

- Put interactive libraries and framework components inside islands, never
  around the whole page.
- Keep essential content and navigation working without WebGL, animation, or
  JavaScript. The living room, for example, falls back to a still poster with
  real links.
