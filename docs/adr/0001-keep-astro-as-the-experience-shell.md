# Keep Astro as the experience shell

The site will remain on Astro rather than migrate to Next.js or a site-wide single-page application. Its content and reading surfaces benefit from static semantic HTML, while Astro's selective client islands can carry the few interactions that need a browser runtime, including an optional locally bundled Three.js experience; this preserves performance and progressive enhancement without limiting the intended visual quality.

Interactive libraries and framework components will be introduced only inside bounded islands when the selected experience requires them. Essential content and navigation must not depend on WebGL, animation, or client-side hydration.
