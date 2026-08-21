# Keep Astro as the experience shell

The site remains on Astro rather than moving to Next.js or a site-wide single-page application. Static semantic HTML suits the content and reading surfaces; selective client islands handle the few interactions that need a browser runtime.

Interactive libraries and framework components belong inside bounded islands. Essential content and navigation must not depend on WebGL, animation, or client-side hydration.
