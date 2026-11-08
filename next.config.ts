import type { NextConfig } from 'next';

/**
 * A static export into `dist/`, so the browser checks serve a plain folder as they did under
 * Astro. Redirects live in `vercel.json`, since an export has no server to answer them.
 */
const config: NextConfig = {
	output: 'export',
	// `next dev` for a browser check sets its own dir, so parallel checks never share a cache.
	distDir: process.env.NEXT_DIST_DIR ?? 'dist',
	// `about/index.html`, as Astro emitted: indexed URLs keep their trailing slash.
	trailingSlash: true,
	images: { unoptimized: true },
	reactStrictMode: true,
};

export default config;
