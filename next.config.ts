import type { NextConfig } from 'next';

const config: NextConfig = {
	// `next dev` for a browser check sets its own dir, so parallel checks never share a cache.
	distDir: process.env.NEXT_DIST_DIR ?? '.next',
	// `about/` as Astro emitted it: indexed URLs keep their trailing slash.
	trailingSlash: true,
	reactStrictMode: true,
	async redirects() {
		return [
			{ source: '/blog', destination: '/writing/', permanent: true },
			{ source: '/blog/:slug*', destination: '/writing/:slug*', permanent: true },
			{ source: '/reads', destination: '/reading/', permanent: true },
			{ source: '/bookshelf', destination: '/reading/', permanent: true },
			{ source: '/bookshelf/archive', destination: '/reading/', permanent: true },
		];
	},
};

export default config;
