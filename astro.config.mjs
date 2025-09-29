// @ts-check

import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'astro/config';
import vercel from '@astrojs/vercel';

// https://astro.build/config
export default defineConfig({
	site: 'https://example.com',
	integrations: [mdx(), sitemap()],

	markdown: {
		// Note: mermaid diagrams need to be added as images, due to vercel deployment issues.
		syntaxHighlight: {
			type: 'shiki',
			excludeLangs: ['mermaid', 'math'],
		},
		shikiConfig: {
			themes: {
				light: 'catppuccin-latte',
				dark: 'github-dark',
			},
		},
	},

	vite: {
		plugins: [tailwindcss()],
	},
	adapter: vercel(),
});
