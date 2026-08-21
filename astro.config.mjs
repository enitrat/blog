// @ts-check

import { rehypeHeadingIds } from '@astrojs/markdown-remark';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import vercel from '@astrojs/vercel';
import { defineConfig } from 'astro/config';
import codeThemeData from './src/styles/code-theme.json' with { type: 'json' };

// JSON imports widen literal values; keep the theme's discriminated `type`
// explicit at the configuration boundary so Astro and Shiki can validate it.
const codeTheme = { ...codeThemeData, type: /** @type {'light'} */ ('light') };

export default defineConfig({
	// Every canonical link, sitemap entry, RSS link, and og:url resolves against
	// this, so a wrong value is silently wrong everywhere rather than a build error.
	site: 'https://msaug.dev',
	integrations: [mdx(), sitemap()],

	// Keeps inbound links and already-indexed pages working after the rename.
	redirects: {
		'/blog': '/writing',
		'/blog/[...slug]': '/writing/[...slug]',
		'/reads': '/bookshelf',
	},

	markdown: {
		// Note: mermaid diagrams need to be added as images, due to vercel deployment issues.
		syntaxHighlight: {
			type: 'shiki',
			excludeLangs: ['mermaid', 'math'],
		},
		shikiConfig: {
			theme: codeTheme,
		},

		rehypePlugins: [rehypeHeadingIds],
	},

	adapter: vercel(),
});
