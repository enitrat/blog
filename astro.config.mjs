// @ts-check

import { existsSync, readdirSync } from 'node:fs';
import { rehypeHeadingIds } from '@astrojs/markdown-remark';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import vercel from '@astrojs/vercel';
import { defineConfig } from 'astro/config';
import codeThemeLight from './src/styles/code-theme.json' with { type: 'json' };
import codeThemeDark from './src/styles/code-theme-dark.json' with { type: 'json' };

// JSON imports widen literal values; keep the theme's discriminated `type`
// explicit at the configuration boundary so Astro and Shiki can validate it.
const codeThemes = {
	light: { ...codeThemeLight, type: /** @type {'light'} */ ('light') },
	dark: { ...codeThemeDark, type: /** @type {'dark'} */ ('dark') },
};

/** True once a trip folder with an `index.md` exists. */
const hasTrips = readdirSync('./src/content/trips', { withFileTypes: true }).some(
	(d) => d.isDirectory() && existsSync(`./src/content/trips/${d.name}/index.md`),
);

/** Essays set no banner: drop a leading `![](./banner.png)` paragraph. The file stays in the repo. */
const remarkDropBanner = () => (/** @type {any} */ tree) => {
	const first = tree.children[0];
	const img = first?.type === 'paragraph' && first.children.length === 1 ? first.children[0] : null;
	if (img?.type === 'image' && /(^|\/)banner\.\w+$/.test(img.url)) tree.children.shift();
};

export default defineConfig({
	// Every canonical link, sitemap entry, and og:url resolves against
	// this, so a wrong value is silently wrong everywhere rather than a build error.
	site: 'https://msaug.dev',
	integrations: [
		mdx(),
		sitemap({
			// /system is a docs page; /places is empty until the first trip (see src/utils/trips.ts).
			filter: (page) => !page.includes('/system') && (hasTrips || !page.includes('/places')),
		}),
	],

	// Keeps inbound links and already-indexed pages working after renames and removals.
	redirects: {
		'/blog': '/writing',
		'/blog/[...slug]': '/writing/[...slug]',
		'/reads': '/reading',
		'/bookshelf': '/reading',
		'/bookshelf/archive': '/reading',
	},

	markdown: {
		// Mermaid diagrams are added as images (Vercel deployment).
		syntaxHighlight: {
			type: 'shiki',
			excludeLangs: ['mermaid', 'math'],
		},
		shikiConfig: {
			// Dual theme: Shiki inlines `color` (light) and `--shiki-dark`; Prose picks one.
			themes: codeThemes,
		},

		remarkPlugins: [remarkDropBanner],
		rehypePlugins: [rehypeHeadingIds],
	},

	adapter: vercel(),
});
