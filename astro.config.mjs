// @ts-check

import { rehypeHeadingIds } from '@astrojs/markdown-remark';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import vercel from '@astrojs/vercel';
import { defineConfig } from 'astro/config';
import codeTheme from './src/styles/code-theme.json' with { type: 'json' };
import { isArgumentHeading } from './src/utils/sections.mjs';

/** Numbers every argument `h2` and hangs a §ordinal link in the outer margin. */
function rehypeSectionMarks() {
	return (tree) => {
		let n = 0;
		const walk = (node) => {
			if (!node.children) return;
			for (const child of node.children) {
				if (
					child.type === 'element' &&
					child.tagName === 'h2' &&
					child.properties?.id &&
					isArgumentHeading(child.properties.id)
				) {
					n += 1;
					child.properties['data-section'] = String(n);
					child.children.unshift({
						type: 'element',
						tagName: 'a',
						properties: {
							className: ['section-mark'],
							href: `#${child.properties.id}`,
							// The glyph and numeral below are decorative duplicates of the
							// adjacent heading, so the link needs its own name.
							'aria-label': `Link to section ${n}`,
						},
						children: [
							{
								type: 'element',
								tagName: 'span',
								properties: {},
								children: [{ type: 'text', value: '§' }],
							},
							{
								type: 'element',
								tagName: 'b',
								properties: {},
								children: [{ type: 'text', value: String(n) }],
							},
						],
					});
				}
				walk(child);
			}
		};
		walk(tree);
	};
}

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

		rehypePlugins: [rehypeHeadingIds, rehypeSectionMarks],
	},

	adapter: vercel(),
});
