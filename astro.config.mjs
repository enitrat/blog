// @ts-check

import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'astro/config';
import vercel from '@astrojs/vercel';
import { rehypeHeadingIds } from '@astrojs/markdown-remark';
import codeTheme from './src/styles/code-theme.json' with { type: 'json' };
import { isArgumentHeading } from './src/utils/sections.mjs';

/**
 * Numbers every `h2` and hangs a section reference (§ plus its ordinal) in the
 * sheet's outer margin beside it. This is the article's place-keyed marginalia:
 * apparatus sitting next to the passage it refers to, with the same ordinals
 * repeated in the contents list so margin and index are one system. Written
 * inline to avoid pulling in a plugin and its unist dependency for thirty
 * lines of tree walking.
 */
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
							// The mark's accessible name; the glyph and numeral are
							// decorative duplicates of the heading beside them.
							'aria-label': `Link to section ${n}`,
						},
						children: [
							{ type: 'element', tagName: 'span', properties: {}, children: [{ type: 'text', value: '§' }] },
							{ type: 'element', tagName: 'b', properties: {}, children: [{ type: 'text', value: String(n) }] },
						],
					});
				}
				walk(child);
			}
		};
		walk(tree);
	};
}

// https://astro.build/config
export default defineConfig({
	site: 'https://example.com',
	integrations: [mdx(), sitemap()],

	// /blog and /reads were the old addresses; the nav now says Writing and
	// Bookshelf, so the routes match the words. Existing inbound links and
	// indexed pages keep working.
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
		// Code is set in the site's own five colours rather than an off-the-shelf
		// theme, so a code block reads as part of the page instead of a pasted
		// terminal. See src/styles/code-theme.json for the scope mapping.
		shikiConfig: {
			theme: codeTheme,
		},

		rehypePlugins: [rehypeHeadingIds, rehypeSectionMarks],
	},

	vite: {
		plugins: [tailwindcss()],
	},
	adapter: vercel(),
});
