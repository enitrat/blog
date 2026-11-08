import 'server-only';
import rehypeShiki from '@shikijs/rehype';
import type { Element, Root as HastRoot } from 'hast';
import { toString as hastToString } from 'hast-util-to-string';
import type { Root as MdastRoot } from 'mdast';
import rehypeRaw from 'rehype-raw';
import rehypeSlug from 'rehype-slug';
import rehypeStringify from 'rehype-stringify';
import remarkGfm from 'remark-gfm';
import remarkParse from 'remark-parse';
import remarkRehype from 'remark-rehype';
import remarkSmartypants from 'remark-smartypants';
import { unified } from 'unified';
import { visit } from 'unist-util-visit';
import codeThemeLight from '@/styles/code-theme.json';
import codeThemeDark from '@/styles/code-theme-dark.json';
import { maybeImage } from './media';

export type Heading = { depth: number; slug: string; text: string };
export type Rendered = { html: string; headings: Heading[] };

/** Essays set no banner: drop a leading `![](./banner.png)` paragraph. The file stays in the repo. */
const remarkDropBanner = () => (tree: MdastRoot) => {
	const first = tree.children[0];
	const img = first?.type === 'paragraph' && first.children.length === 1 ? first.children[0] : null;
	if (img?.type === 'image' && /(^|\/)banner\.\w+$/.test(img.url)) tree.children.shift();
};

/** Points a relative `<img src="./x.png">` at its published copy, with its size and srcset. */
const rehypeLocalImages = (dir: string) => () => (tree: HastRoot) => {
	visit(tree, 'element', (node: Element) => {
		if (node.tagName !== 'img') return;
		const src = String(node.properties.src ?? '');
		if (/^(https?:)?\/\//.test(src) || src.startsWith('/')) return;
		const key = `content/${dir}/${src.replace(/^\.\//, '')}`;
		const img = maybeImage(key);
		if (!img) throw new Error(`Markdown image ${src} in ${dir} is not under src/content.`);
		node.properties.src = img.src;
		node.properties.srcSet = img.srcSet;
		node.properties.sizes = '(min-width: 40rem) 38rem, 100vw';
		node.properties.width = img.width;
		node.properties.height = img.height;
		node.properties.loading = 'lazy';
		node.properties.decoding = 'async';
	});
};

/** Collects every heading with its id, after `rehype-slug` has set them (github-slugger, as Astro did). */
const rehypeCollectHeadings = (out: Heading[]) => () => (tree: HastRoot) => {
	visit(tree, 'element', (node: Element) => {
		const m = /^h([1-6])$/.exec(node.tagName);
		if (!m) return;
		out.push({
			depth: Number(m[1]),
			slug: String(node.properties.id ?? ''),
			text: hastToString(node),
		});
	});
};

/**
 * Markdown to HTML at build time: GFM, inline HTML, heading ids, Shiki in two themes (light
 * inlined as `color`, dark as `--shiki-dark`; Prose picks one), and local images published with a
 * srcset. `dir` is the entry's folder under `src/content`.
 */
export async function renderMarkdown(body: string, dir: string): Promise<Rendered> {
	const headings: Heading[] = [];
	const file = await unified()
		.use(remarkParse)
		.use(remarkGfm)
		// Curly quotes and dashes, as Astro's markdown did by default.
		.use(remarkSmartypants)
		.use(remarkDropBanner)
		.use(remarkRehype, { allowDangerousHtml: true })
		.use(rehypeRaw)
		.use(rehypeSlug)
		.use(rehypeCollectHeadings(headings))
		.use(rehypeLocalImages(dir))
		.use(rehypeShiki, {
			themes: {
				light: { ...codeThemeLight, type: 'light' } as never,
				dark: { ...codeThemeDark, type: 'dark' } as never,
			},
			defaultColor: 'light',
			// Mermaid diagrams are added as images.
			langs: [],
			lazy: true,
			fallbackLanguage: 'text',
			addLanguageClass: true,
		})
		.use(rehypeStringify)
		.process(body);
	return { html: String(file), headings };
}
