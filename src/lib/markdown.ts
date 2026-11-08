import 'server-only';
import rehypeShiki from '@shikijs/rehype';
import type { Element, Root as HastRoot } from 'hast';
import { toString as hastToString } from 'hast-util-to-string';
import type { Root as MdastRoot } from 'mdast';
import rehypeRaw from 'rehype-raw';
import rehypeSlug from 'rehype-slug';
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
export type Rendered = { tree: HastRoot; headings: Heading[] };

/** Essays set no banner: drop a leading `![](./banner.png)` paragraph. The file stays in the repo. */
const remarkDropBanner = () => (tree: MdastRoot) => {
	const first = tree.children[0];
	const img = first?.type === 'paragraph' && first.children.length === 1 ? first.children[0] : null;
	if (img?.type === 'image' && /(^|\/)banner\.\w+$/.test(img.url)) tree.children.shift();
};

/** Points a relative `<img src="./x.png">` at the bundled file with its size, for `next/image`. */
const rehypeLocalImages = (dir: string) => () => async (tree: HastRoot) => {
	const imgs: Element[] = [];
	visit(tree, 'element', (node: Element) => {
		if (node.tagName === 'img') imgs.push(node);
	});
	for (const node of imgs) {
		const src = String(node.properties.src ?? '');
		if (/^(https?:)?\/\//.test(src) || src.startsWith('/')) continue;
		const img = await maybeImage(`content/${dir}/${src.replace(/^\.\//, '')}`);
		if (!img) throw new Error(`Markdown image ${src} in ${dir} is not under src/content.`);
		node.properties.src = img.src;
		node.properties.width = img.width;
		node.properties.height = img.height;
	}
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
 * Names every code block after its language and its place among blocks of that language
 * ("ts sample 2"), so a reader knows which sample a Copy button copies. `CodeBlock` reads it.
 */
const rehypeCodeLabels = () => (tree: HastRoot) => {
	const seen = new Map<string, number>();
	visit(tree, 'element', (node: Element) => {
		if (node.tagName !== 'pre') return;
		const code = node.children.find(
			(c): c is Element => c.type === 'element' && c.tagName === 'code',
		);
		if (!code) return;
		// Shiki writes `class` as a string; remark-rehype writes `className` as a list.
		const classes = [code.properties.className ?? [], code.properties.class ?? []]
			.flat()
			.flatMap((c) => String(c).split(/\s+/));
		const language = classes.find((c) => c.startsWith('language-'))?.slice('language-'.length);
		const kind = language && language !== 'plaintext' && language !== 'text' ? language : 'code';
		const ordinal = (seen.get(kind) ?? 0) + 1;
		seen.set(kind, ordinal);
		node.properties.dataCopyLabel = `${kind} sample ${ordinal}`;
	});
};

const processor = (dir: string, headings: Heading[]) =>
	unified()
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
		.use(rehypeCodeLabels);

/**
 * Markdown to a hast tree at build time: GFM, inline HTML, heading ids, Shiki in two themes (light
 * inlined as `color`, dark as `--shiki-dark`; Prose picks one), and local images resolved to their
 * bundled files. `dir` is the entry's folder under `src/content`. `Prose` renders the tree as React.
 */
export async function renderMarkdown(body: string, dir: string): Promise<Rendered> {
	const headings: Heading[] = [];
	const p = processor(dir, headings);
	const tree = (await p.run(p.parse(body))) as HastRoot;
	return { tree, headings };
}
