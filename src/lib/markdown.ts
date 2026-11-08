import 'server-only';
import type { Element, Root as HastRoot } from 'hast';
import { toString as hastToString } from 'hast-util-to-string';
import type { Root as MdastRoot } from 'mdast';
import rehypePrettyCode from 'rehype-pretty-code';
import rehypeRaw from 'rehype-raw';
import rehypeSlug from 'rehype-slug';
import remarkGfm from 'remark-gfm';
import remarkParse from 'remark-parse';
import remarkRehype from 'remark-rehype';
import remarkSmartypants from 'remark-smartypants';
import type { ShikiTransformer } from 'shiki';
import { unified } from 'unified';
import { visit } from 'unist-util-visit';
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
const sampleLabels = (): ShikiTransformer => {
	const seen = new Map<string, number>();
	return {
		name: 'sample-labels',
		pre(node) {
			const { lang } = this.options;
			const kind = lang === 'plaintext' || lang === 'text' ? 'code' : lang;
			const ordinal = (seen.get(kind) ?? 0) + 1;
			seen.set(kind, ordinal);
			node.properties.dataCopyLabel = `${kind} sample ${ordinal}`;
		},
	};
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
		.use(rehypePrettyCode, {
			// Each token carries `--shiki-light` and `--shiki-dark`; Prose picks one.
			theme: { light: 'github-light', dark: 'github-dark-dimmed' },
			keepBackground: false,
			bypassInlineCode: true,
			defaultLang: { block: 'plaintext' },
			transformers: [sampleLabels()],
		});

/**
 * Markdown to a hast tree at build time: GFM, inline HTML, heading ids, code highlighted in two
 * themes, and local images resolved to their bundled files. `dir` is the entry's folder under
 * `src/content`. `Prose` renders the tree as React.
 */
export async function renderMarkdown(body: string, dir: string): Promise<Rendered> {
	const headings: Heading[] = [];
	const p = processor(dir, headings);
	const tree = (await p.run(p.parse(body))) as HastRoot;
	return { tree, headings };
}
