import type { Element, Root } from 'hast';
import { type Components, toJsxRuntime } from 'hast-util-to-jsx-runtime';
import { ArrowUpIcon } from 'lucide-react';
import Image from 'next/image';
import type { ComponentProps } from 'react';
import { Fragment, jsx, jsxs } from 'react/jsx-runtime';
import { IN_SHORT_LABEL } from '@/lib/in-short';
import { cn } from '@/lib/utils';
import { CodeBlock } from './code-block';

/** Stands in the tree where a back link to "In short" goes; rendered by `components`. */
const BACK_LINK: Element = {
	type: 'element',
	tagName: 'in-short-link',
	properties: {},
	children: [],
};

/** A back link before every section but the first, and one at the end. */
function withBackLinks(tree: Root): Root {
	const children: Root['children'] = [];
	let sections = 0;
	for (const node of tree.children) {
		if (node.type === 'element' && node.tagName === 'h2' && sections++ > 0)
			children.push(BACK_LINK);
		children.push(node);
	}
	children.push(BACK_LINK);
	return { ...tree, children };
}

/** A markdown image: a bundled one (sized by `renderMarkdown`) goes through `next/image`. */
function ProseImage({ src, alt = '', width, height, ...props }: ComponentProps<'img'>) {
	if (typeof src !== 'string' || !width || !height) {
		// biome-ignore lint/performance/noImgElement: an external image with no known size
		return <img src={src} alt={alt} {...props} />;
	}
	return (
		<Image
			src={src}
			alt={alt}
			width={Number(width)}
			height={Number(height)}
			sizes="(min-width: 40rem) 36rem, 100vw"
		/>
	);
}

function BackLink() {
	return (
		<p className="not-prose mt-8">
			<a
				href="#in-short"
				className="inline-flex items-center gap-1 rounded-inline type-control-sm text-faint transition-colors outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
			>
				<ArrowUpIcon aria-hidden className="size-3.5" /> {IN_SHORT_LABEL}
			</a>
		</p>
	);
}

const components = {
	pre: CodeBlock,
	img: ProseImage,
	'in-short-link': BackLink,
} as Partial<Components>;

const PROSE = cn(
	'prose prose-lg max-w-reading essay',
	// Roles swap with the theme, so prose needs no invert.
	'[--tw-prose-body:var(--muted-foreground)] [--tw-prose-headings:var(--foreground)] [--tw-prose-lead:var(--muted-foreground)] [--tw-prose-links:var(--primary)] [--tw-prose-bold:var(--foreground)] [--tw-prose-counters:var(--faint)] [--tw-prose-bullets:var(--faint)] [--tw-prose-hr:var(--border)] [--tw-prose-quotes:var(--muted-foreground)] [--tw-prose-quote-borders:var(--border-strong)] [--tw-prose-captions:var(--faint)] [--tw-prose-kbd:var(--foreground)] [--tw-prose-code:var(--foreground)] [--tw-prose-pre-code:var(--foreground)] [--tw-prose-pre-bg:var(--card)] [--tw-prose-th-borders:var(--border-strong)] [--tw-prose-td-borders:var(--border)]',
	'prose-p:text-pretty prose-li:text-pretty prose-headings:scroll-mt-anchor prose-headings:text-balance',
	// Section breaks in the markdown; the h2 spacing carries them.
	'prose-hr:hidden',
	'prose-a:rounded-inline prose-a:decoration-primary/30 prose-a:underline-offset-[22%] prose-a:transition-colors prose-a:duration-200 prose-a:outline-none prose-a:hover:decoration-current prose-a:focus-visible:ring-2 prose-a:focus-visible:ring-ring',
	// Inline code: a hairline chip, no backticks.
	'[&_:not(pre)>code]:rounded-chip [&_:not(pre)>code]:border [&_:not(pre)>code]:border-border [&_:not(pre)>code]:px-1 [&_:not(pre)>code]:py-0.5',
	// Code blocks: a card; tokens carry both theme colours as variables.
	'prose-figure:my-6 prose-pre:my-0 prose-pre:rounded-row prose-pre:shadow-card prose-pre:outline-none prose-pre:focus-visible:ring-2 prose-pre:focus-visible:ring-ring',
	'[&_pre_span]:text-(--shiki-light) dark:[&_pre_span]:text-(--shiki-dark)',
);

/**
 * Rendered markdown. The `essay` utility in globals.css sets its type; the classes here set its
 * colours, wrapping and chrome. `tree` is the build-time output of `renderMarkdown`; the column
 * width belongs to the page.
 */
export function Prose({
	tree,
	backLinks = false,
	className,
}: {
	tree: Root;
	/** Close each section with a link back to the page's `#in-short` block. */
	backLinks?: boolean;
	className?: string;
}) {
	return (
		<div className={cn(PROSE, className)}>
			{toJsxRuntime(backLinks ? withBackLinks(tree) : tree, { Fragment, jsx, jsxs, components })}
		</div>
	);
}
