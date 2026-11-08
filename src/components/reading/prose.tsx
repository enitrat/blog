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
				className="inline-flex items-center gap-1 rounded-xs text-xs font-medium text-gray-400 transition-colors outline-none hover:text-gray-600 focus-visible:ring-2 focus-visible:ring-brand-500/60 dark:text-iron-400 dark:hover:text-iron-200"
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
	'prose prose-sm prose-gray max-w-none leading-relaxed sm:prose-base dark:prose-invert',
	// The dark ramp is iron, not the inverted gray.
	'dark:[--tw-prose-invert-body:var(--color-iron-300)] dark:[--tw-prose-invert-headings:var(--color-white)] dark:[--tw-prose-invert-bold:var(--color-iron-100)] dark:[--tw-prose-invert-quotes:var(--color-iron-300)] dark:[--tw-prose-invert-counters:var(--color-iron-400)] dark:[--tw-prose-invert-bullets:var(--color-iron-600)] dark:[--tw-prose-invert-captions:var(--color-iron-400)] dark:[--tw-prose-invert-quote-borders:var(--color-iron-800)] dark:[--tw-prose-invert-th-borders:var(--color-iron-700)] dark:[--tw-prose-invert-td-borders:var(--color-iron-800)] dark:[--tw-prose-invert-code:var(--color-iron-100)]',
	'prose-p:text-pretty prose-li:text-pretty prose-headings:scroll-mt-6 prose-headings:font-semibold prose-headings:text-balance',
	// Section breaks in the markdown; the h2 spacing carries them.
	'prose-hr:hidden',
	'prose-a:rounded-xs prose-a:font-normal prose-a:text-brand-600 prose-a:decoration-brand-600/30 prose-a:underline-offset-[22%] prose-a:transition-colors prose-a:duration-200 prose-a:outline-none prose-a:hover:text-brand-700 prose-a:hover:decoration-current prose-a:focus-visible:ring-2 prose-a:focus-visible:ring-brand-500/60 dark:prose-a:text-brand-400 dark:prose-a:decoration-brand-400/30 dark:prose-a:hover:text-brand-300',
	// Inline code: a hairline chip, no backticks.
	'prose-code:font-normal prose-code:before:content-none prose-code:after:content-none [&_:not(pre)>code]:rounded-md [&_:not(pre)>code]:border [&_:not(pre)>code]:border-gray-200 [&_:not(pre)>code]:px-1 [&_:not(pre)>code]:py-0.5 dark:[&_:not(pre)>code]:border-white/10',
	// Code blocks: a white card; tokens carry both theme colours as variables.
	'prose-figure:my-6 prose-pre:my-0 prose-pre:rounded-xl prose-pre:border prose-pre:border-gray-200 prose-pre:bg-white prose-pre:text-gray-800 prose-pre:shadow-xs prose-pre:outline-none prose-pre:focus-visible:ring-2 prose-pre:focus-visible:ring-brand-500/60 dark:prose-pre:border-white/10 dark:prose-pre:bg-iron-900 dark:prose-pre:text-iron-100',
	'[&_pre_span]:text-(--shiki-light) dark:[&_pre_span]:text-(--shiki-dark)',
);

/**
 * Essay typography for rendered markdown. `tree` is the build-time output of `renderMarkdown`;
 * the column width belongs to the page.
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
