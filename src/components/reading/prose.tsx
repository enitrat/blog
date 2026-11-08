import type { Element, Root } from 'hast';
import { type Components, toJsxRuntime } from 'hast-util-to-jsx-runtime';
import Image from 'next/image';
import type { ComponentProps } from 'react';
import { Fragment, jsx, jsxs } from 'react/jsx-runtime';
import { Icon, Link } from '@/ds';
import { cn } from '@/ds/lib/cn';
import { IN_SHORT_LABEL } from '@/utils/inShort';
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
			sizes="(min-width: 40rem) 38rem, 100vw"
		/>
	);
}

const components = {
	pre: CodeBlock,
	img: ProseImage,
	'in-short-link': () => (
		<p className="up">
			<Link href="#in-short" variant="quiet">
				<Icon name="arrow-up" size="sm" /> {IN_SHORT_LABEL}
			</Link>
		</p>
	),
} as Partial<Components>;

/**
 * Essay typography for rendered markdown: Geist at the body role, one `--measure` column,
 * left-aligned and unhyphenated. `tree` is the build-time output of `renderMarkdown`.
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
		<div className={cn('prose', className)}>
			{toJsxRuntime(backLinks ? withBackLinks(tree) : tree, { Fragment, jsx, jsxs, components })}
		</div>
	);
}
