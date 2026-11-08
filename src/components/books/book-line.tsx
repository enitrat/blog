import { cva, type VariantProps } from 'class-variance-authority';
import type { ReactNode } from 'react';

const titleVariants = cva(
	'flex flex-wrap items-center gap-1.5 font-semibold text-balance text-foreground',
	{ variants: { size: { md: 'text-sm', lg: 'text-lg' } }, defaultVariants: { size: 'md' } },
);

/** A book's title with its author beneath, and an optional mono facts line. */
export function BookLine({
	title,
	author,
	size,
	tag,
	children,
}: {
	/** A trailing "(series, tome)" is dropped. */
	title: string;
	author: string;
	size?: VariantProps<typeof titleVariants>['size'];
	/** Beside the title, such as a `StatusDot` or a `Badge`. */
	tag?: ReactNode;
	/** Mono facts under the author, such as "412 pages". */
	children?: ReactNode;
}) {
	return (
		<div>
			<p className={titleVariants({ size })}>
				{title.replace(/\s*\(.*\)\s*$/, '')}
				{tag}
			</p>
			<p className="mt-1 text-sm text-muted-foreground">{author}</p>
			{children && <p className="mt-1 font-mono text-xs text-faint">{children}</p>}
		</div>
	);
}
