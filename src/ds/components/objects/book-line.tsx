import type { ReactNode } from 'react';
import { cn } from '@/ds/lib/cn';

const SIZES = {
	md: 'text-heading',
	lg: 'text-title',
} as const;

/** A book's title with its author beneath, and an optional mono facts line. */
export function BookLine({
	title,
	author,
	size = 'md',
	tag,
	children,
}: {
	/** A trailing "(series, tome)" is dropped. */
	title: string;
	author: string;
	size?: keyof typeof SIZES;
	/** Beside the title, such as a `StatusDot` or a `Tag`. */
	tag?: ReactNode;
	/** Mono facts under the author, such as "412 pages". */
	children?: ReactNode;
}) {
	return (
		<div data-slot="book-line" data-size={size} className="book">
			<p
				// The size first: `cn` drops a leading that precedes a text size.
				className={cn(
					SIZES[size],
					'title flex flex-wrap items-center gap-2 font-semibold leading-tight tracking-snug',
				)}
			>
				{title.replace(/\s*\(.*\)\s*$/, '')}
				{tag}
			</p>
			<p className="by mt-1 text-ui text-muted-foreground">{author}</p>
			{children && (
				<p className="facts mt-1 font-mono text-label font-regular text-muted-foreground">
					{children}
				</p>
			)}
		</div>
	);
}
