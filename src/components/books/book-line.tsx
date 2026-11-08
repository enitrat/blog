import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';

const SIZES = { md: 'text-sm', lg: 'text-lg' } as const;

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
	/** Beside the title, such as a `StatusDot` or a `Badge`. */
	tag?: ReactNode;
	/** Mono facts under the author, such as "412 pages". */
	children?: ReactNode;
}) {
	return (
		<div>
			<p
				className={cn(
					'flex flex-wrap items-center gap-1.5 leading-tight font-semibold text-balance text-gray-800 dark:text-iron-100',
					SIZES[size],
				)}
			>
				{title.replace(/\s*\(.*\)\s*$/, '')}
				{tag}
			</p>
			<p className="mt-1 text-sm text-gray-500 dark:text-iron-400">{author}</p>
			{children && (
				<p className="mt-1 font-mono text-xs text-gray-400 dark:text-iron-500">{children}</p>
			)}
		</div>
	);
}
