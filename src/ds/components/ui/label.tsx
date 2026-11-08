import type { ReactNode } from 'react';
import { cn } from '@/ds/lib/cn';
import { StatusDot, type StatusDotTone } from './status-dot';

const SPACE = { sm: 'mb-3', md: 'mb-4' } as const;

export type LabelProps = {
	children?: ReactNode;
	/** Right-aligned secondary text. */
	meta?: ReactNode;
	dot?: boolean;
	dotTone?: StatusDotTone;
	as?: 'span' | 'p' | 'h2' | 'h3';
	/** Fill the free width of a flex parent, such as a tile header. */
	grow?: boolean;
	/** Gap below the label, so a page never reaches in to space it. */
	space?: keyof typeof SPACE;
	/** Dot-separate the children (a meta line: name · date · read time); wraps. */
	separated?: boolean;
	id?: string;
	className?: string;
};

/** Mono caps caption, with an optional status dot and right-aligned meta. Muted, so it retints on any surface. */
export function Label({
	children,
	meta,
	dot = false,
	dotTone = 'accent',
	as: Tag = 'span',
	grow = false,
	space,
	separated = false,
	id,
	className,
}: LabelProps) {
	return (
		<Tag
			id={id}
			data-slot="label"
			data-meta={meta ? '' : undefined}
			className={cn(
				'type-label m-0 flex items-center gap-(--gap-bench) text-muted-foreground',
				grow && 'flex-1',
				space && SPACE[space],
				className,
			)}
		>
			<span
				className={cn(
					'inline-flex items-center gap-[0.6em]',
					separated &&
						'flex-wrap gap-x-3 gap-y-1 [&>a]:text-link [&>:not(:first-child)]:before:me-3 [&>:not(:first-child)]:before:content-["·"]',
				)}
			>
				{dot && <StatusDot tone={dotTone} />}
				{children}
			</span>
			{meta && <span className="ms-auto text-end">{meta}</span>}
		</Tag>
	);
}
