import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';
import { StatusDot } from './status-dot';

/** Small muted caption, with an optional status dot and right-aligned meta. */
export function Eyebrow({
	as: Tag = 'span',
	dot,
	meta,
	id,
	className,
	children,
}: {
	id?: string;
	as?: 'span' | 'p' | 'h2' | 'h3' | 'div';
	dot?: boolean;
	meta?: ReactNode;
	className?: string;
	children?: ReactNode;
}) {
	return (
		<Tag
			id={id}
			className={cn(
				'flex items-center gap-1.5 text-xs font-medium text-gray-400 dark:text-iron-400',
				className,
			)}
		>
			{dot && <StatusDot />}
			{children}
			{meta && <span className="ms-auto text-end">{meta}</span>}
		</Tag>
	);
}
