import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';
import { StatusDot } from './status-dot';

/** Small-caps caption, with an optional status dot or icon and right-aligned meta. */
export function Eyebrow({
	as: Tag = 'span',
	dot,
	icon,
	meta,
	id,
	className,
	children,
}: {
	id?: string;
	as?: 'span' | 'p' | 'h2' | 'h3' | 'div';
	dot?: boolean;
	icon?: ReactNode;
	meta?: ReactNode;
	className?: string;
	children?: ReactNode;
}) {
	return (
		<Tag
			id={id}
			className={cn(
				'flex items-center gap-1.5 text-eyebrow font-medium text-faint uppercase [&_svg]:size-3.5 [&_svg]:shrink-0',
				className,
			)}
		>
			{dot && <StatusDot />}
			{icon}
			{children}
			{meta && <span className="ms-auto text-end">{meta}</span>}
		</Tag>
	);
}
