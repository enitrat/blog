import { cva, type VariantProps } from 'class-variance-authority';
import NextLink from 'next/link';
import type { ReactNode } from 'react';
import { HoverScale } from '@/components/motion/hover-scale';
import { cn } from '@/lib/utils';
import { Eyebrow } from './eyebrow';

const tileVariants = cva(
	'relative flex h-full min-w-0 flex-col overflow-hidden rounded-xl border border-gray-200 bg-white shadow-xs dark:border-white/10 dark:bg-iron-900',
	{
		variants: {
			/** `compact`: tighter padding for cards that must fit a fixed cell. */
			size: { default: 'gap-4 p-4', compact: 'gap-2 p-3' },
			linked: { true: 'outline-none focus-visible:ring-2 focus-visible:ring-brand-500/60' },
		},
		defaultVariants: { size: 'default' },
	},
);

/** A card: hairline, shadow-xs, rounded-xl, with an optional header of label, meta and action. */
export function Tile({
	label,
	labelAs,
	dot,
	meta,
	action,
	href,
	size,
	className,
	children,
}: {
	label?: ReactNode;
	labelAs?: 'span' | 'h2' | 'h3';
	dot?: boolean;
	meta?: ReactNode;
	action?: ReactNode;
	/** Makes the whole tile one link that springs on hover. A linked tile holds no other links. */
	href?: string;
	size?: VariantProps<typeof tileVariants>['size'];
	className?: string;
	children?: ReactNode;
}) {
	const surface = tileVariants({ size, linked: Boolean(href) });
	const inner = (
		<>
			{(label || meta || action) && (
				<header className="flex items-center justify-between gap-2">
					<Eyebrow as={labelAs} dot={dot} meta={meta} className="flex-1">
						{label}
					</Eyebrow>
					{action}
				</header>
			)}
			<div className="min-h-0 min-w-0 flex-1">{children}</div>
		</>
	);
	if (!href) return <section className={cn(surface, className)}>{inner}</section>;
	return (
		<HoverScale className={cn('h-full', className)}>
			<NextLink href={href} className={surface}>
				{inner}
			</NextLink>
		</HoverScale>
	);
}
