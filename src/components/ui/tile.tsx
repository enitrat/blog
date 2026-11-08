import NextLink from 'next/link';
import type { CSSProperties, ReactNode } from 'react';
import { HoverScale } from '@/components/motion/hover-scale';
import { cn } from '@/lib/utils';
import { Eyebrow } from './eyebrow';

export type TileProps = {
	/** Column span: md (6-column grid), lg (12-column grid). */
	span?: { md?: number; lg?: number };
	rows?: { lg?: number };
	/** Position on the one-column phone grid (lower first). */
	order?: number;
	label?: ReactNode;
	labelAs?: 'span' | 'h2' | 'h3';
	dot?: boolean;
	live?: boolean;
	meta?: ReactNode;
	action?: ReactNode;
	/** Makes the whole tile one link that springs on hover. A linked tile holds no other links. */
	href?: string;
	/** `media`: zero padding, full-bleed body, header over an edge fade. */
	variant?: 'default' | 'media';
	/** Tighter padding for cards that must fit a fixed cell. */
	compact?: boolean;
	/** Names an unlabelled tile by the id of a heading inside it. */
	labelledBy?: string;
	className?: string;
	bodyClassName?: string;
	style?: CSSProperties;
	children?: ReactNode;
};

const SPAN_MD: Record<number, string> = {
	2: 'sm:col-span-2',
	3: 'sm:col-span-3',
	4: 'sm:col-span-4',
	6: 'sm:col-span-6',
};
const SPAN_LG: Record<number, string> = {
	3: 'lg:col-span-3',
	4: 'lg:col-span-4',
	5: 'lg:col-span-5',
	6: 'lg:col-span-6',
	7: 'lg:col-span-7',
	8: 'lg:col-span-8',
	12: 'lg:col-span-12',
};
const ROWS_LG: Record<number, string> = {
	2: 'lg:row-span-2',
	3: 'lg:row-span-3',
	4: 'lg:row-span-4',
};

/** A card on the home grid: hairline, shadow-xs, rounded-xl. */
export function Tile({
	span = {},
	rows = {},
	order,
	label,
	labelAs,
	dot,
	live,
	meta,
	action,
	href,
	variant = 'default',
	compact = false,
	labelledBy,
	className,
	bodyClassName,
	style,
	children,
}: TileProps) {
	const media = variant === 'media';
	const head = (label || meta || action) && (
		<header
			className={cn(
				'flex items-center justify-between gap-2',
				media &&
					'absolute inset-x-0 top-0 z-10 bg-linear-to-b from-gray-50 via-gray-50/50 via-75% to-gray-50/0 p-4 dark:from-iron-fade dark:via-iron-fade/50 dark:to-iron-fade/0',
			)}
		>
			<Eyebrow as={labelAs} dot={dot} live={live} meta={meta} className="flex-1">
				{label}
			</Eyebrow>
			{action}
		</header>
	);
	const inner = (
		<>
			{head}
			<div
				className={cn(
					'min-h-0 min-w-0 flex-1',
					media && 'absolute inset-0 [&>img]:size-full [&>img]:object-cover',
					bodyClassName,
				)}
			>
				{children}
			</div>
		</>
	);
	const surface = cn(
		'relative flex h-full min-w-0 flex-col overflow-hidden rounded-xl border border-gray-200 bg-white shadow-xs dark:border-white/10 dark:bg-iron-900',
		media ? 'min-h-64 p-0' : compact ? 'gap-2 p-3' : 'gap-4 p-4',
		href && 'outline-none focus-visible:ring-2 focus-visible:ring-brand-500/60',
	);
	const place = cn(
		'order-(--order) min-w-0 sm:order-none',
		SPAN_MD[span.md ?? 6],
		SPAN_LG[span.lg ?? 12],
		rows.lg && ROWS_LG[rows.lg],
		className,
	);
	const vars = { ...(order !== undefined && { '--order': order }), ...style } as CSSProperties;
	if (!href) {
		return (
			<section className={cn(surface, place)} style={vars} aria-labelledby={labelledBy}>
				{inner}
			</section>
		);
	}
	return (
		<HoverScale className={cn('h-full', place)} style={vars}>
			<NextLink href={href} className={surface}>
				{inner}
			</NextLink>
		</HoverScale>
	);
}

/** The home grid: one column on phones, 6 from sm, 12 from lg. */
export function Bench({ className, children }: { className?: string; children: ReactNode }) {
	return (
		<div
			className={cn(
				'grid grid-cols-1 gap-1.5 sm:grid-cols-6 lg:auto-rows-[minmax(11rem,auto)] lg:grid-cols-12',
				className,
			)}
		>
			{children}
		</div>
	);
}
