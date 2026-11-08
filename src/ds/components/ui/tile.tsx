import type { CSSProperties, ElementType, ReactNode } from 'react';
import { cn } from '@/ds/lib/cn';
import { Label } from './label';
import type { StatusDotTone } from './status-dot';

const VARIANTS = {
	paper: 'p-(--tile-pad) gap-(--tile-gap)',
	/** Graphite: re-maps the paper names, so every atom inside retints with no prop. */
	inverse: 'p-(--tile-pad) gap-(--tile-gap)',
	/** Zero padding, full-bleed body, header over a scrim. */
	media: 'min-h-(--tile-media-min-h) gap-0 p-0',
} as const;

export type TileVariant = keyof typeof VARIANTS;

export type TileProps = {
	variant?: TileVariant;
	/** Column span inside `Bench`: md (6-column grid, default 6), lg (12-column grid, default 12). */
	span?: { md?: number; lg?: number };
	/** Row span in the 12-column grid. */
	rows?: { lg?: number };
	/** Position on the one-column phone grid (lower first). */
	order?: number;
	label?: ReactNode;
	labelAs?: 'span' | 'h2' | 'h3';
	dot?: boolean;
	dotTone?: StatusDotTone;
	meta?: ReactNode;
	/** Header action, such as `<Link variant="arrow">`. */
	action?: ReactNode;
	/** Makes the whole tile one link. A linked tile holds no links or buttons. */
	href?: string;
	as?: 'div' | 'section' | 'article';
	className?: string;
	bodyClassName?: string;
	style?: CSSProperties;
	children?: ReactNode;
	[attr: `data-${string}` | `aria-${string}`]: unknown;
	id?: string;
};

/**
 * A bench surface: 1px hairline, `--elevation-tile`, optional header. A tile with `href` lifts on
 * hover and settles on press; nothing else lifts.
 */
export function Tile({
	variant = 'paper',
	span = {},
	rows = {},
	order,
	label,
	labelAs,
	dot,
	dotTone,
	meta,
	action,
	href,
	as,
	className,
	bodyClassName,
	style,
	children,
	...rest
}: TileProps) {
	const Root: ElementType = href ? 'a' : (as ?? 'div');
	const hasHead = Boolean(label || meta || action);
	const vars = {
		'--span-md': span.md ?? 6,
		'--span-lg': span.lg ?? 12,
		...(rows.lg && { '--rows-lg': rows.lg }),
		...(order !== undefined && { '--order': order }),
		...style,
	} as CSSProperties;
	return (
		<Root
			className={cn(
				'tile relative flex min-w-0 flex-col overflow-hidden rounded-(--tile-radius) border border-(--tile-edge) bg-(--tile-bg) text-(--tile-fg) no-underline shadow-tile',
				'order-(--order,0) sm:order-0 sm:col-span-(--span-md) lg:col-span-(--span-lg) lg:row-span-(--rows-lg,1)',
				VARIANTS[variant],
				href &&
					'focus-ring transition-[transform,box-shadow,border-color] duration-base ease-out hovered:translate-y-(--lift-distance) hovered:border-(--tile-edge-hover) hovered:shadow-tile-lift active:scale-(--press-scale) active:shadow-tile active:duration-fast motion-reduce:transition-none',
				className,
			)}
			data-variant={variant}
			data-surface={variant === 'paper' ? 'paper' : 'inverse'}
			data-linked={href ? '' : undefined}
			href={href}
			style={vars}
			{...rest}
		>
			{hasHead && (
				<header
					className={cn(
						'tile-head flex items-center justify-between gap-(--tile-gap)',
						variant === 'media' &&
							'absolute inset-x-0 top-0 z-(--z-raised) bg-linear-to-b from-scrim to-transparent p-(--tile-pad)',
					)}
				>
					<Label as={labelAs} dot={dot} dotTone={dotTone} meta={meta} grow>
						{label}
					</Label>
					{action}
				</header>
			)}
			<div
				className={cn(
					'tile-body min-h-0 min-w-0 flex-1',
					variant === 'media' &&
						'absolute inset-0 [&>canvas]:size-full [&>img]:size-full [&>img]:object-cover [&>picture]:size-full',
					bodyClassName,
				)}
			>
				{children}
			</div>
		</Root>
	);
}

/**
 * The tile grid: one column on phones, 6 from 40rem, 12 from 64rem. Tiles declare their own
 * `span`, `rows` and phone `order`. `settle` is the page's one entrance animation.
 */
export function Bench({
	settle = false,
	className,
	children,
}: {
	settle?: boolean;
	className?: string;
	children: ReactNode;
}) {
	return (
		<div
			data-slot="bench"
			data-settle={settle ? '' : undefined}
			className={cn(
				'bench grid grid-cols-[minmax(0,1fr)] gap-(--gap-bench) sm:grid-cols-6 lg:auto-rows-(--bench-row) lg:grid-cols-12',
				settle && 'bench-settle',
				className,
			)}
		>
			{children}
		</div>
	);
}
