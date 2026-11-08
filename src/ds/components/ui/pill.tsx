'use client';

import { Toggle as TogglePrimitive } from '@base-ui/react/toggle';
import type { ComponentProps } from 'react';
import { cn } from '@/ds/lib/cn';

const BASE =
	'relative inline-flex min-h-(--pill-h) items-center whitespace-nowrap rounded-(--pill-radius) border-0 px-(--pill-px) font-mono text-label font-medium leading-none tracking-normal no-underline';
const INTERACTIVE = 'press focus-ring hit-area cursor-pointer';

const VARIANTS = {
	/** A filled `--foreground` link; flips on dark surfaces. */
	link: 'bg-(--pill-link-bg) text-(--pill-link-fg) hovered:bg-(--pill-link-bg-hover) hovered:text-(--pill-link-fg-hover)',
	/** Outlined; static, or a link with `href`. */
	chip: 'edge text-foreground [--edge:var(--pill-edge)]',
} as const;

export type PillVariant = keyof typeof VARIANTS;

export type PillProps = ComponentProps<'span'> & {
	variant?: PillVariant;
	href?: string;
	target?: string;
	rel?: string;
};

/** Small mono capsule: a link or a static chip. A pressed state is `PillToggle`. */
export function Pill({ variant = 'link', href, className, ...props }: PillProps) {
	const classes = cn(
		BASE,
		VARIANTS[variant],
		href && INTERACTIVE,
		href && variant === 'chip' && 'hovered:[--edge:var(--pill-edge-hover)]',
		className,
	);
	if (href) {
		return (
			<a
				data-slot="pill"
				data-variant={variant}
				href={href}
				className={classes}
				{...(props as ComponentProps<'a'>)}
			/>
		);
	}
	return <span data-slot="pill" data-variant={variant} className={classes} {...props} />;
}

export type PillToggleProps = TogglePrimitive.Props;

/** Outlined toggle pill; accent fill while pressed. Base UI owns `aria-pressed`. */
export function PillToggle({ className, ...props }: PillToggleProps) {
	return (
		<TogglePrimitive
			data-slot="pill"
			data-variant="toggle"
			className={cn(
				BASE,
				INTERACTIVE,
				'edge text-foreground [--edge:var(--pill-edge)] hovered:[--edge:var(--pill-edge-hover)]',
				'data-pressed:bg-(--pill-on-bg) data-pressed:text-(--pill-on-fg) data-pressed:[--edge:transparent]',
				className,
			)}
			{...props}
		/>
	);
}
