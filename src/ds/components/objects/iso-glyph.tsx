import type { ReactNode } from 'react';
import { cn } from '@/ds/lib/cn';

/**
 * A tiny isometric object seen from above: one projection, four tones, 1px hairlines. Draw inside
 * it with the helpers in `iso.ts` and the classes `t` (top), `l` (left), `r` (right).
 *
 * `frame` is `[cx, cy, side]`, the square window onto the drawing, so each object fills about 85%
 * of its box and stays centred, hover pose included.
 *
 * Motion: an `IsoPart` moves to its `--pose` (a CSS transform, set by the glyph's class) while the
 * nearest `[data-glyph-host]` is hovered or has focus-visible within. Nothing moves at rest.
 */
export function IsoGlyph({
	size = 36,
	frame = [0, -2, 56],
	className,
	children,
}: {
	/** Box in px; the viewBox scales the drawing crisply. */
	size?: number;
	frame?: readonly [number, number, number];
	className?: string;
	children: ReactNode;
}) {
	const [cx, cy, side] = frame;
	return (
		<svg
			className={cn('iso-glyph', className)}
			width={size}
			height={size}
			viewBox={`${cx - side / 2} ${cy - side / 2} ${side} ${side}`}
			aria-hidden="true"
			focusable="false"
			shapeRendering="geometricPrecision"
		>
			{children}
		</svg>
	);
}

/** The part of an `IsoGlyph` that moves to the glyph's `--pose` while its host is hovered. */
export function IsoPart({ children }: { children: ReactNode }) {
	return <g className="iso-part">{children}</g>;
}
