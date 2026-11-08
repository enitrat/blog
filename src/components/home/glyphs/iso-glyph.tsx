import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';

/**
 * A tiny isometric object seen from above: one projection, four tones, 1px hairlines. Draw inside
 * it with the helpers in `iso.ts` and the classes `t` (top), `l` (left), `r` (right).
 *
 * `frame` is `[cx, cy, side]`, the square window onto the drawing, so each object fills about 85%
 * of its box and stays centred, hover pose included.
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
			className={cn(
				'block flex-none overflow-visible',
				'[&_*]:stroke-gray-500 [&_*]:[stroke-linecap:round] [&_*]:[stroke-linejoin:round] [&_*]:[stroke-width:1px] [&_*]:[vector-effect:non-scaling-stroke] dark:[&_*]:stroke-iron-400',
				'[&_.t]:fill-white [&_.l]:fill-gray-100 [&_.r]:fill-gray-200 dark:[&_.l]:fill-iron-700 dark:[&_.r]:fill-iron-800 dark:[&_.t]:fill-iron-600',
				className,
			)}
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

/**
 * The part of an `IsoGlyph` that moves to its `pose` (transform utilities behind
 * `group-hover/glyph:`) while the nearest `group/glyph` is hovered. Still under reduced motion.
 */
export function IsoPart({ pose, children }: { pose: string; children: ReactNode }) {
	return (
		<g
			className={cn(
				'origin-[0_0] transition-transform duration-300 ease-[cubic-bezier(.34,1.4,.64,1)] motion-reduce:transition-none',
				pose,
			)}
		>
			{children}
		</g>
	);
}
