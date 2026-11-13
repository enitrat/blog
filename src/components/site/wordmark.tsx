'use client';

import { motion } from 'motion/react';
import { spring } from '@/lib/motion';
import { cn } from '@/lib/utils';
import { LETTERS, M_GLYPH, SPARK } from './wordmark-paths';

const turn = { rest: { rotate: 0, scale: 1 }, hover: { rotate: 90, scale: 1.15 } };

/** One paint layer: the letters and the sparkle, in the same colours. */
function Layer({ className }: { className: string }) {
	return (
		<>
			<path d={LETTERS} className={className} />
			<motion.path d={SPARK} variants={turn} transition={spring.pop} className={className} />
		</>
	);
}

/**
 * The site wordmark: white letters in a fat outline with a hard shadow, like a sticker. Its
 * colours are the sticker roles, the same in both themes. Each layer is painted whole before the
 * next, so no shadow lands on an outline and no outline on a fill. The sparkle turns a quarter on
 * hover.
 */
export function Wordmark({ className }: { className?: string }) {
	return (
		<motion.svg
			viewBox="-10.5 -10.9 469.1 211.1"
			aria-hidden
			initial="rest"
			animate="rest"
			whileHover="hover"
			className={cn('h-10 w-auto flex-none', className)}
		>
			<g
				transform="translate(220.4 97) rotate(-8) skewX(-4.5) translate(-220.4 -97)"
				strokeWidth={18}
				strokeLinejoin="round"
			>
				<g transform="translate(5.25 7.5)">
					<Layer className="fill-sticker-shadow stroke-sticker-shadow" />
				</g>
				<Layer className="fill-none stroke-sticker-outline" />
				<Layer className="fill-sticker-fill" />
			</g>
		</motion.svg>
	);
}

/**
 * The wordmark's M as a sticker of its own, square, for where the two-line wordmark cannot be read:
 * the favicon is this mark at 16px. Same layers, colours, tilt and slant as the wordmark;
 * `public/favicon.svg` is a static copy, kept in step by `src/lib/behaviours.test.ts`.
 */
export function StickerMark({ className }: { className?: string }) {
	return (
		<svg viewBox="-16 -6 120 120" aria-hidden className={cn('size-7 flex-none', className)}>
			<g
				transform="translate(44 54) rotate(-8) skewX(-4.5) translate(-44 -54)"
				strokeWidth={18}
				strokeLinejoin="round"
			>
				<path
					d={M_GLYPH}
					transform="translate(5.25 7.5)"
					className="fill-sticker-shadow stroke-sticker-shadow"
				/>
				<path d={M_GLYPH} className="fill-none stroke-sticker-outline" />
				<path d={M_GLYPH} className="fill-sticker-fill" />
			</g>
		</svg>
	);
}
