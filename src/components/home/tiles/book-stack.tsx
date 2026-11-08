'use client';

import { motion, useReducedMotion } from 'motion/react';
import Image, { type StaticImageData } from 'next/image';
import { useCardHeld } from '@/components/ui/card';
import { spring } from '@/lib/motion';

type Pose = { x: number; y: number; rotate: number; scale: number };

/** Behind the front book at rest, and fanned out while the card is held; index 0 is the front. */
const REST: Pose[] = [
	{ x: 0, y: 0, rotate: 0, scale: 1 },
	{ x: 14, y: 4, rotate: 7, scale: 0.92 },
	{ x: -14, y: 6, rotate: -7, scale: 0.88 },
];
const HELD: Pose[] = [
	{ x: 0, y: -6, rotate: -1, scale: 1 },
	{ x: 34, y: 8, rotate: 16, scale: 0.92 },
	{ x: -34, y: 10, rotate: -16, scale: 0.88 },
];

/**
 * Covers piled in the middle of the card, sized to the room it is given. Held (hover or focus),
 * the pile fans. Under reduced motion the pose changes without a spring.
 */
export function BookStack({
	books,
}: {
	books: readonly { id: string; cover: StaticImageData | undefined }[];
}) {
	const held = useCardHeld();
	const reduce = useReducedMotion();
	const poses = held ? HELD : REST;
	return (
		<div className="relative mt-3 min-h-0 flex-1 [container-type:size]">
			{books
				.map((b, i) => ({ ...b, i }))
				.reverse()
				.map(({ id, cover, i }) => (
					<motion.div
						key={id}
						className="absolute top-1/2 left-1/2 aspect-2/3 -translate-x-1/2 -translate-y-1/2 overflow-hidden rounded-[0.25rem_0.375rem_0.375rem_0.25rem] bg-muted shadow-book"
						style={{ height: 'min(100cqh - 12px, 86cqw * 1.5)', zIndex: books.length - i }}
						initial={false}
						animate={poses[i]}
						transition={reduce ? { duration: 0 } : spring.fan}
					>
						{cover && <Image src={cover} alt="" sizes="12rem" className="size-full object-cover" />}
						<span
							aria-hidden
							className="absolute inset-0 bg-[linear-gradient(90deg,oklch(0_0_0/.18),transparent_6%,oklch(1_0_0/.08)_8%,transparent_14%)]"
						/>
					</motion.div>
				))}
		</div>
	);
}
