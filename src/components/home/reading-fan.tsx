'use client';

import { motion, useReducedMotion } from 'motion/react';
import type { StaticImageData } from 'next/image';
import { useTileHeld } from '@/components/home/bento-tile';
import { GlowImage } from '@/components/ui/glow-image';
import { spring } from '@/lib/motion';

/** Index 0 is the front cover; the second sits tucked behind it, and both fan while the tile is held. */
const REST = [
	{ x: -6, y: 0, rotate: -2 },
	{ x: 10, y: 2, rotate: 5 },
];
const HELD = [
	{ x: -18, y: -4, rotate: -6 },
	{ x: 22, y: 2, rotate: 8 },
];

/**
 * Up to two covers, each lit by its own glow, sized to the room the tile gives them. Decorative
 * and inert: clicks fall through to the tile's link, and the fan follows the tile being held.
 */
export function ReadingFan({ covers }: { covers: { id: string; cover: StaticImageData }[] }) {
	const reduce = useReducedMotion();
	const held = useTileHeld();
	const poses = held && !reduce ? HELD : REST;
	return (
		<div aria-hidden className="pointer-events-none relative min-h-0 flex-1 [container-type:size]">
			{covers
				.slice(0, 2)
				.map((c, i) => ({ ...c, i }))
				.reverse()
				.map(({ id, cover, i }) => (
					<motion.div
						key={id}
						className="absolute top-1/2 left-1/2 aspect-2/3 -translate-x-1/2 -translate-y-1/2"
						style={{ height: 'min(88cqh, 95cqw)', zIndex: 2 - i }}
						initial={false}
						animate={poses[i]}
						transition={spring.fan}
					>
						<GlowImage
							src={cover}
							sizes="96px"
							className="size-full"
							imageClassName="rounded-[0.1875rem_0.3125rem_0.3125rem_0.1875rem] shadow-book"
						/>
					</motion.div>
				))}
		</div>
	);
}
