'use client';

import { motion, useReducedMotion } from 'motion/react';
import { createContext, type ReactNode, useContext, useState } from 'react';
import { spring } from '@/lib/motion';
import { cn } from '@/lib/utils';

/** Milliseconds between neighbouring tiles popping in. */
const POP_STAGGER = 45;

const HeldContext = createContext(false);

/** True while the pointer is over the tile or keyboard focus is inside it. */
export const useTileHeld = () => useContext(HeldContext);

/**
 * A home tile: a bezel (a card-coloured frame) around a grey well that holds the content.
 *
 * Motion has three layers, and every tile follows them:
 * - Surface: held (hovered, or keyboard focus inside), the bezel rises 2px and a deeper shadow
 *   cross-fades in. A `lift` tile, one whose surface is a link, also presses on tap.
 * - Object: the tile's one signature object wakes, reading `useTileHeld()` and moving on
 *   `spring.fan`.
 * - Affordance: arrows nudge, glows brighten and titles take the accent, in CSS keyed to
 *   `group-data-[held]/tile`.
 * The tile pops in once, in CSS so it never waits for hydration, staggered by `index`. Under reduced motion nothing moves.
 */
export function BentoTile({
	index,
	lift = false,
	labelledBy,
	className,
	children,
}: {
	index: number;
	lift?: boolean;
	/** Id of the tile's eyebrow heading. */
	labelledBy: string;
	className?: string;
	children: ReactNode;
}) {
	const reduce = useReducedMotion();
	const [hovered, setHovered] = useState(false);
	const [focused, setFocused] = useState(false);
	const held = hovered || focused;
	return (
		<HeldContext.Provider value={held}>
			<div
				className={cn('min-h-0 min-w-0 animate-tile-in motion-reduce:animate-none', className)}
				style={{ animationDelay: `${index * POP_STAGGER}ms` }}
			>
				<motion.section
					aria-labelledby={labelledBy}
					data-held={held || undefined}
					className={cn(
						'group/tile group/card relative h-full rounded-bezel bg-card p-bezel shadow-bezel',
						lift && 'cursor-pointer',
					)}
					initial={false}
					animate={{ y: held && !reduce ? -2 : 0 }}
					transition={spring.hover}
					whileTap={!reduce && lift ? { scale: 0.985, transition: spring.tap } : undefined}
					onHoverStart={() => setHovered(true)}
					onHoverEnd={() => setHovered(false)}
					onFocus={(e) => setFocused(e.target.matches(':focus-visible'))}
					onBlur={() => setFocused(false)}
				>
					<span
						aria-hidden
						className="pointer-events-none absolute inset-0 rounded-bezel opacity-0 shadow-bezel-raised transition-opacity duration-200 ease-out group-data-[held]/tile:opacity-100 motion-reduce:transition-none"
					/>
					<div className="relative flex h-full min-h-0 flex-col overflow-hidden rounded-well bg-well p-inset shadow-well">
						{children}
					</div>
				</motion.section>
			</div>
		</HeldContext.Provider>
	);
}
