'use client';

import { motion, useReducedMotion } from 'motion/react';
import { createContext, type ReactNode, useContext, useState } from 'react';
import { spring } from '@/lib/motion';
import { cn } from '@/lib/utils';

/** Milliseconds between neighbouring tiles popping in. */
const POP_STAGGER = 45;

const HeldContext = createContext(false);

/**
 * True while the tile is held (hovered, or keyboard focus inside) and motion is welcome. A tile's
 * signature object takes its awake pose from this, so under reduced motion no object strikes a pose
 * at all; colour cues key off `group-data-[held]/tile` and still answer.
 */
export function useTileAwake() {
	const held = useContext(HeldContext);
	const still = useReducedMotion();
	return held && !still;
}

/**
 * A home tile: a bezel (a card-coloured frame) around a grey well that holds the content. Under 36rem
 * of bento the tile is `phone` tall, a height class chosen for what it holds; wider, the bento's
 * tracks size it.
 *
 * Motion has three layers, and every tile follows them:
 * - Surface: held, the bezel rises 2px and a deeper shadow cross-fades in. A `lift` tile, one whose
 *   surface is a link, also presses on tap.
 * - Object: the tile's one signature object wakes from `useTileAwake()`, on `spring.fan`.
 * - Affordance: arrows nudge, glows brighten and titles take the accent, in CSS keyed to
 *   `group-data-[held]/tile`.
 * The tile pops in once, in CSS so it never waits for hydration, staggered by `index`.
 */
export function BentoTile({
	index,
	lift = false,
	phone = 'h-64',
	labelledBy,
	className,
	children,
}: {
	index: number;
	lift?: boolean;
	/** Height class under 36rem of bento, where the tile sits alone or in a pair. */
	phone?: string;
	/** Id of the tile's eyebrow heading. */
	labelledBy: string;
	className?: string;
	children: ReactNode;
}) {
	const [hovered, setHovered] = useState(false);
	const [focused, setFocused] = useState(false);
	const still = useReducedMotion();
	const held = hovered || focused;
	const rise = held && !still;
	return (
		<HeldContext.Provider value={held}>
			<div
				className={cn('min-h-0 min-w-0 animate-tile-in @[36rem]/bento:h-auto', phone, className)}
				style={{ animationDelay: `${index * POP_STAGGER}ms` }}
			>
				<motion.section
					aria-labelledby={labelledBy}
					data-held={held || undefined}
					className={cn(
						'group/tile group/card relative h-full rounded-bezel bg-card p-bezel shadow-bezel',
						// The press is CSS: motion's whileTap would make the section itself a tab stop that does
						// nothing, beside the link that already stretches over it.
						lift && 'cursor-pointer transition-[scale] duration-150 ease-out active:scale-[0.985]',
					)}
					initial={false}
					animate={{ y: rise ? -2 : 0 }}
					transition={spring.hover}
					onHoverStart={() => setHovered(true)}
					onHoverEnd={() => setHovered(false)}
					onFocus={(e) => setFocused(e.target.matches(':focus-visible'))}
					onBlur={() => setFocused(false)}
				>
					<span
						aria-hidden
						className="pointer-events-none absolute inset-0 rounded-bezel opacity-0 shadow-bezel-raised transition-opacity duration-200 ease-out group-data-[held]/tile:opacity-100"
					/>
					<div className="relative flex h-full min-h-0 flex-col overflow-hidden rounded-well bg-well p-inset shadow-well">
						{children}
					</div>
				</motion.section>
			</div>
		</HeldContext.Provider>
	);
}
