'use client';

import { motion, useReducedMotion } from 'motion/react';
import { createContext, type ReactNode, useContext, useState } from 'react';
import { spring } from '@/lib/motion';
import { cn } from '@/lib/utils';

/** Seconds between neighbouring cards popping in. */
const POP_STAGGER = 0.045;

const HeldContext = createContext(false);

/** True while the pointer is over the card or focus is inside it. */
export const useCardHeld = () => useContext(HeldContext);

/**
 * The one card: one inset, one radius, one shadow. With an `index` it pops in on mount, staggered
 * by that position; without one it renders in place. A `lift` card rises on hover and presses on tap; every card tells its children when it
 * is held, so a stack inside can fan. Under reduced motion it renders in place and does not move.
 */
export function Card({
	index,
	lift = false,
	labelledBy,
	className,
	children,
}: {
	/** Position in the pop-in order. Omit it and the card does not pop in. */
	index?: number;
	/** The whole card is a link: it rises on hover and presses on tap. */
	lift?: boolean;
	/** Id of the card's eyebrow heading, when it has one. */
	labelledBy?: string;
	className?: string;
	children: ReactNode;
}) {
	const reduce = useReducedMotion();
	const motile = !reduce;
	const [hovered, setHovered] = useState(false);
	const [focused, setFocused] = useState(false);
	return (
		<HeldContext.Provider value={hovered || focused}>
			<motion.section
				aria-labelledby={labelledBy}
				className={cn(
					'group/card relative isolate flex h-full min-h-0 min-w-0 flex-col overflow-hidden rounded-card bg-card p-inset shadow-card',
					lift &&
						'transition-shadow duration-300 hover:shadow-lifted motion-reduce:transition-none',
					className,
				)}
				initial={motile && index !== undefined ? { opacity: 0, y: 14, scale: 0.98 } : false}
				animate={{ opacity: 1, y: 0, scale: 1 }}
				transition={{ ...spring.enter, delay: (index ?? 0) * POP_STAGGER }}
				whileHover={motile && lift ? { y: -3, transition: spring.hover } : undefined}
				whileTap={motile && lift ? { y: -1, scale: 0.985, transition: spring.tap } : undefined}
				onHoverStart={() => setHovered(true)}
				onHoverEnd={() => setHovered(false)}
				onFocus={(e) => setFocused(e.target.matches(':focus-visible'))}
				onBlur={() => setFocused(false)}
			>
				{children}
			</motion.section>
		</HeldContext.Provider>
	);
}

/** The one eyebrow row every card opens with: icon, label, and an optional trailing slot. */
export function CardHeader({
	id,
	icon,
	label,
	onMedia,
	className,
	children,
}: {
	id: string;
	icon: ReactNode;
	label: ReactNode;
	/** Over a photo: light text on the scrim. */
	onMedia?: boolean;
	className?: string;
	children?: ReactNode;
}) {
	return (
		<header
			className={cn(
				'flex h-5 flex-none items-center gap-1.5',
				onMedia ? 'text-scrim-foreground/85' : 'text-faint',
				className,
			)}
		>
			<span
				aria-hidden
				className="grid flex-none place-items-center [&_svg]:size-3.5 [&_svg]:stroke-[1.75]"
			>
				{icon}
			</span>
			<h2 id={id} className="type-eyebrow whitespace-nowrap">
				{label}
			</h2>
			{children && (
				<div
					className={cn(
						// A narrow card truncates the trailing note rather than wrapping the row.
						'ml-auto flex min-w-0 items-center gap-0.5 type-caption whitespace-nowrap [&>*]:min-w-0 [&>*]:truncate [&_svg]:transition-transform [&_svg]:duration-300 motion-safe:group-hover/card:[&_svg]:translate-x-0.5 motion-safe:group-hover/card:[&_svg]:-translate-y-0.5',
						onMedia ? 'text-scrim-foreground' : 'text-muted-foreground',
					)}
				>
					{children}
				</div>
			)}
		</header>
	);
}

/** Scrims over a full-bleed photo. The scrim colour is night in both themes. */
export function Scrims({ bottom = 'h-[70%]' }: { bottom?: string | false }) {
	return (
		<>
			<span
				aria-hidden
				className="pointer-events-none absolute inset-x-0 top-0 -z-10 h-[90px] bg-linear-to-b from-scrim/55 to-transparent"
			/>
			{bottom && (
				<span
					aria-hidden
					className={cn(
						'pointer-events-none absolute inset-x-0 bottom-0 -z-10 bg-linear-to-t from-scrim/90 via-scrim/35 via-65% to-transparent',
						bottom,
					)}
				/>
			)}
		</>
	);
}
