'use client';

import { motion, useReducedMotion } from 'motion/react';
import type { ReactNode } from 'react';
import { hoverScale, spring } from '@/lib/motion';
import { cn } from '@/lib/utils';

/** Seconds between neighbouring cells popping in. */
const POP_STAGGER = 0.045;

/**
 * One cell of the home grid. It pops in on mount, staggered by `index`, and springs to scale 1.02
 * on hover. With `drag` it can be picked up (scale 1.1, grabbing cursor) and snaps home on release.
 * Under reduced motion it renders in place and does not move.
 */
export function Cell({
	index,
	hover = true,
	drag = false,
	className,
	children,
}: {
	/** Position in the pop-in order. */
	index: number;
	/** Off where the child already springs on hover (a linked tile). */
	hover?: boolean;
	drag?: boolean;
	className?: string;
	children: ReactNode;
}) {
	const reduce = useReducedMotion();
	const motile = !reduce;
	return (
		<motion.div
			className={cn('min-h-0 min-w-0', drag && motile && 'cursor-grab', className)}
			initial={motile ? { opacity: 0, scale: 0.9, y: -20 } : false}
			animate={{ opacity: 1, scale: 1, y: 0 }}
			transition={{ ...spring.enter, delay: index * POP_STAGGER }}
			whileHover={
				motile && hover ? { scale: hoverScale.surface, transition: spring.hover } : undefined
			}
			drag={drag && motile}
			dragSnapToOrigin
			dragElastic={0.2}
			dragMomentum={false}
			whileDrag={{ scale: 1.1, zIndex: 30, cursor: 'grabbing', transition: spring.hover }}
		>
			{children}
		</motion.div>
	);
}
