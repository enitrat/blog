'use client';

/*
 * Adapted from motion-primitives (https://github.com/ibelick/motion-primitives,
 * components/core/magnetic.tsx). MIT License, Copyright (c) ibelick.
 * Changed: the site's hover spring, only the element itself is the action area, pointer events
 * that ignore touch, and no pull under reduced motion.
 */
import { motion, useMotionValue, useReducedMotion, useSpring } from 'motion/react';
import { type PointerEvent, type ReactNode, useRef } from 'react';
import { spring } from '@/lib/motion';

const { type: _, ...springOptions } = spring.hover;

/** Drifts toward the mouse while hovered and springs home on leave. Inert on touch and under reduced motion. */
export function Magnetic({
	children,
	intensity = 0.6,
	range = 100,
	className,
}: {
	children: ReactNode;
	/** Share of the pointer's offset from the centre that the content follows. */
	intensity?: number;
	/** Distance from the centre, px, past which the pull fades to nothing. */
	range?: number;
	className?: string;
}) {
	const ref = useRef<HTMLDivElement>(null);
	const reduce = useReducedMotion();
	const x = useMotionValue(0);
	const y = useMotionValue(0);
	const springX = useSpring(x, springOptions);
	const springY = useSpring(y, springOptions);

	const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
		if (e.pointerType !== 'mouse' || reduce || !ref.current) return;
		const rect = ref.current.getBoundingClientRect();
		const dx = e.clientX - (rect.left + rect.width / 2);
		const dy = e.clientY - (rect.top + rect.height / 2);
		const scale = Math.max(0, 1 - Math.hypot(dx, dy) / range);
		x.set(dx * intensity * scale);
		y.set(dy * intensity * scale);
	};
	const onPointerLeave = () => {
		x.set(0);
		y.set(0);
	};

	return (
		<motion.div
			ref={ref}
			className={className}
			style={{ x: springX, y: springY }}
			onPointerMove={onPointerMove}
			onPointerLeave={onPointerLeave}
		>
			{children}
		</motion.div>
	);
}
