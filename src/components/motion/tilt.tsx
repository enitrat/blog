'use client';

/*
 * Adapted from motion-primitives (https://github.com/ibelick/motion-primitives,
 * components/core/tilt.tsx). MIT License, Copyright (c) ibelick.
 * Changed: the site's hover spring, pointer events that ignore touch, and no tilt under reduced motion.
 */
import {
	type MotionStyle,
	motion,
	useMotionTemplate,
	useMotionValue,
	useReducedMotion,
	useSpring,
	useTransform,
} from 'motion/react';
import { type PointerEvent, type ReactNode, useRef } from 'react';
import { spring } from '@/lib/motion';

const { type: _, ...springOptions } = spring.hover;

/** Leans toward the mouse and settles flat on leave. Inert on touch and under reduced motion. */
export function Tilt({
	children,
	className,
	style,
	rotationFactor = 15,
}: {
	children: ReactNode;
	className?: string;
	style?: MotionStyle;
	/** Degrees of lean at the edge. */
	rotationFactor?: number;
}) {
	const ref = useRef<HTMLDivElement>(null);
	const reduce = useReducedMotion();
	const x = useMotionValue(0);
	const y = useMotionValue(0);
	const xSpring = useSpring(x, springOptions);
	const ySpring = useSpring(y, springOptions);
	const rotateX = useTransform(ySpring, [-0.5, 0.5], [rotationFactor, -rotationFactor]);
	const rotateY = useTransform(xSpring, [-0.5, 0.5], [-rotationFactor, rotationFactor]);
	const transform = useMotionTemplate`perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;

	const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
		if (e.pointerType !== 'mouse' || reduce || !ref.current) return;
		const rect = ref.current.getBoundingClientRect();
		x.set((e.clientX - rect.left) / rect.width - 0.5);
		y.set((e.clientY - rect.top) / rect.height - 0.5);
	};
	const onPointerLeave = () => {
		x.set(0);
		y.set(0);
	};

	return (
		<motion.div
			ref={ref}
			className={className}
			style={{ transformStyle: 'preserve-3d', ...style, transform }}
			onPointerMove={onPointerMove}
			onPointerLeave={onPointerLeave}
		>
			{children}
		</motion.div>
	);
}
