'use client';

/*
 * Based on Bencho's Tilt block (https://github.com/cabralorenzo/bencho, blocks/tilt), MIT,
 * Copyright (c) Lorenzo Cabra. Kept from it: one sprung pointer position is the whole state, and
 * the lean and the glare are both read off it; the frame, which never moves, is what the pointer
 * is measured against; leaving is `pointerout` with a containment test; the card sinks a little
 * while touched. Changed: springs and rendering come from `motion/react`, the light is the site's
 * glare rather than Bencho's dent and rim, and it stays inert on touch.
 */
import { motion, useMotionTemplate, useSpring, useTransform } from 'motion/react';
import { type PointerEvent, type ReactNode, useEffect, useRef, useState } from 'react';
import { cn } from '@/ds/lib/cn';
import { canHover, prefersReducedMotion, SPRING_MOUSE } from '@/lib/motion';

/** Perspective of the lean, px. */
const DEPTH = 1000;
/** How far the object retreats while it is touched, px. Past about twenty it reads as a zoom. */
const SINK = 14;

const live = () => canHover() && !prefersReducedMotion();
const clamp = (v: number) => Math.min(1, Math.max(-1, v));

/**
 * A physical object that leans toward the pointer and settles flat on leave. Wrap a `Plate`.
 * Put an animation (a spin) on an inner element, never on the Tilt. Inert on touch and under
 * reduced motion. `--tilt-radius` sets the glare's corner, `--glare-strength` its opacity.
 */
export function Tilt({
	max = 7.5,
	glare = true,
	className,
	children,
}: {
	/** Degrees of lean at the edge. */
	max?: number;
	glare?: boolean;
	className?: string;
	children: ReactNode;
}) {
	const frame = useRef<HTMLDivElement>(null);
	const [active, setActive] = useState(false);
	// Where the pointer is over the frame, -1..1 on each axis, and how much of the touch applies.
	const x = useSpring(0, SPRING_MOUSE);
	const y = useSpring(0, SPRING_MOUSE);
	const lit = useSpring(0, SPRING_MOUSE);
	const rotateX = useTransform(y, (v) => -v * max);
	const rotateY = useTransform(x, (v) => v * max);
	const z = useTransform(lit, (v) => -v * SINK);
	const glareX = useMotionTemplate`${useTransform(x, (v) => (v + 1) * 50)}%`;
	const glareY = useMotionTemplate`${useTransform(y, (v) => (v + 1) * 50)}%`;
	const glareAt = { '--glare-x': glareX, '--glare-y': glareY };

	// One read per hover: measured on the first move after entering, dropped when layout may move.
	const rect = useRef<DOMRect | null>(null);
	useEffect(() => {
		const drop = () => {
			rect.current = null;
		};
		addEventListener('scroll', drop, { passive: true, capture: true });
		addEventListener('resize', drop, { passive: true });
		addEventListener('transitionend', drop, { capture: true });
		return () => {
			removeEventListener('scroll', drop, { capture: true });
			removeEventListener('resize', drop);
			removeEventListener('transitionend', drop, { capture: true });
		};
	}, []);

	const track = (e: PointerEvent) => {
		if (e.pointerType !== 'mouse' || !live() || !frame.current) return;
		rect.current ??= frame.current.getBoundingClientRect();
		const r = rect.current;
		x.set(clamp(((e.clientX - r.left) / r.width) * 2 - 1));
		y.set(clamp(((e.clientY - r.top) / r.height) * 2 - 1));
		lit.set(1);
		setActive(true);
	};
	const release = () => {
		rect.current = null;
		if (!active) return;
		x.set(0);
		y.set(0);
		lit.set(0);
		setActive(false);
	};

	return (
		<div
			ref={frame}
			className="tilt-frame relative rounded-[inherit]"
			onPointerEnter={() => {
				rect.current = null;
			}}
			onPointerMove={track}
			onPointerOut={(e) => {
				const to = e.relatedTarget as Node | null;
				if (!to || !frame.current?.contains(to)) release();
			}}
			onPointerCancel={release}
		>
			<motion.div
				className={cn(
					'tilt relative rounded-[inherit] data-tilt-active:will-change-transform',
					className,
				)}
				data-tilt=""
				data-tilt-active={active ? '' : undefined}
				style={{
					transformPerspective: DEPTH,
					rotateX,
					rotateY,
					z,
					...glareAt,
				}}
			>
				{children}
				{glare && <span aria-hidden="true" className="glare tilt-glare" />}
			</motion.div>
		</div>
	);
}
