'use client';

import { motion, useSpring } from 'motion/react';
import { type ReactNode, useEffect, useRef } from 'react';
import { cn } from '@/ds/lib/cn';
import { canHover, prefersReducedMotion, SPRING_MOUSE } from '@/lib/motion';

/** How far outside its box the pull reaches, px. */
const MARGIN = 16;

const live = () => canHover() && !prefersReducedMotion();

/**
 * Its content drifts toward the pointer while the pointer is within its box plus 16px, and springs
 * home on leave. Use it on a button or pill; keep `strength` at 0.15 to 0.2 on small targets.
 * The wrapper never moves, so its rect is the home box; inert on touch and under reduced motion.
 */
export function Magnetic({
	strength = 0.35,
	className,
	children,
}: {
	strength?: number;
	className?: string;
	children: ReactNode;
}) {
	const ref = useRef<HTMLSpanElement>(null);
	const x = useSpring(0, SPRING_MOUSE);
	const y = useSpring(0, SPRING_MOUSE);

	// The pull reaches past the wrapper's box, so it listens on the document. The home box is
	// measured lazily, once, and dropped when layout may have moved under a still pointer.
	useEffect(() => {
		const el = ref.current;
		if (!el) return;
		let home: { cx: number; cy: number; hw: number; hh: number } | null = null;
		let on = false;
		let last: PointerEvent | null = null;
		let queued = 0;
		const run = () => {
			queued = 0;
			if (!last) return;
			if (!home) {
				const r = el.getBoundingClientRect();
				home = {
					cx: r.left + r.width / 2,
					cy: r.top + r.height / 2,
					hw: r.width / 2 + MARGIN,
					hh: r.height / 2 + MARGIN,
				};
			}
			const dx = last.clientX - home.cx;
			const dy = last.clientY - home.cy;
			const inside = Math.abs(dx) <= home.hw && Math.abs(dy) <= home.hh;
			if (!inside && !on) return;
			on = inside;
			x.set(inside ? dx * strength : 0);
			y.set(inside ? dy * strength : 0);
		};
		const onMove = (e: PointerEvent) => {
			if (e.pointerType !== 'mouse' || !live()) return;
			last = e;
			queued ||= requestAnimationFrame(run);
		};
		const release = () => {
			last = null;
			if (!on) return;
			on = false;
			x.set(0);
			y.set(0);
		};
		const invalidate = () => {
			home = null;
			release();
		};
		const resized = new ResizeObserver(() => {
			home = null;
		});
		resized.observe(document.documentElement);
		document.addEventListener('pointermove', onMove);
		addEventListener('scroll', invalidate, { passive: true, capture: true });
		addEventListener('resize', invalidate, { passive: true });
		document.documentElement.addEventListener('pointerleave', release);
		return () => {
			cancelAnimationFrame(queued);
			resized.disconnect();
			document.removeEventListener('pointermove', onMove);
			removeEventListener('scroll', invalidate, { capture: true });
			removeEventListener('resize', invalidate);
			document.documentElement.removeEventListener('pointerleave', release);
		};
	}, [strength, x, y]);

	return (
		<span ref={ref} className={cn('magnetic inline-flex', className)} data-magnetic="">
			<motion.span className="inline-flex" style={{ x, y }}>
				{children}
			</motion.span>
		</span>
	);
}
