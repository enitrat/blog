'use client';

import { type ReactNode, useEffect, useRef } from 'react';
import { cn } from '@/ds/lib/cn';
import { initTilt } from './pointer';

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
	const ref = useRef<HTMLDivElement>(null);
	useEffect(() => (ref.current ? initTilt(ref.current) : undefined), []);
	return (
		<div
			ref={ref}
			className={cn(
				'tilt relative rounded-[inherit] data-tilt-active:will-change-transform',
				className,
			)}
			data-tilt=""
			data-tilt-max={max}
		>
			{children}
			{glare && <span aria-hidden="true" className="glare tilt-glare" />}
		</div>
	);
}
