'use client';

import { type ReactNode, useEffect, useRef } from 'react';
import { cn } from '@/ds/lib/cn';
import { registerMagnetic } from './pointer';

/**
 * Its child drifts toward the pointer while the pointer is within its box plus 16px, and springs
 * home on leave. Use it on a button or pill; keep `strength` at 0.15 to 0.2 on small targets.
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
	useEffect(() => (ref.current ? registerMagnetic(ref.current) : undefined), []);
	return (
		<span
			ref={ref}
			className={cn('magnetic inline-flex', className)}
			data-magnetic=""
			data-magnetic-strength={strength}
		>
			{children}
		</span>
	);
}
