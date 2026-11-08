'use client';

import { type ComponentProps, useEffect, useRef } from 'react';
import { cn } from '@/ds/lib/cn';
import { initGlide } from './glide-controller';

/**
 * A group of items (each `data-glide-item`) sharing ONE highlight that glides to the hovered or
 * keyboard-focused item. Without JS the items keep their own hover styles.
 */
export function Glide({
	as = 'div',
	shape = 'pill',
	className,
	...props
}: ComponentProps<'div'> & { as?: 'div' | 'ul' | 'nav'; shape?: 'pill' | 'row' }) {
	const Tag = as as 'div';
	const ref = useRef<HTMLDivElement>(null);
	useEffect(() => (ref.current ? initGlide(ref.current) : undefined), []);
	return (
		<Tag
			ref={ref}
			data-glide=""
			data-shape={shape}
			className={cn('glide glide-highlight', className)}
			{...props}
		/>
	);
}
