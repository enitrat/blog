'use client';

import { motion } from 'motion/react';
import type { ComponentProps } from 'react';
import { hoverScale, spring } from '@/lib/motion';

/** Springs to scale 1.02 on hover (1.08 with `small`). Render-only wrapper; children stay server-rendered. */
export function HoverScale({
	small = false,
	...props
}: ComponentProps<typeof motion.div> & { small?: boolean }) {
	return (
		<motion.div
			whileHover={{ scale: small ? hoverScale.small : hoverScale.surface }}
			transition={small ? spring.pop : spring.hover}
			{...props}
		/>
	);
}
