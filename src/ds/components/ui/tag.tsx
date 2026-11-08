import type { ComponentProps } from 'react';
import { cn } from '@/ds/lib/cn';

const TONES = {
	neutral: 'bg-foreground text-surface',
	accent: 'bg-accent text-accent-foreground',
} as const;

export type TagTone = keyof typeof TONES;

/** Tiny status flag, not interactive ("Reading", "Latest"). */
export function Tag({
	tone = 'neutral',
	className,
	...props
}: ComponentProps<'span'> & { tone?: TagTone }) {
	return (
		<span
			data-slot="tag"
			data-tone={tone}
			className={cn(
				'type-label inline-flex items-center whitespace-nowrap rounded-pill px-2 py-0.5',
				TONES[tone],
				className,
			)}
			{...props}
		/>
	);
}
