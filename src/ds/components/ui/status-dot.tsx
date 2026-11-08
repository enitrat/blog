import { cn } from '@/ds/lib/cn';

const TONES = {
	accent: 'bg-accent-graphic',
	muted: 'bg-muted-foreground opacity-60',
} as const;

export type StatusDotTone = keyof typeof TONES;

/** State marker. Accent means playing, current or latest. `live` pulses softly, never under reduced motion. */
export function StatusDot({
	tone = 'accent',
	live = false,
	className,
}: {
	tone?: StatusDotTone;
	live?: boolean;
	className?: string;
}) {
	return (
		<span
			data-slot="status-dot"
			data-tone={tone}
			data-live={live ? '' : undefined}
			aria-hidden="true"
			className={cn(
				'inline-block size-(--dot-size) flex-none rounded-pill',
				TONES[tone],
				live && 'motion-safe:animate-pulse-soft',
				className,
			)}
		/>
	);
}
