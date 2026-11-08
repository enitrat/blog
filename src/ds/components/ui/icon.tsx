import { cn } from '@/ds/lib/cn';

const STROKE = {
	'arrow-right': 'M5 12h14M13 6l6 6-6 6',
	'arrow-up': 'M12 19V5M6 11l6-6 6 6',
	external: 'M9 5H5v14h14v-4M13 5h6v6M19 5l-8 8',
	'chevron-left': 'M15 5l-7 7 7 7',
	'chevron-right': 'M9 5l7 7-7 7',
} as const;
const FILL = {
	play: 'M7 4.5v15l12-7.5z',
	pause: 'M6 4.5h4v15H6zM14 4.5h4v15h-4z',
} as const;

const SIZES = {
	md: 'size-(--icon-md) align-(--icon-md-align)',
	sm: 'size-(--icon-sm) align-(--icon-sm-align)',
} as const;

export type IconName = keyof typeof STROKE | keyof typeof FILL;

/** Inline SVG glyph in `currentColor`, hidden from assistive tech unless it has a `title`. */
export function Icon({
	name,
	size = 'md',
	title,
	className,
}: {
	name: IconName;
	size?: keyof typeof SIZES;
	title?: string;
	className?: string;
}) {
	const filled = name in FILL;
	return (
		<svg
			data-slot="icon"
			className={cn(
				'icon inline-block flex-none',
				SIZES[size],
				filled
					? 'fill-current stroke-none'
					: 'fill-none stroke-current [stroke-linecap:round] [stroke-linejoin:round] stroke-(length:--icon-stroke)',
				className,
			)}
			viewBox="0 0 24 24"
			data-size={size}
			data-kind={filled ? 'fill' : 'stroke'}
			aria-hidden={title ? undefined : 'true'}
			role={title ? 'img' : undefined}
			aria-label={title}
			focusable="false"
		>
			<path d={filled ? FILL[name as keyof typeof FILL] : STROKE[name as keyof typeof STROKE]} />
		</svg>
	);
}
