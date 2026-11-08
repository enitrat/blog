import Image, { type StaticImageData } from 'next/image';
import type { CSSProperties, ReactNode } from 'react';
import { cn } from '@/ds/lib/cn';

const RATIOS = {
	'2/3': 'aspect-2/3',
	'1/1': 'aspect-square',
	'4/3': 'aspect-4/3',
	'3/2': 'aspect-3/2',
} as const;

export type PlateRatio = keyof typeof RATIOS;

/**
 * One physical object on a tile (a cover, artwork, a card): `--elevation-object`, a hairline and a
 * radius. An image via `src`, or typographic content as children.
 *
 * `glow` is a CSS colour sampled from the image at build time (`glowOf`): a soft halo behind the
 * plate, the one sanctioned coloured glow.
 *
 * Inherited hooks a parent may set: `--plate-shadow` (a Deck swaps in the lift),
 * `--plate-glow-strength`, `--plate-glow-inset`.
 */
export function Plate({
	src,
	alt = '',
	ratio = '2/3',
	radius = 'sm',
	glow,
	sizes = '16rem',
	loading = 'lazy',
	className,
	children,
}: {
	src?: StaticImageData;
	/** Use "" when a caption sits beside it. */
	alt?: string;
	ratio?: PlateRatio;
	radius?: 'sm' | 'md';
	glow?: string;
	sizes?: string;
	loading?: 'lazy' | 'eager';
	className?: string;
	/** Content when there is no `src`. */
	children?: ReactNode;
}) {
	return (
		<div
			className={cn('plate relative w-full', RATIOS[ratio], className)}
			data-radius={radius}
			data-glow={glow ? '' : undefined}
			style={glow ? ({ '--plate-glow': glow } as CSSProperties) : undefined}
		>
			{glow && <span className="plate-glow" aria-hidden="true" />}
			<div className="plate-face">
				{src ? (
					<Image
						className="block size-full object-cover"
						src={src}
						sizes={sizes}
						alt={alt}
						loading={loading}
					/>
				) : (
					children
				)}
			</div>
		</div>
	);
}
