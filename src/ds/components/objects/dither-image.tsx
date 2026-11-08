'use client';

import Image, { type StaticImageData } from 'next/image';
import { type CSSProperties, useEffect, useRef } from 'react';
import { initDither } from './dither';

export type DitherSource = StaticImageData;

/**
 * An image shown as an ordered-dither duotone that reveals its true colours on hover, focus or
 * tap. Without JS it is a plain responsive image.
 */
export function DitherImage({
	image,
	alt,
	ratio = `${image.width} / ${image.height}`,
	sizes = '(min-width: 60rem) 38rem, 100vw',
	tone = 'ink',
	reveal = 'hover',
}: {
	image: DitherSource;
	alt: string;
	/** CSS aspect-ratio, such as "4 / 3"; the image's own by default. */
	ratio?: string;
	sizes?: string;
	/** ink: foreground on surface. accent: accent on surface. The darker colour takes the shadows. */
	tone?: 'ink' | 'accent';
	/** hover: the script wraps the image in a pressable button. never: stays dithered. */
	reveal?: 'hover' | 'never';
}) {
	const ref = useRef<HTMLDivElement>(null);
	useEffect(() => (ref.current ? initDither(ref.current) : undefined), []);
	return (
		<div
			ref={ref}
			data-slot="dither"
			data-dither=""
			data-tone={tone}
			data-reveal={reveal}
			className="dither"
			style={{ '--dither-ratio': ratio } as CSSProperties}
		>
			<Image src={image} sizes={sizes} alt={alt} />
			{/* biome-ignore lint/a11y/noAriaHiddenOnFocusable: the canvas is never focusable; it repeats the image */}
			<canvas aria-hidden="true" />
		</div>
	);
}
