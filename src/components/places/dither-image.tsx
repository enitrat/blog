'use client';

import Image, { type StaticImageData } from 'next/image';
import { useTheme } from 'next-themes';
import { useEffect, useRef, useState } from 'react';
import { cn } from '@/lib/utils';
import { paint } from './dither';

/** Ink and paper per theme, darker first: gray-800 on gray-50, iron-900 under iron-300. */
const DUOTONE = {
	light: [
		[0x18, 0x22, 0x30],
		[0xf9, 0xfa, 0xfb],
	],
	dark: [
		[0x1a, 0x1a, 0x1e],
		[0xd1, 0xd1, 0xd6],
	],
} as const;

/** Milliseconds a resize settles before the redraw. */
const SETTLE = 120;

/**
 * An image shown as an ordered-dither duotone that reveals its true colours on hover, focus or
 * tap. Before the canvas paints it is a plain responsive image.
 */
export function DitherImage({
	image,
	alt,
	ratio = `${image.width} / ${image.height}`,
	sizes = '(min-width: 60rem) 38rem, 100vw',
	className,
}: {
	image: StaticImageData;
	alt: string;
	/** CSS aspect-ratio, such as "4 / 3"; the image's own by default. */
	ratio?: string;
	sizes?: string;
	className?: string;
}) {
	const img = useRef<HTMLImageElement>(null);
	const canvas = useRef<HTMLCanvasElement>(null);
	const [ready, setReady] = useState(false);
	const [revealed, setRevealed] = useState(false);
	const { resolvedTheme } = useTheme();
	const [dark, light] = DUOTONE[resolvedTheme === 'dark' ? 'dark' : 'light'];

	useEffect(() => {
		const cv = canvas.current;
		if (!img.current || !cv) return;
		const draw = () => {
			if (img.current && paint(img.current, cv, dark, light)) setReady(true);
		};
		draw();
		let timer = 0;
		const ro = new ResizeObserver(() => {
			clearTimeout(timer);
			timer = window.setTimeout(draw, SETTLE);
		});
		ro.observe(cv);
		img.current.addEventListener('load', draw);
		const el = img.current;
		return () => {
			clearTimeout(timer);
			ro.disconnect();
			el.removeEventListener('load', draw);
		};
	}, [dark, light]);

	return (
		<button
			type="button"
			aria-pressed={revealed}
			onClick={() => setRevealed((r) => !r)}
			className={cn(
				'group relative block w-full cursor-pointer overflow-hidden bg-gray-50 outline-none focus-visible:ring-2 focus-visible:ring-brand-500/60 dark:bg-iron-900',
				className,
			)}
			style={{ aspectRatio: ratio }}
		>
			<Image ref={img} src={image} sizes={sizes} alt={alt} className="size-full object-cover" />
			{/* biome-ignore lint/a11y/noAriaHiddenOnFocusable: the canvas is never focusable; it repeats the image */}
			<canvas
				ref={canvas}
				aria-hidden="true"
				className={cn(
					'pointer-events-none absolute inset-0 size-full opacity-0 transition-opacity duration-150 [image-rendering:pixelated] group-hover:opacity-0 group-focus-visible:opacity-0 motion-reduce:transition-none',
					ready && !revealed && 'opacity-100',
				)}
			/>
		</button>
	);
}
