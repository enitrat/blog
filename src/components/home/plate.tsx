import Image, { type StaticImageData } from 'next/image';
import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';

const RATIOS = {
	'2/3': 'aspect-2/3',
	'1/1': 'aspect-square',
	'3/2': 'aspect-3/2',
} as const;

/**
 * One object on a tile (a cover, artwork): a hairline media frame with shadow-xs. An image via `src`,
 * or typographic content as children. `glow` puts a blurred, saturated copy of the image behind it.
 */
export function Plate({
	src,
	alt = '',
	ratio = '2/3',
	glow = false,
	sizes = '16rem',
	className,
	children,
}: {
	src?: StaticImageData;
	/** Use "" when a caption sits beside it. */
	alt?: string;
	ratio?: keyof typeof RATIOS;
	glow?: boolean;
	sizes?: string;
	className?: string;
	/** Content when there is no `src`. */
	children?: ReactNode;
}) {
	return (
		<div className={cn('relative isolate w-full', RATIOS[ratio], className)}>
			{src && glow && (
				<Image
					aria-hidden
					src={src}
					sizes={sizes}
					alt=""
					className="absolute inset-0 -z-10 size-full translate-y-2 rounded-md object-cover opacity-70 blur-lg saturate-[6] dark:opacity-15 dark:saturate-[12]"
				/>
			)}
			<div className="relative size-full overflow-hidden rounded-md border border-iron-950/10 bg-gray-100 shadow-xs dark:border-white/10 dark:bg-iron-800">
				{src ? (
					<Image src={src} sizes={sizes} alt={alt} className="size-full object-cover" />
				) : (
					children
				)}
			</div>
		</div>
	);
}
