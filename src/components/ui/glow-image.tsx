import Image, { type StaticImageData } from 'next/image';
import { cn } from '@/lib/utils';

/**
 * An image lit by its own colours: a blurred, saturated copy sits behind it as an ambient glow,
 * brightening while its home tile (or the sidebar photo) is held. The glow is blurred past any
 * detail, so it asks for the smallest rendition the image has, not a second full-size one.
 * `imageClassName` shapes the image itself (radius, shadow); the glow follows its box. `eager` for
 * an image that is the page's largest above the fold. `alt` names the image itself; the glow is
 * always decorative.
 */
export function GlowImage({
	src,
	alt = '',
	sizes,
	eager = false,
	className,
	imageClassName,
}: {
	src: StaticImageData;
	/** Empty when the image is decoration, as most glowing images are. */
	alt?: string;
	sizes: string;
	eager?: boolean;
	className?: string;
	imageClassName?: string;
}) {
	const loading = eager ? 'eager' : 'lazy';
	return (
		<span className={cn('relative block', className)}>
			<span
				aria-hidden
				className="absolute inset-[6%] ambient-glow transition-opacity duration-300 ease-out group-hover/photo:opacity-95 group-data-[held]/tile:opacity-95"
			>
				<Image src={src} alt="" sizes="2rem" loading={loading} className="size-full object-cover" />
			</span>
			<span className={cn('relative block size-full overflow-hidden', imageClassName)}>
				<Image
					src={src}
					alt={alt}
					sizes={sizes}
					loading={loading}
					fetchPriority={eager ? 'high' : undefined}
					className="size-full object-cover"
				/>
				<span aria-hidden className="absolute inset-0 rounded-[inherit] shadow-image-edge" />
			</span>
		</span>
	);
}
