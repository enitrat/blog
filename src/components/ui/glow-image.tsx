import Image, { type StaticImageData } from 'next/image';
import { cn } from '@/lib/utils';

/**
 * An image lit by its own colours: a blurred, saturated copy sits behind it as an ambient glow,
 * brightening while its home tile (or the sidebar photo) is held.
 * `imageClassName` shapes the image itself (radius, shadow); the glow follows its box.
 */
export function GlowImage({
	src,
	sizes,
	className,
	imageClassName,
}: {
	src: StaticImageData;
	sizes: string;
	className?: string;
	imageClassName?: string;
}) {
	return (
		<span className={cn('relative block', className)}>
			<span
				aria-hidden
				className="absolute inset-[6%] ambient-glow transition-opacity duration-300 ease-out group-hover/photo:opacity-95 group-data-[held]/tile:opacity-95 motion-reduce:transition-none"
			>
				<Image src={src} alt="" sizes={sizes} className="size-full object-cover" />
			</span>
			<span className={cn('relative block size-full overflow-hidden', imageClassName)}>
				<Image src={src} alt="" sizes={sizes} className="size-full object-cover" />
				<span aria-hidden className="absolute inset-0 rounded-[inherit] shadow-image-edge" />
			</span>
		</span>
	);
}
