import Image, { type StaticImageData } from 'next/image';
import { cn } from '@/lib/utils';

/**
 * A real cover at its own aspect ratio, one height for every cover in a section. Decorative: the
 * title line beside it names the book.
 */
export function BookCover({
	cover,
	className,
}: {
	cover: StaticImageData;
	/** Sets the height, such as `h-40`. */
	className?: string;
}) {
	return (
		<Image
			src={cover}
			alt=""
			sizes="12rem"
			className={cn(
				'w-auto flex-none rounded-media object-cover bg-muted shadow-card ring-1 ring-border/60',
				className,
			)}
		/>
	);
}
