'use client';

import { Dialog as DialogPrimitive } from '@base-ui/react/dialog';
import Image, { type StaticImageData } from 'next/image';
import { type ReactNode, useEffect, useState } from 'react';
import { Button, buttonVariants } from '@/components/ui/button';
import {
	Carousel,
	type CarouselApi,
	CarouselContent,
	CarouselItem,
	CarouselNext,
	CarouselPrevious,
} from '@/components/ui/carousel';
import { Dialog, DialogClose, DialogOverlay, DialogPortal } from '@/components/ui/dialog';

export type LightboxPhoto = { image: StaticImageData; alt: string; caption?: string };

/**
 * A link to the first photo that opens the whole set in a dialog instead, one photo at a time.
 * Base UI owns focus, Escape and scroll lock; Embla swipes; ArrowLeft and ArrowRight move.
 */
export function Lightbox({
	title,
	photos,
	href,
	children,
}: {
	title: string;
	photos: LightboxPhoto[];
	/** The first photo's file: the link still works without JS. */
	href: string;
	children: ReactNode;
}) {
	const [open, setOpen] = useState(false);
	const [api, setApi] = useState<CarouselApi>();
	const [index, setIndex] = useState(0);

	useEffect(() => {
		if (!api) return;
		const onSelect = () => setIndex(api.selectedScrollSnap());
		onSelect();
		api.on('select', onSelect);
		return () => {
			api.off('select', onSelect);
		};
	}, [api]);

	return (
		<>
			<a
				href={href}
				onClick={(e) => {
					if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
					e.preventDefault();
					setIndex(0);
					setOpen(true);
				}}
				className={buttonVariants({ variant: 'ghost', size: 'sm' })}
			>
				{children}
			</a>
			<Dialog open={open} onOpenChange={setOpen}>
				<DialogPortal>
					<DialogOverlay className="bg-gray-900/80 dark:bg-black/80" />
					<DialogPrimitive.Popup
						aria-label="Photos"
						onClick={(e) => {
							if (e.target === e.currentTarget) setOpen(false);
						}}
						className="fixed inset-0 z-50 mx-auto flex max-w-6xl flex-col justify-center p-4 text-white outline-none transition-opacity duration-150 data-ending-style:opacity-0 data-starting-style:opacity-0"
					>
						<Carousel setApi={setApi} opts={{ loop: true }} className="flex flex-col gap-4">
							<div className="flex items-center justify-between gap-4">
								<p className="text-xs font-medium tabular-nums" aria-live="polite">
									{title} · {index + 1} / {photos.length}
								</p>
								<DialogClose render={<Button variant="glass" size="sm" />}>Close</DialogClose>
							</div>
							<CarouselContent>
								{photos.map((p) => (
									<CarouselItem key={p.image.src}>
										<figure className="flex flex-col items-center gap-2">
											<Image
												src={p.image}
												alt={p.alt}
												sizes="(min-width: 72rem) 72rem, 100vw"
												className="h-auto max-h-[70dvh] w-auto max-w-full rounded-xl object-contain"
											/>
											{p.caption && <figcaption className="text-xs">{p.caption}</figcaption>}
										</figure>
									</CarouselItem>
								))}
							</CarouselContent>
							{photos.length > 1 && (
								<div className="flex items-center justify-between gap-4">
									<CarouselPrevious
										size="icon-sm"
										aria-label="Previous photo"
										className="static my-0"
									/>
									<CarouselNext size="icon-sm" aria-label="Next photo" className="static my-0" />
								</div>
							)}
						</Carousel>
					</DialogPrimitive.Popup>
				</DialogPortal>
			</Dialog>
		</>
	);
}
