'use client';

import { type KeyboardEvent, useEffect, useState } from 'react';
import { Button } from '@/ds/components/ui/button';
import { DialogContent, Dialog as DialogRoot } from '@/ds/components/ui/dialog';
import { Icon } from '@/ds/components/ui/icon';

export type LightboxPhoto = { src: string; alt: string; caption?: string };

type PhotoSet = { title: string; photos: LightboxPhoto[]; index: number };

/**
 * Opens any `[data-lightbox]` trigger on the page: the trigger carries `data-title`,
 * `data-photos` (JSON `LightboxPhoto[]`) and optionally `data-start`. Use a real link to the first
 * photo as the trigger, so it still works without JS. Returns the teardown.
 */
function listenForTriggers(open: (set: PhotoSet) => void): () => void {
	const onClick = (e: MouseEvent) => {
		const t = (e.target as Element).closest<HTMLElement>('[data-lightbox]');
		if (!t?.dataset.photos) return;
		e.preventDefault();
		open({
			title: t.dataset.title ?? '',
			photos: JSON.parse(t.dataset.photos),
			index: Number(t.dataset.start ?? 0),
		});
	};
	document.addEventListener('click', onClick);
	return () => document.removeEventListener('click', onClick);
}

/**
 * One dialog per page that shows a photo set, one at a time. Base UI owns focus, Escape, the
 * backdrop dismiss and scroll lock; ArrowLeft and ArrowRight move.
 */
export function Lightbox() {
	const [set, setSet] = useState<PhotoSet | null>(null);
	const [open, setOpen] = useState(false);
	useEffect(
		() =>
			listenForTriggers((next) => {
				setSet(next);
				setOpen(true);
			}),
		[],
	);
	const n = set?.photos.length ?? 0;
	const step = (by: number) =>
		setSet((s) => (s ? { ...s, index: (s.index + by + s.photos.length) % s.photos.length } : s));
	const onKeyDown = (e: KeyboardEvent) => {
		if (e.key === 'ArrowRight') step(1);
		if (e.key === 'ArrowLeft') step(-1);
	};
	const photo = set?.photos[set.index];
	return (
		<DialogRoot open={open} onOpenChange={setOpen}>
			<DialogContent
				size="full"
				showCloseButton={false}
				data-lightbox=""
				aria-label="Photos"
				onKeyDown={onKeyDown}
				onClick={(e) => {
					if (e.target === e.currentTarget) setOpen(false);
				}}
				className="lightbox mx-auto max-w-(--lightbox-max-w) content-center gap-3 p-4"
			>
				<div className="bar flex items-center justify-between gap-3">
					<p className="count type-label" aria-live="polite">
						{set && `${set.title} · ${set.index + 1} / ${n}`}
					</p>
					<Button variant="ghost" size="sm" onClick={() => setOpen(false)}>
						Close
					</Button>
				</div>
				<figure className="grid justify-items-center gap-2">
					{photo && (
						// biome-ignore lint/performance/noImgElement: the photo is already a published file
						<img
							src={photo.src}
							alt={photo.alt}
							className="h-auto max-h-(--lightbox-img-max-h) w-auto max-w-full object-contain"
						/>
					)}
					{photo?.caption && <figcaption className="type-label">{photo.caption}</figcaption>}
				</figure>
				<div className="nav flex items-center justify-between gap-3">
					<Button variant="ghost" size="sm" aria-label="Previous photo" onClick={() => step(-1)}>
						<Icon name="chevron-left" />
					</Button>
					<Button variant="ghost" size="sm" aria-label="Next photo" onClick={() => step(1)}>
						<Icon name="chevron-right" />
					</Button>
				</div>
			</DialogContent>
		</DialogRoot>
	);
}
