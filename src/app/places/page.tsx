import { getImageProps, type StaticImageData } from 'next/image';
import { notFound } from 'next/navigation';
import { DitherImage } from '@/components/places/dither-image';
import { Lightbox } from '@/components/places/lightbox';
import { Prose } from '@/components/reading/prose';
import { Card } from '@/components/ui/card';
import { Page, PageHeader } from '@/components/ui/page';
import { StatusDot } from '@/components/ui/status-dot';
import { getTrips, type Trip } from '@/lib/content';
import { formatMonthYear } from '@/lib/format';
import { renderMarkdown } from '@/lib/markdown';
import { maybeImage } from '@/lib/media';
import { pageMeta } from '@/lib/meta';

export const metadata = pageMeta({
	title: 'Places',
	description: 'Trips and photographs.',
	path: '/places/',
});

/** A photo of a trip, by its path relative to the trip folder (`./cover.jpg`). */
const photoOf = async (trip: Trip, src: string): Promise<StaticImageData> => {
	const key = `content/${trip.dir}/${src.replace(/^\.\//, '')}`;
	const img = await maybeImage(key);
	if (!img) throw new Error(`${trip.dir}: ${src} is not an image in the trip folder.`);
	return img;
};

/** The lightbox shows the photo optimised to 1600px wide, or its own width when it is smaller. */
const LIGHTBOX_WIDTH = 1600;
const lightboxSrc = (img: StaticImageData) =>
	getImageProps({ src: img, alt: '', width: Math.min(img.width, LIGHTBOX_WIDTH) }).props.src;

/** Trips and photographs. The header shows the link and the sitemap lists the page once a trip exists. */
export default async function PlacesPage() {
	if (getTrips().length === 0) notFound();
	const trips = await Promise.all(
		getTrips().map(async (trip) => {
			const { title, country, date, cover, photos } = trip.data;
			return {
				id: trip.id,
				title,
				country,
				date,
				cover: await photoOf(trip, cover),
				set: await Promise.all(
					photos.map(async (p, n) => ({
						image: await photoOf(trip, p.src),
						alt: p.caption ?? `${title}, photo ${n + 1}`,
						caption: p.caption,
					})),
				),
				notes: trip.body.trim() ? await renderMarkdown(trip.body, trip.dir) : undefined,
			};
		}),
	);
	return (
		<Page>
			<PageHeader>Places</PageHeader>
			<div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2 lg:auto-rows-[minmax(11rem,auto)]">
				{trips.map((t) => (
					<Card key={t.id}>
						<article id={t.id} className="flex flex-col gap-4">
							<DitherImage
								image={t.cover}
								alt={`${t.title}, ${t.country}`}
								ratio="4 / 3"
								sizes="(min-width: 40rem) 24rem, 100vw"
								className="rounded-media border border-border"
							/>
							<div className="flex flex-col gap-1.5">
								<p className="flex items-center gap-1.5 type-meta text-faint">
									<StatusDot />
									{formatMonthYear(t.date)}
								</p>
								<h2 className="type-subtitle text-foreground">
									{t.title}
									<span className="text-muted-foreground">, {t.country}</span>
								</h2>
							</div>
							{t.set.length > 0 && (
								<div className="flex">
									<Lightbox title={t.title} photos={t.set} href={lightboxSrc(t.set[0].image)}>
										{t.set.length} photos
									</Lightbox>
								</div>
							)}
							{t.notes && <Prose tree={t.notes} />}
						</article>
					</Card>
				))}
			</div>
		</Page>
	);
}
