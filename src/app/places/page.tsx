import { getImageProps } from 'next/image';
import { Prose } from '@/components/reading/prose';
import { Bench, Button, DitherImage, Label, Lightbox, Page, PageTitle, Tile } from '@/ds';
import { getTrips, type Trip } from '@/lib/content';
import { renderMarkdown } from '@/lib/markdown';
import { type Img, maybeImage } from '@/lib/media';
import { pageMeta } from '@/lib/meta';

export const metadata = pageMeta({
	title: 'Places',
	description: 'Trips and photographs.',
	path: '/places/',
});

const monthYear = new Intl.DateTimeFormat('en-GB', { month: 'long', year: 'numeric' });

/** A photo of a trip, by its path relative to the trip folder (`./cover.jpg`). */
const photoOf = async (trip: Trip, src: string): Promise<Img> => {
	const key = `content/${trip.dir}/${src.replace(/^\.\//, '')}`;
	const img = await maybeImage(key);
	if (!img) throw new Error(`${trip.dir}: ${src} is not an image in the trip folder.`);
	return img;
};

/** The lightbox shows the photo optimised to 1600px wide, or its own width when it is smaller. */
const LIGHTBOX_WIDTH = 1600;
const lightboxSrc = (img: Img) =>
	getImageProps({ src: img, alt: '', width: Math.min(img.width, LIGHTBOX_WIDTH) }).props.src;

/** Trips and photographs. The header shows the link and the sitemap lists the page once a trip exists. */
export default async function PlacesPage() {
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
						src: lightboxSrc(await photoOf(trip, p.src)),
						alt: p.caption ?? `${title}, photo ${n + 1}`,
						caption: p.caption,
					})),
				),
				notes: trip.body.trim() ? (await renderMarkdown(trip.body, trip.dir)).tree : undefined,
			};
		}),
	);
	return (
		<Page>
			<PageTitle>Places</PageTitle>
			{trips.length === 0 ? (
				<p className="empty text-muted-foreground">No trips yet.</p>
			) : (
				<Bench>
					{trips.map((t) => (
						<Tile key={t.id} as="article" span={{ md: 3, lg: 6 }} id={t.id}>
							<DitherImage
								image={t.cover}
								alt={`${t.title}, ${t.country}`}
								ratio="4 / 3"
								sizes="(min-width: 64rem) 38rem, 100vw"
							/>
							<div className="meta mt-2 grid gap-2">
								<Label dot>{monthYear.format(t.date)}</Label>
								<h2 className="text-heading font-semibold tracking-snug">
									{t.title}
									<span className="country font-regular text-muted-foreground">, {t.country}</span>
								</h2>
							</div>
							{t.set.length > 0 && (
								<div className="actions flex">
									<Button
										variant="ghost"
										size="sm"
										href={t.set[0].src}
										data-lightbox=""
										data-title={t.title}
										data-photos={JSON.stringify(t.set)}
									>
										{t.set.length} photos
									</Button>
								</div>
							)}
							{t.notes && <Prose tree={t.notes} />}
						</Tile>
					))}
				</Bench>
			)}
			{trips.length > 0 && <Lightbox />}
		</Page>
	);
}
