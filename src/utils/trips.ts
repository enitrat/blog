import { type CollectionEntry, getCollection } from 'astro:content';

const files = import.meta.glob('../content/trips/*/index.md');

/** Trips, newest first. Skips `getCollection` while no trip folder exists, since an empty collection logs a warning. */
export async function getTrips(): Promise<CollectionEntry<'trips'>[]> {
	if (Object.keys(files).length === 0) return [];
	return (await getCollection('trips')).sort(
		(a, b) => b.data.date.getTime() - a.data.date.getTime(),
	);
}
