import type { MetadataRoute } from 'next';
import { getPosts, getReports, getTrips, postUrl, workUrl } from '@/lib/content';
import { SITE_URL } from '@/lib/meta';

export const dynamic = 'force-static';

/**
 * Every indexed page, as Astro's sitemap listed them. `/system` is a docs page, and `/places` is
 * left out until the first trip exists.
 */
export default function sitemap(): MetadataRoute.Sitemap {
	const paths = [
		'/',
		'/about/',
		...(getTrips().length > 0 ? ['/places/'] : []),
		'/reading/',
		'/work/',
		...getReports().map(workUrl),
		'/writing/',
		...getPosts().map(postUrl),
	];
	return [...new Set(paths)].sort().map((path) => ({ url: new URL(path, SITE_URL).href }));
}
