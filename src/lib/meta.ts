import type { Metadata } from 'next';

export const SITE_URL = 'https://msaug.dev';

/** Title, description, canonical URL and the Open Graph and Twitter cards of one page. */
export function pageMeta({
	title,
	description,
	path,
	image = '/opengraph-image',
}: {
	title: string;
	description: string;
	/** Route with its trailing slash, such as `/about/`. */
	path: string;
	/** The share card. A page's own openGraph replaces the root one, images included. */
	image?: string;
}): Metadata {
	return {
		title,
		description,
		alternates: { canonical: path },
		openGraph: { type: 'website', url: path, title, description, images: image },
		twitter: { card: 'summary_large_image', title, description, images: image },
	};
}
