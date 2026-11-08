import { existsSync } from 'node:fs';
import { notFound } from 'next/navigation';
import { getPosts } from '@/lib/content';
import { OG_SIZE, ogImage, plateUrl } from '@/lib/og';

export const size = OG_SIZE;
export const contentType = 'image/png';

export function generateStaticParams() {
	return getPosts().map((post) => ({ slug: post.id }));
}

const formatDate = (date: Date) =>
	date.toLocaleDateString('en-GB', {
		year: 'numeric',
		month: 'long',
		day: 'numeric',
		timeZone: 'UTC',
	});

export default async function Image({ params }: { params: Promise<{ slug: string }> }) {
	const { slug } = await params;
	const post = getPosts().find((p) => p.id === slug) ?? notFound();
	const banner = ['png', 'jpg']
		.map((ext) => `src/content/blog/${post.id}/banner.${ext}`)
		.find(existsSync);
	const { title } = post.data;
	return ogImage({
		title,
		titleSize: title.length > 60 ? 56 : title.length > 36 ? 68 : 80,
		kicker: formatDate(post.data.pubDate),
		plate: banner && (await plateUrl(banner)),
		footer: 'Mathieu Saugier',
	});
}
