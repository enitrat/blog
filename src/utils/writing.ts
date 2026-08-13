import { type CollectionEntry, getCollection } from 'astro:content';

type Post = CollectionEntry<'blog'>;

/** The one place that knows the route shape of an article. */
export const postUrl = (post: Pick<Post, 'id'>) => `/writing/${post.id}/`;

/**
 * Every index of the site's writing: English only, newest first.
 *
 * A translation is reachable from the article itself, never a second item,
 * so feeds and indexes list each piece exactly once.
 */
export async function getWritingIndex(): Promise<Post[]> {
	const posts = await getCollection('blog');
	return posts
		.filter((post) => post.data.lang === 'en')
		.sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf());
}

/**
 * Translations share a directory: `piece/en.md` and `piece/fr.md` are one
 * piece, so the base slug (the first path segment) is the grouping key.
 */
export async function getTranslations(post: Post) {
	const baseSlug = post.id.split('/')[0];
	const posts = await getCollection('blog');
	return posts
		.filter((p) => p.id.split('/')[0] === baseSlug && p.id !== post.id)
		.map((p) => ({ lang: p.data.lang, href: postUrl(p), title: p.data.title }));
}

/**
 * Adjacent essays in this post's own language, so the article offers a way
 * onward rather than only a way back.
 */
export async function getAdjacent(post: Post) {
	const posts = await getCollection('blog');
	const siblings = posts
		.filter((p) => p.data.lang === post.data.lang)
		.sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf());
	const index = siblings.findIndex((p) => p.id === post.id);
	const toLink = (entry: Post | undefined) =>
		entry ? { href: postUrl(entry), title: entry.data.title } : undefined;
	return {
		newer: toLink(index > 0 ? siblings[index - 1] : undefined),
		older: toLink(index >= 0 ? siblings[index + 1] : undefined),
	};
}
