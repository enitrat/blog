import { type CollectionEntry, getCollection } from 'astro:content';

type Post = CollectionEntry<'blog'>;

/** The one place that knows the route shape of an article. */
export const postUrl = (post: Pick<Post, 'id'>) => `/writing/${post.id}/`;

const byNewest = (a: Post, b: Post) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf();

/** Every index of the site's writing, newest first. */
export async function getWritingIndex(): Promise<Post[]> {
	return (await getCollection('blog')).sort(byNewest);
}

/** Resolves the post's `related` slugs to links, silently dropping any that name nothing. */
export async function getRelated(post: Post) {
	const slugs: string[] = post.data.related ?? [];
	if (slugs.length === 0) return [];
	const posts = await getCollection('blog');
	return slugs
		.map((slug) => posts.find((p) => p.id === slug))
		.filter((p): p is Post => p !== undefined)
		.map((p) => ({ href: postUrl(p), title: p.data.title }));
}

/** Adjacent essays, so the article offers a way onward rather than only a way back. */
export async function getAdjacent(post: Post) {
	const posts = await getWritingIndex();
	const index = posts.findIndex((p) => p.id === post.id);
	const toLink = (entry: Post | undefined) =>
		entry ? { href: postUrl(entry), title: entry.data.title } : undefined;
	return {
		newer: toLink(index > 0 ? posts[index - 1] : undefined),
		older: toLink(index >= 0 ? posts[index + 1] : undefined),
	};
}
