import 'server-only';
import { getPosts, type Post } from '@/lib/content';
import { type Img, maybeImage } from '@/lib/media';

/** The one place that knows the route shape of an article. */
export const postUrl = (post: Pick<Post, 'id'>) => `/writing/${post.id}/`;

const byNewest = (a: Post, b: Post) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf();

/** Every index of the site's writing, newest first. */
export function getWritingIndex(): Post[] {
	return [...getPosts()].sort(byNewest);
}

/** Resolves the post's `related` slugs to links, silently dropping any that name nothing. */
export function getRelated(post: Post) {
	const posts = getPosts();
	return post.data.related
		.map((slug) => posts.find((p) => p.id === slug))
		.filter((p): p is Post => p !== undefined)
		.map((p) => ({ href: postUrl(p), title: p.data.title }));
}

/** Adjacent essays, so the article offers a way onward rather than only a way back. */
export function getAdjacent(post: Post) {
	const posts = getWritingIndex();
	const index = posts.findIndex((p) => p.id === post.id);
	const toLink = (entry: Post | undefined) =>
		entry ? { href: postUrl(entry), title: entry.data.title } : undefined;
	return {
		newer: toLink(index > 0 ? posts[index - 1] : undefined),
		older: toLink(index >= 0 ? posts[index + 1] : undefined),
	};
}

/** The post's `banner.*` image from its folder, if it has one. */
export function bannerFor(post: Pick<Post, 'id'>): Img | undefined {
	for (const ext of ['png', 'jpg', 'jpeg', 'webp']) {
		const img = maybeImage(`content/blog/${post.id}/banner.${ext}`);
		if (img) return img;
	}
	return undefined;
}
