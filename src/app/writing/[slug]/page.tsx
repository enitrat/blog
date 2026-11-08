import type { Metadata } from 'next';
import { notFound } from 'next/navigation';
import { BlogPost } from '@/components/reading/blog-post';
import { getAdjacent, getPosts, getRelated, postUrl } from '@/lib/content';
import { inShort } from '@/lib/in-short';
import { renderMarkdown } from '@/lib/markdown';
import { pageMeta } from '@/lib/meta';

type Params = { params: Promise<{ slug: string }> };

export const dynamicParams = false;

export function generateStaticParams() {
	return getPosts().map((post) => ({ slug: post.id }));
}

const postOf = async ({ params }: Params) => {
	const { slug } = await params;
	return getPosts().find((p) => p.id === slug) ?? notFound();
};

export async function generateMetadata(props: Params): Promise<Metadata> {
	const post = await postOf(props);
	return pageMeta({
		title: post.data.title,
		description: post.data.description,
		path: postUrl(post),
	});
}

export default async function PostPage(props: Params) {
	const post = await postOf(props);
	const tree = await renderMarkdown(post.body, post.dir);
	// Sections without a paragraph (such as "Part 1:" headings) stay in the text but not in the skim list.
	const items = inShort(tree, post.data.hooks).filter((item) => item.hook);
	const minutes = Math.max(1, Math.round(post.body.split(/\s+/).length / 230));
	const { newer, older } = getAdjacent(post);
	return (
		<BlogPost
			data={post.data}
			tree={tree}
			inShort={items}
			minutes={minutes}
			newer={newer}
			older={older}
			related={getRelated(post)}
		/>
	);
}
