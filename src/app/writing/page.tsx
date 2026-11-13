import { Page, PageHeader } from '@/components/ui/page';
import { WritingRow } from '@/components/writing/writing-row';
import { bannerFor, getPosts, postUrl } from '@/lib/content';
import { formatDate } from '@/lib/format';
import { pageMeta } from '@/lib/meta';

export const metadata = pageMeta({
	title: 'Writing',
	description: 'I write about software, AI, Cairo, and what I learn while building with them.',
	path: '/writing/',
});

export default async function WritingPage() {
	const posts = getPosts();
	const banners = await Promise.all(posts.map(bannerFor));
	return (
		<Page>
			<PageHeader>Writing</PageHeader>
			{posts.length > 0 ? (
				<ul className="flex flex-col">
					{posts.map((post, i) => (
						<WritingRow
							key={post.id}
							href={postUrl(post)}
							date={formatDate(post.data.pubDate)}
							datetime={post.data.pubDate.toISOString()}
							title={post.data.title}
							description={post.data.description}
							banner={banners[i]}
						/>
					))}
				</ul>
			) : (
				<p className="type-body text-muted-foreground">No posts yet.</p>
			)}
		</Page>
	);
}
