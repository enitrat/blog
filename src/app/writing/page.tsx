import { Page, PageHeader } from '@/components/ui/page';
import { WritingRow } from '@/components/writing/writing-row';
import { bannerFor, getPosts, postUrl } from '@/lib/content';
import { pageMeta } from '@/lib/meta';

export const metadata = pageMeta({
	title: 'Writing',
	description: 'I write about software, AI, Cairo, and what I learn while building with them.',
	path: '/writing/',
});

const fmt = (d: Date) =>
	d.toLocaleDateString('en-GB', { year: 'numeric', month: 'short', day: '2-digit' });

export default async function WritingPage() {
	const posts = getPosts();
	const banners = await Promise.all(posts.map(bannerFor));
	return (
		<Page width="wide">
			<PageHeader>Writing</PageHeader>
			{posts.length > 0 ? (
				<ul className="flex flex-col">
					{posts.map((post, i) => (
						<WritingRow
							key={post.id}
							href={postUrl(post)}
							date={fmt(post.data.pubDate)}
							datetime={post.data.pubDate.toISOString()}
							title={post.data.title}
							description={post.data.description}
							banner={banners[i]}
						/>
					))}
				</ul>
			) : (
				<p className="text-sm text-muted-foreground">No posts yet.</p>
			)}
		</Page>
	);
}
