import { ListRow, ListRows } from '@/components/ui/list-row';
import { Page, PageHeader } from '@/components/ui/page';
import { getPosts, postUrl } from '@/lib/content';
import { pageMeta } from '@/lib/meta';

export const metadata = pageMeta({
	title: 'Writing',
	description: 'I write about software, AI, Cairo, and what I learn while building with them.',
	path: '/writing/',
});

const fmt = (d: Date) =>
	d.toLocaleDateString('en-GB', { year: 'numeric', month: 'short', day: '2-digit' });

export default function WritingPage() {
	const posts = getPosts();
	return (
		<Page>
			<PageHeader>Writing</PageHeader>
			{posts.length > 0 ? (
				<ListRows>
					{posts.map((post) => (
						<ListRow
							key={post.id}
							headingLevel="h2"
							href={postUrl(post)}
							date={fmt(post.data.pubDate)}
							datetime={post.data.pubDate.toISOString()}
							title={post.data.title}
							description={post.data.description}
						/>
					))}
				</ListRows>
			) : (
				<p className="text-sm text-muted-foreground">No posts yet.</p>
			)}
		</Page>
	);
}
