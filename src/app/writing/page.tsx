import { ListRow, ListRows, Page, PageTitle } from '@/ds';
import { pageMeta } from '@/lib/meta';
import { getWritingIndex, postUrl } from '@/utils/writing';

export const metadata = pageMeta({
	title: 'Writing',
	description: 'I write about software, AI, Cairo, and what I learn while building with them.',
	path: '/writing/',
});

const fmt = (d: Date) =>
	d.toLocaleDateString('en-GB', { year: 'numeric', month: 'short', day: '2-digit' });

export default function WritingPage() {
	const posts = getWritingIndex();
	return (
		<Page>
			<PageTitle>Writing</PageTitle>
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
			{posts.length === 0 && <p>No posts yet.</p>}
		</Page>
	);
}
