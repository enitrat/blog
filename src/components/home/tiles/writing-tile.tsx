import 'server-only';
import { Link, Tile } from '@/ds';
import { Deck } from '@/ds/components/objects/deck';
import { PostCard } from '@/ds/components/objects/post-card';
import { bannerFor, getWritingIndex, postUrl } from '@/utils/writing';

/**
 * The last three posts as a stack of index cards that is the tile's whole composition: the newest
 * in front and readable at rest, the others peeking out behind; while the tile is in view and
 * untouched the front card is dealt to the back every 5s. Hovering or focusing inside fans them in
 * place and the card under the pointer comes forward; letting go squares the pile up. Where hover
 * is unavailable, or the tile is narrow, the same three cards show as a scrolling row.
 */
export async function WritingTile({ order }: { order?: number }) {
	const posts = await Promise.all(
		getWritingIndex()
			.slice(0, 3)
			.map(async (p) => ({ ...p, banner: await bannerFor(p) })),
	);
	if (posts.length === 0) return null;
	const card = (p: (typeof posts)[number], size: 'sm' | 'md') => (
		<PostCard
			key={p.id}
			href={postUrl(p)}
			title={p.data.title}
			description={p.data.description}
			date={p.data.pubDate}
			image={p.banner}
			headingLevel={3}
			size={size}
		/>
	);
	return (
		<Tile
			label="Writing"
			labelAs="h2"
			data-deck-host=""
			span={{ md: 6, lg: 4 }}
			order={order}
			as="section"
			action={
				<Link variant="arrow" href="/writing/">
					All writing
				</Link>
			}
		>
			<div className="flex h-full flex-col">
				<Deck
					host="ancestor"
					className="writing-deck"
					label="Latest writing"
					spread={0.26}
					angle={5}
					cycle={5}
					cards={posts.map((p) => card(p, 'md'))}
				/>
				<ul className="writing-row">
					{posts.map((p) => (
						<li key={p.id}>{card(p, 'sm')}</li>
					))}
				</ul>
			</div>
		</Tile>
	);
}
