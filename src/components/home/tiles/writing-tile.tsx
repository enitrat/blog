import 'server-only';
import { Tile } from '@/components/ui/tile';
import { bannerFor, getWritingIndex, postUrl } from '@/utils/writing';
import { Deck } from '../deck';
import { HomeLink } from '../home-link';
import { PostCard } from '../post-card';

/**
 * The last three posts as a stack of index cards that is the tile's whole composition: the newest
 * in front and readable at rest, the others peeking out behind; while the tile is in view and
 * untouched the front card is dealt to the back every 5s. Hovering or focusing inside fans them in
 * place and the card under the pointer comes forward; letting go squares the pile up. Where hover
 * is unavailable, or the tile is narrow, the same three cards show as a scrolling row.
 */
export async function WritingTile() {
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
			size={size}
		/>
	);
	return (
		<Tile
			compact
			label="Writing"
			labelAs="h2"
			action={
				<HomeLink variant="arrow" href="/writing/">
					All writing
				</HomeLink>
			}
		>
			<div className="flex h-full flex-col justify-center">
				<Deck
					label="Latest writing"
					spread={0.7}
					angle={5}
					cycle={5}
					max="8.5rem"
					fit="100%"
					className="hidden pointer-fine:sm:flex"
					cards={posts.map((p) => card(p, 'sm'))}
				/>
				<div className="relative -mx-4 pointer-fine:sm:hidden">
					<ul className="flex snap-x snap-mandatory gap-1.5 overflow-x-auto px-4 py-2 [scrollbar-width:none]">
						{posts.map((p) => (
							<li key={p.id} className="flex-[0_0_min(72%,15rem)] snap-start">
								{card(p, 'sm')}
							</li>
						))}
					</ul>
					<div
						aria-hidden="true"
						className="pointer-events-none absolute inset-y-0 right-0 w-8 bg-linear-to-l from-white via-white/50 via-75% to-white/0 dark:from-iron-900 dark:via-iron-900/50 dark:to-iron-900/0"
					/>
				</div>
			</div>
		</Tile>
	);
}
