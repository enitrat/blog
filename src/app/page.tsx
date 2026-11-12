import {
	ArrowUpRightIcon,
	AtSignIcon,
	BookOpenIcon,
	BriefcaseIcon,
	Disc3Icon,
	PenLineIcon,
	SparklesIcon,
} from 'lucide-react';
import Link from 'next/link';
import { books } from '@/booksData';
import { BentoTile } from '@/components/home/bento-tile';
import { BrandLinks } from '@/components/home/brand-links';
import { HomeFactsBelow, HomeSidebar } from '@/components/home/home-sidebar';
import { LastSet } from '@/components/home/last-set';
import { ReadingFan } from '@/components/home/reading-fan';
import { WorkFolder } from '@/components/home/work-folder';
import { WritingCarousel } from '@/components/home/writing-carousel';
import { CardHeader } from '@/components/ui/card';
import { PROJECT, SET, SITE_DESCRIPTION } from '@/consts';
import { bannerFor, getPosts, getReports, postUrl } from '@/lib/content';
import { findImage, image, maybeImage } from '@/lib/media';
import { pageMeta } from '@/lib/meta';

export const metadata = {
	...pageMeta({ title: 'Mathieu Saugier', description: SITE_DESCRIPTION, path: '/' }),
	title: { absolute: 'Mathieu Saugier' },
};

const dateFormat = new Intl.DateTimeFormat('en-GB', {
	day: 'numeric',
	month: 'short',
	year: 'numeric',
	timeZone: 'UTC',
});
const stamp = (d: Date | null) => d?.getTime() ?? 0;
const short = (t: string) => t.replace(/\s*\(.*\)\s*$/, '');
const FOCUS = 'outline-none focus-visible:ring-2 focus-visible:ring-ring';

/**
 * Home: under the site header, a pinned sidebar (who Mathieu is) beside a bento of what he makes.
 * Both start where every page starts: the sidebar at the container's left edge, its first line
 * where other pages put their title. From 64rem the bento fills the screen below the header in
 * three rows, each clamped to 13–17rem so a tall monitor does not blow the tiles up and a short
 * window does not crush them:
 *
 *   Writing (4 x 2)            | Last set (2)
 *                              | Reading (2)
 *   Work (2) | Elsewhere (2)   | Project (2)
 *
 * Below 64rem the sidebar holds only the intro, and the fun facts follow the tiles.
 *
 * Motion follows BentoTile's three layers: the surface lifts, one object per tile wakes (banner,
 * record, covers, folder sheets, icons, the live preview), and arrows and titles answer in CSS.
 */
export default async function Home() {
	const posts = await Promise.all(
		getPosts()
			.slice(0, 4)
			.map(async (p) => ({
				title: p.data.title,
				description: p.data.description,
				date: dateFormat.format(p.data.pubDate),
				iso: p.data.pubDate.toISOString().slice(0, 10),
				href: postUrl(p),
				banner: await bannerFor(p),
			})),
	);
	const reading = books
		.filter((b) => b.status === 'reading')
		.sort((a, b) => stamp(b.dateStarted) - stamp(a.dateStarted));
	const covers = (
		await Promise.all(
			reading.slice(0, 2).map(async (b) => ({
				id: b.edition.isbn13,
				cover: await findImage(`covers/${b.edition.isbn13}`),
			})),
		)
	).flatMap((c) => (c.cover ? [{ id: c.id, cover: c.cover }] : []));
	const sheets = await Promise.all(
		getReports().map(async (r) => ({
			title: r.data.title,
			role: r.data.role ?? r.data.summary,
			logo: await maybeImage(`logos/${r.id}.png`),
		})),
	);
	const art = await image('set/bouncyx-seven.jpg');
	const book = reading[0];

	return (
		<div className="grid gap-8 pt-8 pb-16 lg:grid-cols-[20rem_1fr] lg:pt-0 lg:pb-0">
			<HomeSidebar />
			<div className="grid grid-cols-1 auto-rows-[16rem] gap-bento sm:grid-cols-2 lg:grid-cols-6 lg:grid-rows-[repeat(3,clamp(13rem,calc((100dvh-var(--spacing-header)-3.5rem-2*var(--spacing-bento))/3),17rem))] lg:pt-8 lg:pb-6">
				{posts.length > 0 && (
					<BentoTile
						index={0}
						lift
						labelledBy="writing-title"
						className="sm:col-span-2 sm:row-span-2 lg:col-span-4"
					>
						<CardHeader id="writing-title" icon={<PenLineIcon />} label="Writing">
							<Link
								href="/writing/"
								className={`relative inline-flex items-center gap-0.5 rounded-inline transition-colors after:absolute after:-inset-x-1 after:-inset-y-3 hover:text-foreground ${FOCUS}`}
							>
								All writing
								<ArrowUpRightIcon aria-hidden className="size-3.5" />
							</Link>
						</CardHeader>
						<WritingCarousel posts={posts} />
					</BentoTile>
				)}

				<BentoTile index={1} labelledBy="set-title" className="lg:col-span-2">
					<CardHeader id="set-title" icon={<Disc3Icon />} label="Last set">
						<time dateTime={SET.recorded} className="type-meta">
							{dateFormat.format(new Date(SET.recorded))}
						</time>
					</CardHeader>
					<LastSet art={art} />
				</BentoTile>

				{book && (
					<BentoTile index={2} lift labelledBy="reading-title" className="lg:col-span-2">
						<CardHeader id="reading-title" icon={<BookOpenIcon />} label="Reading">
							{reading.length > 1 && (
								<span className="type-meta whitespace-nowrap">+{reading.length - 1} more</span>
							)}
						</CardHeader>
						{/* Covers beside the title when the tile is wide enough for a short title on one line;
						    otherwise the title sits under them, so "East of Eden" never breaks in two. */}
						<div className="@container mt-2 flex min-h-0 flex-1">
							<div className="flex min-h-0 w-full flex-1 flex-col gap-2 @[15rem]:flex-row @[15rem]:gap-3">
								<div className="flex min-h-0 flex-1 flex-col @[15rem]:w-2/5 @[15rem]:flex-none">
									<ReadingFan covers={covers} />
								</div>
								<div className="flex min-w-0 flex-col justify-end @[15rem]:flex-1">
									<h3 className="type-subtitle transition-colors duration-150 ease-out group-data-[held]/tile:text-primary">
										<Link
											href="/reading/"
											className={`rounded-inline after:absolute after:inset-0 ${FOCUS}`}
										>
											{short(book.title)}
										</Link>
									</h3>
									<p className="type-caption text-faint">{book.author}</p>
								</div>
							</div>
						</div>
					</BentoTile>
				)}

				<BentoTile index={3} labelledBy="work-title" className="lg:col-span-2">
					<CardHeader id="work-title" icon={<BriefcaseIcon />} label="Work" />
					<WorkFolder sheets={sheets} />
				</BentoTile>

				<BentoTile index={4} labelledBy="elsewhere-title" className="lg:col-span-2">
					<CardHeader id="elsewhere-title" icon={<AtSignIcon />} label="Elsewhere" />
					<BrandLinks />
				</BentoTile>

				<BentoTile index={5} lift labelledBy="project-title" className="lg:col-span-2">
					<CardHeader id="project-title" icon={<SparklesIcon />} label="Project">
						<span className="type-meta">{new URL(PROJECT.url).host}</span>
					</CardHeader>
					<span className="relative mt-3 block min-h-0 flex-1 overflow-hidden rounded-media bg-muted fade-block-end">
						{/* Twice the window's height, so holding the tile can scroll the live page up a little. */}
						<span className="absolute inset-x-0 top-0 h-[200%] transition-transform duration-500 ease-[cubic-bezier(0.23,1,0.32,1)] group-data-[held]/tile:-translate-y-[12%] motion-reduce:transition-none">
							<iframe
								src={PROJECT.url}
								title={`${PROJECT.title}, live preview`}
								tabIndex={-1}
								aria-hidden
								loading="lazy"
								className="pointer-events-none absolute top-0 left-0 h-[250%] w-[250%] origin-top-left scale-[0.4] border-0"
							/>
						</span>
					</span>
					<h3 className="mt-2 type-subtitle transition-colors duration-150 ease-out group-data-[held]/tile:text-primary">
						<a
							href={PROJECT.url}
							target="_blank"
							rel="noreferrer"
							className={`rounded-inline after:absolute after:inset-0 ${FOCUS}`}
						>
							{PROJECT.title}
							<ArrowUpRightIcon
								aria-hidden
								className="ml-1 inline size-4 align-[-0.125em] transition-transform duration-150 ease-out group-data-[held]/tile:translate-x-0.5 group-data-[held]/tile:-translate-y-0.5 motion-reduce:transition-none"
							/>
						</a>
					</h3>
				</BentoTile>
			</div>
			<HomeFactsBelow className="lg:hidden" />
		</div>
	);
}
