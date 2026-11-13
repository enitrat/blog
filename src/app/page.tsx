import {
	ArrowUpRightIcon,
	AtSignIcon,
	BookOpenIcon,
	BriefcaseIcon,
	Disc3Icon,
	PenLineIcon,
	SparklesIcon,
} from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';
import { books } from '@/booksData';
import { BentoTile } from '@/components/home/bento-tile';
import { BrandLinks } from '@/components/home/brand-links';
import { HomeSidebar } from '@/components/home/home-sidebar';
import { LastSet } from '@/components/home/last-set';
import { ReadingFan } from '@/components/home/reading-fan';
import { WorkFolder } from '@/components/home/work-folder';
import { WritingCarousel } from '@/components/home/writing-carousel';
import { CardHeader } from '@/components/ui/card';
import { PROJECT, SET, SITE_DESCRIPTION } from '@/consts';
import { shortTitle } from '@/lib/books';
import { bannerFor, getPosts, getReports, postUrl } from '@/lib/content';
import { formatDate, isoDate } from '@/lib/format';
import { findImage, image, maybeImage } from '@/lib/media';
import { pageMeta } from '@/lib/meta';

export const metadata = {
	...pageMeta({ title: 'Mathieu Saugier', description: SITE_DESCRIPTION, path: '/' }),
	title: { absolute: 'Mathieu Saugier' },
};

const stamp = (d: Date | null) => d?.getTime() ?? 0;

/**
 * Home: the sidebar (who Mathieu is) beside a bento of what he makes. From `lg` the sidebar is pinned
 * and the page scrolls only the bento, when the bento is taller than the window. Nothing follows the
 * grid, so the pinned sidebar's container ends exactly where the page does.
 *
 * The bento lays itself out by its own width, not the window's: the sidebar takes 20rem from `lg`,
 * so a window breakpoint cannot tell how much room the tiles have. Three layouts:
 *
 * One order at every width, the reading order: Writing, Work, Project, then Last set and
 * Reading, then Elsewhere to sign off.
 *
 *   under 36rem   one column; tiles 16rem tall, Elsewhere as tall as its icons
 *   36rem–58rem   two columns:  Writing (2 x 2) / Work | Project / Last set | Reading / Elsewhere (2),
 *                 four 16rem rows and a last one as tall as Elsewhere's icons
 *   58rem up      six columns, one screen, rows clamped to 13–17rem:
 *                   Writing (4 x 2)               | Work (2)
 *                                                 | Project (2)
 *                   Last set (2) | Reading (2)    | Elsewhere (2)
 *
 * 58rem is the narrowest bento whose two-column tiles still set "Bouncyx Seven" beside its cover.
 */
export default async function Home() {
	const posts = await Promise.all(
		getPosts()
			.slice(0, 4)
			.map(async (p) => ({
				title: p.data.title,
				description: p.data.description,
				date: formatDate(p.data.pubDate),
				iso: isoDate(p.data.pubDate),
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
	const shot = await image('projects/morph.png');
	const book = reading[0];

	return (
		<div className="grid gap-8 pt-8 pb-16 lg:grid-cols-[20rem_1fr] lg:pt-0 lg:pb-0">
			<HomeSidebar />
			<div className="@container/bento order-2 lg:order-none">
				<div className="grid grid-cols-1 gap-bento lg:pt-8 lg:pb-6 @[36rem]/bento:auto-rows-[16rem] @[36rem]/bento:grid-rows-[repeat(4,16rem)_auto] @[36rem]/bento:grid-cols-2 @[58rem]/bento:grid-cols-6 @[58rem]/bento:grid-rows-[repeat(3,clamp(13rem,calc((100dvh-var(--spacing-header)-3.5rem-2*var(--spacing-bento))/3),17rem))]">
					{posts.length > 0 && (
						<BentoTile
							index={0}
							lift
							labelledBy="writing-title"
							className="@[36rem]/bento:col-span-2 @[36rem]/bento:row-span-2 @[58rem]/bento:col-span-4"
						>
							<CardHeader id="writing-title" icon={<PenLineIcon />} label="Writing">
								<Link
									href="/writing/"
									className="relative inline-flex items-center gap-0.5 rounded-inline transition-colors hit-area hover:text-foreground"
								>
									All writing
									<ArrowUpRightIcon aria-hidden className="size-3.5" />
								</Link>
							</CardHeader>
							<WritingCarousel posts={posts} />
						</BentoTile>
					)}

					<BentoTile index={1} labelledBy="work-title" className="@[58rem]/bento:col-span-2">
						<CardHeader id="work-title" icon={<BriefcaseIcon />} label="Work" />
						<WorkFolder sheets={sheets} />
					</BentoTile>

					<BentoTile
						index={2}
						lift
						labelledBy="project-title"
						className="@[58rem]/bento:col-span-2"
					>
						<CardHeader id="project-title" icon={<SparklesIcon />} label="Project">
							<span className="type-meta">{new URL(PROJECT.url).host}</span>
						</CardHeader>
						<span className="relative mt-3 block min-h-0 flex-1 overflow-hidden rounded-media bg-scrim fade-block-end">
							{/* Taller than its window, so holding the tile scrolls the page up a little. */}
							<Image
								src={shot}
								alt=""
								sizes="(min-width: 64rem) 30vw, 100vw"
								className="h-auto w-full transition-transform duration-500 ease-[cubic-bezier(0.23,1,0.32,1)] group-data-[held]/tile:-translate-y-[8%]"
							/>
						</span>
						<h3 className="mt-2 type-subtitle transition-colors duration-150 ease-out group-data-[held]/tile:text-primary">
							<a
								href={PROJECT.url}
								target="_blank"
								rel="noreferrer"
								className="rounded-inline link-cover"
							>
								{PROJECT.title}
								<ArrowUpRightIcon
									aria-hidden
									className="ml-1 inline size-4 align-[-0.125em] transition-transform duration-150 ease-out group-data-[held]/tile:translate-x-0.5 group-data-[held]/tile:-translate-y-0.5"
								/>
							</a>
						</h3>
					</BentoTile>

					<BentoTile index={3} labelledBy="set-title" className="@[58rem]/bento:col-span-2">
						<CardHeader id="set-title" icon={<Disc3Icon />} label="Last set">
							<time dateTime={SET.recorded} className="type-meta">
								{formatDate(new Date(SET.recorded))}
							</time>
						</CardHeader>
						<LastSet art={art} />
					</BentoTile>

					{book && (
						<BentoTile
							index={4}
							lift
							labelledBy="reading-title"
							className="@[58rem]/bento:col-span-2"
						>
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
											<Link href="/reading/" className="rounded-inline link-cover">
												{shortTitle(book.title)}
											</Link>
										</h3>
										<p className="type-caption text-faint">{book.author}</p>
									</div>
								</div>
							</div>
						</BentoTile>
					)}

					<BentoTile
						index={5}
						fit
						labelledBy="elsewhere-title"
						className="@[36rem]/bento:col-span-2"
					>
						<CardHeader id="elsewhere-title" icon={<AtSignIcon />} label="Elsewhere" />
						<BrandLinks />
					</BentoTile>
				</div>
			</div>
		</div>
	);
}
