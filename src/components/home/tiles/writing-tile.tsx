import 'server-only';
import { ArrowUpRightIcon, PenLineIcon } from 'lucide-react';
import Image from 'next/image';
import Link from 'next/link';
import { bannerFor, getPosts, postUrl } from '@/lib/content';
import { Card, CardHeader, Scrims } from '../card';

const dateFormat = new Intl.DateTimeFormat('en-GB', {
	day: 'numeric',
	month: 'short',
	year: 'numeric',
	timeZone: 'UTC',
});

const FOCUS = 'outline-none focus-visible:ring-2 focus-visible:ring-ring';

/**
 * The newest post, its banner full-bleed under a scrim with the title set large, and links to the
 * posts after it. The title link covers the card; the other links sit above it.
 */
export async function WritingTile({ className }: { className?: string }) {
	const [latest, ...older] = getPosts();
	if (!latest) return null;
	const banner = await bannerFor(latest);
	return (
		<Card index={1} lift labelledBy="writing-title" className={className}>
			{banner ? (
				<Image
					src={banner}
					alt=""
					fill
					sizes="(min-width: 68.75rem) 75vw, 100vw"
					className="-z-20 object-cover transition-transform duration-900 ease-[cubic-bezier(.23,1,.32,1)] motion-safe:group-hover/card:scale-[1.035]"
				/>
			) : (
				<span aria-hidden className="absolute inset-0 -z-20 bg-scrim" />
			)}
			<Scrims />
			<CardHeader id="writing-title" icon={<PenLineIcon />} label="Writing" onMedia>
				<Link
					href="/writing/"
					className={`relative z-10 inline-flex items-center gap-0.5 rounded-sm hover:underline ${FOCUS}`}
				>
					All writing
					<ArrowUpRightIcon aria-hidden className="size-3.5" />
				</Link>
			</CardHeader>
			<div className="mt-3 flex min-h-0 flex-1 flex-col justify-end text-scrim-foreground">
				<time
					dateTime={latest.data.pubDate.toISOString().slice(0, 10)}
					className="font-mono text-xs tracking-[-0.01em] text-scrim-foreground/75 tabular-nums"
				>
					{dateFormat.format(latest.data.pubDate)}
				</time>
				<h3 className="mt-2 font-display text-[2.125rem] leading-none tracking-[-0.015em] text-balance sm:text-[2.75rem]">
					<Link
						href={postUrl(latest)}
						className={`rounded-sm after:absolute after:inset-0 after:z-0 ${FOCUS}`}
					>
						{latest.data.title}
					</Link>
				</h3>
				<p className="mt-2 line-clamp-2 max-w-[46ch] text-scrim-foreground/80 max-sm:hidden">
					{latest.data.description}
				</p>
				{older.length > 0 && (
					<ul className="mt-4 flex min-w-0 gap-1.5">
						{older.slice(0, 3).map((p) => (
							<li key={p.id} className="min-w-0">
								<Link
									href={postUrl(p)}
									className={`relative z-10 block h-[26px] truncate rounded-full bg-scrim-foreground/15 px-2.5 text-xs leading-[26px] text-scrim-foreground/90 backdrop-blur-md transition-colors hover:bg-scrim-foreground/25 ${FOCUS}`}
								>
									{p.data.title}
								</Link>
							</li>
						))}
					</ul>
				)}
			</div>
		</Card>
	);
}
