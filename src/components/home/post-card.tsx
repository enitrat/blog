import type { StaticImageData } from 'next/image';
import NextLink from 'next/link';
import { Plate } from './plate';

const dateFormat = new Intl.DateTimeFormat('en-GB', {
	day: 'numeric',
	month: 'short',
	year: 'numeric',
	timeZone: 'UTC',
});

/**
 * A small index card for one piece of writing, and one link: the banner on top, or a typographic
 * card (title set large, mono date) when there is no image; then title and description.
 */
export function PostCard({
	href,
	title,
	description,
	date,
	image,
}: {
	href: string;
	title: string;
	description: string;
	date: Date;
	/** Banner; omit for the typographic card. */
	image?: StaticImageData;
}) {
	const when = dateFormat.format(date);
	const iso = date.toISOString().slice(0, 10);
	const stamp = 'font-mono text-[0.6875rem] tracking-wide text-gray-400 dark:text-iron-500';
	return (
		<NextLink
			href={href}
			className="flex h-full flex-col gap-2 overflow-hidden rounded-md border border-gray-200 bg-white p-2 text-gray-800 shadow-sm shadow-zinc-800/15 outline-none focus-visible:ring-2 focus-visible:ring-brand-500/60 dark:border-white/10 dark:bg-iron-900 dark:text-iron-100"
		>
			<Plate src={image} ratio="3/2" sizes="16rem" className="[&>div]:rounded-xs">
				<span className="flex h-full flex-col justify-between gap-2 p-3">
					<strong className="line-clamp-3 text-sm font-semibold text-balance">{title}</strong>
					<time dateTime={iso} className={stamp}>
						{when}
					</time>
				</span>
			</Plate>
			<div className="flex min-w-0 flex-1 flex-col gap-1">
				<h3 className="line-clamp-2 text-xs font-semibold">{title}</h3>
				<p className="line-clamp-2 min-h-[2lh] text-xs leading-normal text-pretty text-gray-500 dark:text-iron-400">
					{description}
				</p>
				{image && (
					<time
						dateTime={iso}
						className={`${stamp} mt-auto border-t border-gray-100 pt-2 dark:border-white/10`}
					>
						{when}
					</time>
				)}
			</div>
		</NextLink>
	);
}
