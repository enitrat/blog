import NextLink from 'next/link';
import { cn } from '@/ds/lib/cn';
import type { Img } from '@/lib/media';
import { Plate } from './plate';

const SIZES = {
	md: 'gap-3 p-3',
	/** The compact card of a touch row. */
	sm: 'gap-2 p-2',
} as const;

export type PostCardSize = keyof typeof SIZES;

const HEADINGS = { 2: 'h2', 3: 'h3', 4: 'h4', 5: 'h5', 6: 'h6' } as const;

/**
 * A small index card for one piece of writing, and one link. The card is the shadowed object
 * (`--plate-shadow`, default `--elevation-object`, so a `Deck` lifts it); the `Plate` inside gives
 * up its own shadow. Image on top, or a typographic card (title set large, mono date) when there
 * is no image; then title and description.
 */
export function PostCard({
	href,
	title,
	description,
	date,
	image,
	headingLevel = 3,
	size = 'md',
}: {
	href: string;
	title: string;
	description: string;
	/** Shown as `en-GB` day, month, year (UTC). */
	date: Date;
	/** Banner; omit for the typographic card. */
	image?: Img;
	headingLevel?: keyof typeof HEADINGS;
	size?: PostCardSize;
}) {
	const Heading = HEADINGS[headingLevel];
	const when = new Intl.DateTimeFormat('en-GB', {
		day: 'numeric',
		month: 'short',
		year: 'numeric',
		timeZone: 'UTC',
	}).format(date);
	const iso = date.toISOString().slice(0, 10);
	return (
		<NextLink
			className={cn(
				'post-card focus-ring flex h-full flex-col overflow-hidden rounded-md border border-border bg-surface text-surface-foreground no-underline',
				SIZES[size],
			)}
			href={href}
			data-size={size}
		>
			<div className="post-card-media">
				<Plate src={image} ratio="3/2" radius="sm" sizes="16rem">
					<span className="post-card-type">
						<strong>{title}</strong>
						<time dateTime={iso}>{when}</time>
					</span>
				</Plate>
			</div>
			<div className="flex min-w-0 flex-1 flex-col gap-1">
				<Heading className="post-card-title">{title}</Heading>
				<p className="post-card-desc">{description}</p>
				{image && (
					<time className="post-card-date" dateTime={iso}>
						{when}
					</time>
				)}
			</div>
		</NextLink>
	);
}
