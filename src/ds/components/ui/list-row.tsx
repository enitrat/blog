import NextLink from 'next/link';
import type { ReactNode } from 'react';
import { cn } from '@/ds/lib/cn';
import { Glide } from './glide';
import { Icon } from './icon';
import { Tag } from './tag';

export type ListRowProps = {
	href?: string;
	/** Mono text in the left column. A range ("August 2023 – October 2025") breaks into start and end. */
	date?: string;
	/** Machine value; renders the date as a `<time>`. */
	datetime?: string;
	title: ReactNode;
	description?: ReactNode;
	/** Small accent tag beside the title, such as "Latest". */
	tag?: string;
	headingLevel?: 'h2' | 'h3';
	/** Extra text under the date. */
	meta?: ReactNode;
	/** Extra content under the description. No links when `href` is set. */
	children?: ReactNode;
};

/**
 * One hairline-separated row: date, title, description, arrow on hover. A linked row is a
 * `Glide` item, so `ListRows variant="tile"` gives the rows one shared highlight.
 */
export function ListRow({
	href,
	date,
	datetime,
	title,
	description,
	tag,
	headingLevel: Heading = 'h3',
	meta,
	children,
}: ListRowProps) {
	const range = date?.split(' – ');
	const isRange = range?.length === 2;
	const hasWhen = Boolean(date || meta);
	const Stamp = datetime ? 'time' : 'span';
	const inner = cn(
		'group/row grid items-baseline gap-(--gap-bench) py-4 text-foreground no-underline max-sm:grid-cols-1 max-sm:gap-1',
		hasWhen
			? href
				? 'grid-cols-[var(--when-width)_minmax(0,1fr)_--spacing(6)]'
				: 'grid-cols-[var(--when-width)_minmax(0,1fr)]'
			: href
				? 'grid-cols-[minmax(0,1fr)_--spacing(6)]'
				: 'grid-cols-1',
		href && 'focus-visible:outline-offset-[-2px]',
	);
	const body = (
		<>
			{hasWhen && (
				<div className="type-label flex flex-wrap gap-x-2 text-muted-foreground">
					{date && (
						<Stamp
							dateTime={datetime}
							className={cn(isRange && 'grid max-sm:flex max-sm:flex-wrap max-sm:gap-x-2')}
						>
							{isRange ? (
								<>
									<span className="whitespace-nowrap">{range[0]}</span>
									<span className="whitespace-nowrap">– {range[1]}</span>
								</>
							) : (
								date
							)}
						</Stamp>
					)}
					{meta}
				</div>
			)}
			<div>
				<Heading className="flex flex-wrap items-center gap-2 text-ui font-medium leading-tight tracking-snug text-balance">
					{title}
					{tag && <Tag tone="accent">{tag}</Tag>}
				</Heading>
				{description && <p className="type-small mt-1 text-muted-foreground">{description}</p>}
				{children}
			</div>
			{href && (
				<span className="inline-flex -translate-x-2 justify-self-end opacity-0 transition-[opacity,transform] duration-fast group-hover/row:translate-x-0 group-hover/row:opacity-100 group-focus-visible/row:translate-x-0 group-focus-visible/row:opacity-100 max-sm:hidden motion-reduce:translate-x-0 motion-reduce:transition-none">
					<Icon name="arrow-right" />
				</span>
			)}
		</>
	);
	return (
		<article
			data-slot="list-row"
			className="row border-t border-border [--when-width:--spacing(30)]"
		>
			{href?.startsWith('/') ? (
				<NextLink href={href} className={cn('inner', inner)} data-glide-item="">
					{body}
				</NextLink>
			) : href ? (
				<a href={href} className={cn('inner', inner)} data-glide-item="">
					{body}
				</a>
			) : (
				<div className={cn('inner', inner)}>{body}</div>
			)}
		</article>
	);
}

/**
 * The group holding `ListRow`s. `page`: capped at `--list-max` with a closing hairline. `tile`:
 * pulls up under the tile header, and linked rows share one `Glide` highlight.
 */
export function ListRows({
	variant = 'page',
	children,
}: {
	variant?: 'page' | 'tile';
	children: ReactNode;
}) {
	return (
		<div
			data-slot="list-rows"
			data-variant={variant}
			className={cn('rows', variant === 'page' ? 'max-w-list border-b border-border' : '-mt-2')}
		>
			{variant === 'tile' ? <Glide shape="row">{children}</Glide> : children}
		</div>
	);
}
