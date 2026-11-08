import { ArrowRightIcon } from 'lucide-react';
import NextLink from 'next/link';
import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';

export type ListRowProps = {
	href?: string;
	/** Left-column text. A range ("August 2023 – October 2025") breaks into start and end. */
	date?: string;
	/** Machine value; renders the date as a `<time>`. */
	datetime?: string;
	title: ReactNode;
	description?: ReactNode;
	headingLevel?: 'h2' | 'h3';
};

/** One hairline-separated row: date, title, description, and an arrow on hover when linked. */
export function ListRow({
	href,
	date,
	datetime,
	title,
	description,
	headingLevel: Heading = 'h3',
}: ListRowProps) {
	const range = date?.split(' – ');
	const Stamp = datetime ? 'time' : 'span';
	const body = (
		<>
			{date && (
				<div className="flex flex-wrap gap-x-2 text-xs font-medium text-faint tabular-nums sm:w-32 sm:flex-none">
					<Stamp dateTime={datetime} className="flex flex-wrap gap-x-1.5 sm:flex-col">
						{range?.length === 2 ? (
							<>
								<span className="whitespace-nowrap">{range[0]}</span>
								<span className="whitespace-nowrap">– {range[1]}</span>
							</>
						) : (
							date
						)}
					</Stamp>
				</div>
			)}
			<div className="min-w-0 flex-1">
				<Heading className="flex flex-wrap items-center gap-1.5 text-sm font-semibold text-balance text-foreground">
					{title}
				</Heading>
				{description && (
					<p className="mt-0.5 text-sm leading-relaxed text-pretty text-muted-foreground">
						{description}
					</p>
				)}
			</div>
			{href && (
				<ArrowRightIcon
					aria-hidden
					className="hidden size-4 flex-none -translate-x-1 self-center text-faint opacity-0 transition group-hover:translate-x-0 group-hover:opacity-100 group-focus-visible:opacity-100 sm:block"
				/>
			)}
		</>
	);
	const row = cn(
		'group flex flex-col gap-1 py-4 sm:flex-row sm:items-baseline sm:gap-4',
		href &&
			'-mx-3 rounded-xl px-3 transition-colors outline-none hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring',
	);
	return (
		<li className="border-t border-border first:border-t-0">
			{href?.startsWith('/') ? (
				<NextLink href={href} className={row}>
					{body}
				</NextLink>
			) : href ? (
				<a href={href} className={row}>
					{body}
				</a>
			) : (
				<div className={row}>{body}</div>
			)}
		</li>
	);
}

export function ListRows({ className, children }: { className?: string; children: ReactNode }) {
	return <ul className={cn('flex flex-col', className)}>{children}</ul>;
}
