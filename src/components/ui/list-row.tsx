import { ArrowRightIcon } from 'lucide-react';
import NextLink from 'next/link';
import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';
import { Badge } from './badge';

export type ListRowProps = {
	href?: string;
	/** Left-column text. A range ("August 2023 – October 2025") breaks into start and end. */
	date?: string;
	/** Machine value; renders the date as a `<time>`. */
	datetime?: string;
	title: ReactNode;
	description?: ReactNode;
	badge?: string;
	headingLevel?: 'h2' | 'h3';
	meta?: ReactNode;
	children?: ReactNode;
};

/** One hairline-separated row: date, title, description, and an arrow on hover when linked. */
export function ListRow({
	href,
	date,
	datetime,
	title,
	description,
	badge,
	headingLevel: Heading = 'h3',
	meta,
	children,
}: ListRowProps) {
	const range = date?.split(' – ');
	const Stamp = datetime ? 'time' : 'span';
	const body = (
		<>
			{(date || meta) && (
				<div className="flex flex-wrap gap-x-2 text-xs font-medium text-gray-400 tabular-nums sm:w-32 sm:flex-none dark:text-iron-500">
					{date && (
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
					)}
					{meta}
				</div>
			)}
			<div className="min-w-0 flex-1">
				<Heading className="flex flex-wrap items-center gap-1.5 text-sm font-semibold text-balance text-gray-800 dark:text-iron-100">
					{title}
					{badge && <Badge>{badge}</Badge>}
				</Heading>
				{description && (
					<p className="mt-0.5 text-sm leading-relaxed text-pretty text-gray-500 dark:text-iron-400">
						{description}
					</p>
				)}
				{children}
			</div>
			{href && (
				<ArrowRightIcon
					aria-hidden
					className="hidden size-4 flex-none -translate-x-1 self-center text-gray-400 opacity-0 transition group-hover:translate-x-0 group-hover:opacity-100 group-focus-visible:opacity-100 sm:block dark:text-iron-400"
				/>
			)}
		</>
	);
	const row = cn(
		'group flex flex-col gap-1 py-4 sm:flex-row sm:items-baseline sm:gap-4',
		href &&
			'-mx-3 rounded-xl px-3 transition-colors outline-none hover:bg-gray-100/50 focus-visible:ring-2 focus-visible:ring-brand-500/60 dark:hover:bg-white/5',
	);
	return (
		<li className="border-t border-gray-200 first:border-t-0 dark:border-white/10">
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
