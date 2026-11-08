import type { ReactNode } from 'react';
import { cn } from '@/ds/lib/cn';
import { Label } from './label';

const WIDTHS = {
	/** The shell's `--page-max`, left-aligned: index pages. */
	wide: '',
	/** A centred `--measure` column: essay, about, work detail. */
	measure: 'mx-auto max-w-measure',
} as const;

/** The one page layout: vertical rhythm and bottom padding for every page body. */
export function Page({
	width = 'wide',
	as: Root = 'div',
	className,
	children,
}: {
	width?: keyof typeof WIDTHS;
	as?: 'div' | 'article';
	className?: string;
	children: ReactNode;
}) {
	return (
		<Root
			data-slot="page"
			data-width={width}
			className={cn(
				'grid grid-cols-[minmax(0,1fr)] gap-(--gap-page) pt-8 pb-16',
				WIDTHS[width],
				className,
			)}
		>
			{children}
		</Root>
	);
}

/** The one title voice: an h1 with an optional kicker above and a lead under it. */
export function PageTitle({
	kicker,
	lead,
	children,
}: {
	/** Text (set as a `Label`) or a custom node, such as an essay's meta line. */
	kicker?: ReactNode;
	lead?: ReactNode;
	children: ReactNode;
}) {
	return (
		<header data-slot="page-title" className="grid justify-items-start gap-3">
			{typeof kicker === 'string' ? <Label>{kicker}</Label> : kicker}
			<h1 className="type-title">{children}</h1>
			{lead && <p className="type-heading max-w-measure text-muted-foreground">{lead}</p>}
		</header>
	);
}
