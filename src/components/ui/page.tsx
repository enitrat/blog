import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';

/**
 * A page body: one centred column, the same width on every page, with the page rhythm. One width
 * means every title starts at the same place, so moving between pages never shifts it. Content
 * fits the column; long text caps its own measure inside it (see `Prose`). Home alone is wider.
 */
export function Page({
	as: Root = 'div',
	className,
	children,
}: {
	as?: 'div' | 'article';
	className?: string;
	children: ReactNode;
}) {
	return (
		<Root
			className={cn('mx-auto flex w-full max-w-page flex-col gap-rhythm pt-8 pb-16', className)}
		>
			{children}
		</Root>
	);
}

export function PageHeader({
	kicker,
	lead,
	children,
}: {
	kicker?: ReactNode;
	lead?: ReactNode;
	children: ReactNode;
}) {
	return (
		<header className="flex flex-col gap-2">
			{kicker}
			<h1 className="type-headline text-balance text-foreground">{children}</h1>
			{lead && <p className="type-body text-pretty text-muted-foreground">{lead}</p>}
		</header>
	);
}
