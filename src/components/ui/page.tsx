import { cva, type VariantProps } from 'class-variance-authority';
import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';

const pageVariants = cva('mx-auto flex w-full flex-col gap-10 pt-8 pb-16', {
	variants: { width: { wide: 'max-w-7xl', text: 'max-w-xl', list: 'max-w-3xl' } },
	defaultVariants: { width: 'list' },
});

/** A page body: one centred column with the page rhythm. */
export function Page({
	width,
	as: Root = 'div',
	className,
	children,
}: {
	width?: VariantProps<typeof pageVariants>['width'];
	as?: 'div' | 'article';
	className?: string;
	children: ReactNode;
}) {
	return <Root className={cn(pageVariants({ width }), className)}>{children}</Root>;
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
			<h1 className="font-display text-4xl text-balance text-foreground sm:text-5xl">{children}</h1>
			{lead && <p className="text-sm leading-relaxed text-pretty text-muted-foreground">{lead}</p>}
		</header>
	);
}
