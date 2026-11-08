import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';

const WIDTHS = { wide: 'max-w-7xl', text: 'max-w-xl', list: 'max-w-3xl' } as const;

/** A page body: one centred column with the page rhythm. */
export function Page({
	width = 'list',
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
			className={cn('mx-auto flex w-full flex-col gap-10 pt-8 pb-16', WIDTHS[width], className)}
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
			<h1 className="text-xl font-semibold text-balance text-gray-900 dark:text-white">
				{children}
			</h1>
			{lead && (
				<p className="text-sm leading-relaxed text-pretty text-gray-500 dark:text-iron-300">
					{lead}
				</p>
			)}
		</header>
	);
}
