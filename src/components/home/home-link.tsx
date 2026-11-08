import { ArrowRightIcon, ArrowUpRightIcon } from 'lucide-react';
import NextLink from 'next/link';
import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';

const VARIANTS = {
	/** Brand text link in running text. */
	inline:
		'text-brand-600 underline decoration-brand-600/30 underline-offset-[22%] hover:text-brand-700 hover:decoration-brand-700 dark:text-brand-400 dark:decoration-brand-400/30 dark:hover:text-brand-300',
	/** Muted; steps darker on hover. */
	quiet: 'text-gray-400 hover:text-gray-600 dark:text-iron-400 dark:hover:text-iron-100',
	/** A tile action with a trailing arrow. */
	arrow:
		'inline-flex items-center gap-1 text-xs font-medium text-gray-500 hover:text-gray-800 dark:text-iron-400 dark:hover:text-white',
	/** Opens in a new tab, with a trailing arrow. */
	external:
		'inline-flex items-center gap-1 text-sm font-medium text-gray-500 hover:text-gray-800 dark:text-iron-400 dark:hover:text-white',
} as const;

/** A text link on the home bench. Internal hrefs use Next's router; `external` opens a new tab. */
export function HomeLink({
	href,
	variant = 'inline',
	className,
	children,
}: {
	href: string;
	variant?: keyof typeof VARIANTS;
	className?: string;
	children: ReactNode;
}) {
	const classes = cn(
		'group/link rounded-sm transition-colors outline-none focus-visible:ring-2 focus-visible:ring-brand-500/60',
		VARIANTS[variant],
		className,
	);
	if (variant === 'external') {
		return (
			<a href={href} target="_blank" rel="noopener noreferrer" className={classes}>
				{children}
				<ArrowUpRightIcon
					aria-hidden
					className="size-3.5 transition-transform motion-safe:group-hover/link:translate-x-px motion-safe:group-hover/link:-translate-y-px"
				/>
			</a>
		);
	}
	return (
		<NextLink href={href} className={classes}>
			{children}
			{variant === 'arrow' && (
				<ArrowRightIcon
					aria-hidden
					className="size-3.5 transition-transform motion-safe:group-hover/link:translate-x-0.5"
				/>
			)}
		</NextLink>
	);
}
