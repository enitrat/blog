import { cva, type VariantProps } from 'class-variance-authority';
import { ArrowRightIcon, ArrowUpRightIcon } from 'lucide-react';
import NextLink from 'next/link';
import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';

const textLinkVariants = cva(
	'group/link transition-colors outline-none focus-visible:ring-2 focus-visible:ring-brand-500/60',
	{
		variants: {
			variant: {
				/** Muted; one step darker on hover. Under a page or beside a control. */
				muted:
					'inline-flex items-center gap-1 rounded-xs text-sm font-medium text-gray-500 hover:text-gray-800 dark:text-iron-400 dark:hover:text-iron-100',
				/** Brand text link in running text. */
				inline:
					'rounded-sm text-brand-600 underline decoration-brand-600/30 underline-offset-[22%] hover:text-brand-700 hover:decoration-brand-700 dark:text-brand-400 dark:decoration-brand-400/30 dark:hover:text-brand-300',
				/** Lighter than muted. */
				quiet:
					'rounded-sm text-gray-400 hover:text-gray-600 dark:text-iron-400 dark:hover:text-iron-100',
				/** A tile's header action. */
				action:
					'inline-flex items-center gap-1 rounded-sm text-xs font-medium text-gray-500 hover:text-gray-800 dark:text-iron-400 dark:hover:text-white',
			},
		},
		defaultVariants: { variant: 'muted' },
	},
);

/**
 * A text link. `arrow` trails an arrow; `external` opens a new tab and trails a corner arrow.
 * Internal links go through Next's router.
 */
export function TextLink({
	href,
	variant,
	icon,
	className,
	children,
}: {
	href: string;
	icon?: 'arrow' | 'external';
	className?: string;
	children: ReactNode;
} & VariantProps<typeof textLinkVariants>) {
	const classes = cn(textLinkVariants({ variant }), className);
	if (icon === 'external') {
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
			{icon === 'arrow' && (
				<ArrowRightIcon
					aria-hidden
					className="size-3.5 transition-transform motion-safe:group-hover/link:translate-x-0.5"
				/>
			)}
		</NextLink>
	);
}
