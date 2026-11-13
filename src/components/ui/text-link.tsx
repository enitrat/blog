import { cva, type VariantProps } from 'class-variance-authority';
import { ArrowRightIcon, ArrowUpRightIcon } from 'lucide-react';
import NextLink from 'next/link';
import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';

const textLinkVariants = cva('group/link transition-colors', {
	variants: {
		variant: {
			/** Muted; one step darker on hover. Under a page or beside a control. */
			muted:
				'inline-flex items-center gap-1 rounded-inline type-control text-muted-foreground hover:text-foreground',
			/** Accent text link in running text. */
			inline:
				'rounded-inline text-primary underline decoration-primary/30 underline-offset-[22%] hover:decoration-primary',
			/** Lighter than muted. */
			quiet: 'rounded-inline text-faint hover:text-foreground',
			/** A tile's header action. */
			action:
				'inline-flex items-center gap-1 rounded-inline type-control-sm text-muted-foreground hover:text-foreground',
		},
	},
	defaultVariants: { variant: 'muted' },
});

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
					className="size-3.5 transition-transform group-hover/link:translate-x-px group-hover/link:-translate-y-px"
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
					className="size-3.5 transition-transform group-hover/link:translate-x-0.5"
				/>
			)}
		</NextLink>
	);
}
