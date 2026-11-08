import NextLink from 'next/link';
import type { ComponentProps } from 'react';
import { cn } from '@/ds/lib/cn';
import { Icon } from './icon';

const VARIANTS = {
	/** Underlined in running text. */
	inline: 'inline underline-soft hovered:decoration-current',
	/** Muted; underlines on hover. */
	quiet:
		'hit-area-y no-underline text-muted-foreground hovered:text-link hovered:underline hovered:underline-offset-[0.2em]',
	/** A UI action with a trailing arrow. */
	arrow: 'hit-area-y no-underline type-control items-center',
	/** Opens in a new tab, with a trailing glyph. */
	external: 'hit-area-y no-underline type-control items-center',
} as const;

export type LinkVariant = keyof typeof VARIANTS;

export type LinkProps = Omit<ComponentProps<'a'>, 'href'> & {
	href: string;
	variant?: LinkVariant;
};

/** Text link. Every variant but `inline` extends a 44px hit area. Internal hrefs use Next's router. */
export function Link({ href, variant = 'inline', className, children, ...props }: LinkProps) {
	const external = variant === 'external';
	const classes = cn(
		'group/link relative inline-flex items-baseline gap-[0.35em] text-link transition-colors duration-fast focus-visible:rounded-sm focus-visible:outline-offset-3',
		VARIANTS[variant],
		className,
	);
	const body = (
		<>
			{children}
			{variant === 'arrow' && (
				<Icon
					name="arrow-right"
					size="sm"
					className="transition-transform duration-fast motion-safe:group-hover/link:translate-x-0.5"
				/>
			)}
			{external && (
				<Icon
					name="external"
					size="sm"
					className="transition-transform duration-fast motion-safe:group-hover/link:translate-x-px motion-safe:group-hover/link:-translate-y-px"
				/>
			)}
		</>
	);
	const attrs = {
		'data-slot': 'link',
		'data-variant': variant,
		className: classes,
		...(external && { target: '_blank', rel: 'noopener noreferrer' }),
		...props,
	};
	if (href.startsWith('/') && !external) {
		return (
			<NextLink href={href} {...attrs}>
				{body}
			</NextLink>
		);
	}
	return (
		<a href={href} {...attrs}>
			{body}
		</a>
	);
}
