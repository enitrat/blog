import { ArrowRightIcon, ArrowUpRightIcon } from 'lucide-react';
import NextLink from 'next/link';
import type { ReactNode } from 'react';

/**
 * A small text link under a page: muted, one step darker on hover. `arrow` trails an arrow;
 * `external` opens a new tab and trails a corner arrow.
 */
export function TextLink({
	href,
	icon,
	children,
}: {
	href: string;
	icon?: 'arrow' | 'external';
	children: ReactNode;
}) {
	const className =
		'group inline-flex items-center gap-1 rounded-xs text-sm font-medium text-gray-500 transition-colors outline-none hover:text-gray-800 focus-visible:ring-2 focus-visible:ring-brand-500/60 dark:text-iron-400 dark:hover:text-iron-100';
	const body = (
		<>
			{children}
			{icon === 'arrow' && (
				<ArrowRightIcon
					aria-hidden
					className="size-3.5 transition-transform group-hover:translate-x-0.5"
				/>
			)}
			{icon === 'external' && (
				<ArrowUpRightIcon
					aria-hidden
					className="size-3.5 transition-transform group-hover:translate-x-px group-hover:-translate-y-px"
				/>
			)}
		</>
	);
	if (icon === 'external') {
		return (
			<a href={href} target="_blank" rel="noopener noreferrer" className={className}>
				{body}
			</a>
		);
	}
	return (
		<NextLink href={href} className={className}>
			{body}
		</NextLink>
	);
}
