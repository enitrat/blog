'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { CONTACT_LINKS } from '@/consts';
import { cn } from '@/lib/utils';

/**
 * The contact links, on every page. From `lg` home fills one screen beside its pinned sidebar, and
 * its Elsewhere tile already holds them, so the footer steps aside there.
 */
export function Footer() {
	const home = usePathname() === '/';
	return (
		<footer
			className={cn(
				'flex flex-wrap items-center justify-between gap-x-4 pt-10 pb-8',
				home && 'lg:hidden',
			)}
		>
			<nav aria-label="Contact" className="-mx-2 flex flex-wrap gap-x-1 type-control">
				{CONTACT_LINKS.map((c) => (
					<a
						key={c.href}
						href={c.href}
						rel="me noreferrer"
						target="_blank"
						className="inline-flex min-h-11 min-w-11 items-center justify-center rounded-chip px-2 text-faint transition-colors outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
					>
						{c.label}
					</a>
				))}
			</nav>
			<Link
				href="/design/"
				className="-mx-2 inline-flex min-h-11 items-center rounded-chip px-2 type-control text-faint transition-colors outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
			>
				Design
			</Link>
		</footer>
	);
}
