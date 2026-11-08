'use client';

import { usePathname } from 'next/navigation';
import { CONTACT_LINKS } from '@/consts';

/** The contact links, on every page but home, whose Elsewhere card already holds them. */
export function Footer() {
	if (usePathname() === '/') return null;
	return (
		<footer className="pt-10 pb-8">
			<nav aria-label="Contact" className="-mx-2 flex flex-wrap gap-x-1 text-sm">
				{CONTACT_LINKS.map((c) => (
					<a
						key={c.href}
						href={c.href}
						rel="me noreferrer"
						target="_blank"
						className="inline-flex min-h-11 min-w-11 items-center justify-center rounded-md px-2 text-faint transition-colors outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
					>
						{c.label}
					</a>
				))}
			</nav>
		</footer>
	);
}
