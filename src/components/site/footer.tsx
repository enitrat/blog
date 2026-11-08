'use client';

import { usePathname } from 'next/navigation';
import { CONTACT_LINKS } from '@/consts';

/** The contact links, on every page but home, whose Elsewhere card already holds them. */
export function Footer() {
	if (usePathname() === '/') return null;
	return (
		<footer className="pt-10 pb-8">
			<nav aria-label="Contact" className="flex flex-wrap gap-x-4 gap-y-1.5 text-sm">
				{CONTACT_LINKS.map((c) => (
					<a
						key={c.href}
						href={c.href}
						rel="me noreferrer"
						target="_blank"
						className="text-faint transition-colors hover:text-foreground"
					>
						{c.label}
					</a>
				))}
			</nav>
		</footer>
	);
}
