'use client';

import { usePathname } from 'next/navigation';
import { Glide, Link } from '@/ds';

export type NavLink = { label: string; href: string };

/** Site name and primary navigation. The current section carries `aria-current`. */
export function Header({ links }: { links: NavLink[] }) {
	const path = usePathname().replace(/(.)\/$/, '$1');
	const current = (href: string) => path === href || path.startsWith(`${href}/`);
	return (
		<header className="header flex flex-wrap items-center justify-between gap-x-6 gap-y-2 py-6">
			<Link
				className="home font-semibold tracking-snug"
				href="/"
				aria-label="Mathieu Saugier, home"
				aria-current={path === '/' ? 'page' : undefined}
			>
				Mathieu Saugier
			</Link>
			<Glide
				as="nav"
				shape="pill"
				aria-label="Primary navigation"
				className="flex flex-wrap gap-x-5 gap-y-2"
			>
				{links.map((l) => (
					<Link
						key={l.href}
						variant="quiet"
						href={l.href}
						data-glide-item=""
						aria-current={current(l.href) ? 'page' : undefined}
						className="aria-[current=page]:font-medium aria-[current=page]:text-link aria-[current=page]:underline-accent"
					>
						{l.label}
					</Link>
				))}
			</Glide>
		</header>
	);
}
