'use client';

import { motion } from 'motion/react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';
import { spring } from '@/lib/motion';
import { ThemeToggle } from './theme-toggle';

export type NavLink = { label: string; href: string };

export function Header({ links }: { links: NavLink[] }) {
	const path = usePathname().replace(/(.)\/$/, '$1');
	const current = (href: string) => path === href || path.startsWith(`${href}/`);
	const [hovered, setHovered] = useState<string | null>(null);
	return (
		<header className="sticky top-3 z-40 flex flex-wrap items-center justify-between gap-2 pt-2 pb-6 sm:h-16 sm:flex-nowrap sm:gap-4 sm:py-0">
			<Link
				href="/"
				aria-label="Mathieu Saugier, home"
				aria-current={path === '/' ? 'page' : undefined}
				className="-my-3 py-3 text-sm font-semibold whitespace-nowrap text-foreground transition-opacity hover:opacity-70"
			>
				Mathieu Saugier
			</Link>
			<div className="contents sm:flex sm:items-center sm:gap-1.5">
				<nav
					aria-label="Primary navigation"
					onMouseLeave={() => setHovered(null)}
					className="order-last flex w-full rounded-full bg-card/60 p-0.5 shadow-card backdrop-blur-xs backdrop-saturate-150 sm:order-none sm:w-auto"
				>
					{links.map((l) => (
						<Link
							key={l.href}
							href={l.href}
							onMouseEnter={() => setHovered(l.href)}
							aria-current={current(l.href) ? 'page' : undefined}
							className="relative flex-1 rounded-full px-2 py-3 text-center text-sm font-medium text-muted-foreground transition-colors outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring aria-[current=page]:text-foreground sm:flex-none sm:px-3 sm:py-1.5"
						>
							{hovered === l.href && (
								<motion.span
									layoutId="nav-highlight"
									transition={spring.layout}
									className="absolute inset-0 -z-10 rounded-full bg-muted"
								/>
							)}
							{l.label}
						</Link>
					))}
				</nav>
				<ThemeToggle />
			</div>
		</header>
	);
}
