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
		<header className="sticky top-3 z-40 flex items-center justify-between gap-2 py-6 sm:gap-4">
			<Link
				href="/"
				aria-label="Mathieu Saugier, home"
				aria-current={path === '/' ? 'page' : undefined}
				className="text-sm font-semibold whitespace-nowrap text-gray-800 transition-opacity hover:opacity-70 dark:text-iron-100"
			>
				Mathieu Saugier
			</Link>
			<div className="flex items-center gap-1.5">
				<nav
					aria-label="Primary navigation"
					onMouseLeave={() => setHovered(null)}
					className="flex rounded-full border border-gray-200/50 bg-white/50 p-0.5 backdrop-blur-xs backdrop-saturate-150 dark:border-iron-600/25 dark:bg-iron-950/65"
				>
					{links.map((l) => (
						<Link
							key={l.href}
							href={l.href}
							onMouseEnter={() => setHovered(l.href)}
							aria-current={current(l.href) ? 'page' : undefined}
							className="relative rounded-full px-2 py-1.5 text-sm font-medium text-gray-500 transition-colors outline-none hover:text-gray-800 focus-visible:ring-2 focus-visible:ring-brand-500/60 aria-[current=page]:text-gray-900 sm:px-3 dark:text-iron-400 dark:hover:text-iron-100 dark:aria-[current=page]:text-white"
						>
							{hovered === l.href && (
								<motion.span
									layoutId="nav-highlight"
									transition={spring.layout}
									className="absolute inset-0 -z-10 rounded-full bg-gray-100 dark:bg-iron-800"
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
