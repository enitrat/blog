'use client';

import { motion } from 'motion/react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { spring } from '@/lib/motion';
import type { NavLink } from '@/lib/nav';
import { ThemeToggle } from './theme-toggle';
import { Wordmark } from './wordmark';

/** The wordmark, linking home. */
function SiteName() {
	const home = usePathname() === '/';
	return (
		<Link
			href="/"
			aria-label="msaug.dev, home"
			aria-current={home ? 'page' : undefined}
			className="flex min-h-11 items-center rounded-inline transition-opacity hover:opacity-70"
		>
			<Wordmark />
		</Link>
	);
}

/**
 * The primary nav: plain text links, no container. An underline marks the current page and slides
 * to the next one on navigation, so the current page is marked by more than a text colour.
 */
function SiteNav({ links }: { links: NavLink[] }) {
	const path = usePathname().replace(/(.)\/$/, '$1');
	const current = (href: string) => path === href || path.startsWith(`${href}/`);
	return (
		<nav
			aria-label="Primary navigation"
			className="-mx-4 -my-1 w-[calc(100%+2rem)] overflow-x-auto px-4 py-1 [scrollbar-width:none] sm:my-0 sm:py-0 sm:mx-0 sm:ml-auto sm:w-auto sm:overflow-visible sm:px-0"
		>
			<ul className="flex gap-4">
				{links.map((l) => {
					const here = current(l.href);
					return (
						<li key={l.href}>
							<Link
								href={l.href}
								aria-current={here ? 'page' : undefined}
								className="relative inline-flex min-h-11 items-center rounded-inline type-control whitespace-nowrap text-muted-foreground transition-colors duration-150 hover:text-foreground aria-[current=page]:text-foreground"
							>
								{l.label}
								{here && (
									// 4px under the text: half the 1.25rem line, plus the gap.
									<motion.span
										aria-hidden
										layoutId="nav-underline"
										transition={spring.layout}
										className="absolute inset-x-0 top-[calc(50%+0.875rem)] h-[2.5px] rounded-full bg-primary"
									/>
								)}
							</Link>
						</li>
					);
				})}
			</ul>
		</nav>
	);
}

/**
 * The one header, on every page. It spans the window, with a hairline along its bottom edge that
 * separates it from the page, as the rule beside home's sidebar does; its content sits in the site
 * container. Pinned from `sm` up. On a phone it takes two rows, too much to pin over an essay, so
 * it scrolls away there.
 */
export function Header({ links }: { links: NavLink[] }) {
	return (
		<header className="relative z-40 border-b border-border bg-background/90 backdrop-blur-sm sm:sticky sm:top-0 sm:h-header">
			<div className="mx-auto flex w-full max-w-wide flex-wrap items-center justify-between gap-x-6 px-4 pt-1.5 sm:h-full sm:flex-nowrap sm:px-8 sm:pt-0">
				<SiteName />
				<div className="-mr-1 sm:order-last sm:-ml-2">
					<ThemeToggle />
				</div>
				<SiteNav links={links} />
			</div>
		</header>
	);
}
