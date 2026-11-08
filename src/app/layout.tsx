import { Analytics } from '@vercel/analytics/next';
import { GeistMono } from 'geist/font/mono';
import { GeistSans } from 'geist/font/sans';
import type { Metadata, Viewport } from 'next';
import type { ReactNode } from 'react';
import { Footer } from '@/components/site/footer';
import { Header, type NavLink } from '@/components/site/header';
import { cn } from '@/ds';
import { getTrips } from '@/lib/content';
import { SITE_URL } from '@/lib/meta';
import './globals.css';

export const metadata: Metadata = {
	metadataBase: new URL(SITE_URL),
	icons: { icon: { url: '/favicon.svg', type: 'image/svg+xml' } },
};

export const viewport: Viewport = { width: 'device-width', initialScale: 1 };

/** The one shell: header, main and footer on the `--background` bench, inside `--page-max`. */
export default function RootLayout({ children }: { children: ReactNode }) {
	const hasTrips = getTrips().length > 0;
	const links: NavLink[] = [
		{ label: 'Writing', href: '/writing' },
		{ label: 'Reading', href: '/reading' },
		{ label: 'Places', href: '/places' },
		{ label: 'About', href: '/about' },
	].filter((l) => l.href !== '/places' || hasTrips);
	return (
		<html lang="en" className={cn(GeistSans.variable, GeistMono.variable)} suppressHydrationWarning>
			<body>
				<a
					className="skip absolute top-3 left-3 z-(--z-skip) -translate-y-[200%] rounded-pill bg-foreground px-4 py-2 text-surface focus-visible:translate-y-0"
					href="#main"
				>
					Skip to content
				</a>
				<div className="shell mx-auto w-[min(100%,var(--page-max))] px-(--page-gutter)">
					<Header links={links} />
					<main id="main">{children}</main>
					<Footer />
				</div>
				<Analytics />
			</body>
		</html>
	);
}
