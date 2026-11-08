import { Analytics } from '@vercel/analytics/next';
import { GeistMono } from 'geist/font/mono';
import type { Metadata, Viewport } from 'next';
import { Figtree } from 'next/font/google';
import type { ReactNode } from 'react';
import { Footer } from '@/components/site/footer';
import { Header, type NavLink } from '@/components/site/header';
import { ThemeProvider } from '@/components/theme-provider';
import { getTrips } from '@/lib/content';
import { SITE_URL } from '@/lib/meta';
import { cn } from '@/lib/utils';
import './globals.css';

const figtree = Figtree({ subsets: ['latin'], variable: '--font-figtree', display: 'swap' });

export const metadata: Metadata = {
	metadataBase: new URL(SITE_URL),
	icons: { icon: { url: '/favicon.svg', type: 'image/svg+xml' } },
};

export const viewport: Viewport = {
	themeColor: [
		{ media: '(prefers-color-scheme: light)', color: '#f9fafb' },
		{ media: '(prefers-color-scheme: dark)', color: '#131316' },
	],
};

export default function RootLayout({ children }: { children: ReactNode }) {
	const links: NavLink[] = [
		{ label: 'Writing', href: '/writing' },
		{ label: 'Reading', href: '/reading' },
		{ label: 'Places', href: '/places' },
		{ label: 'About', href: '/about' },
	].filter((l) => l.href !== '/places' || getTrips().length > 0);
	return (
		<html lang="en" className={cn(figtree.variable, GeistMono.variable)} suppressHydrationWarning>
			<body>
				<ThemeProvider>
					<a
						href="#main"
						className="sr-only rounded-full bg-gray-900 px-4 py-2 text-sm text-white focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-50"
					>
						Skip to content
					</a>
					<div className="mx-auto w-full max-w-7xl px-4">
						<Header links={links} />
						<main id="main">{children}</main>
						<Footer />
					</div>
				</ThemeProvider>
				<Analytics />
			</body>
		</html>
	);
}
