import { Analytics } from '@vercel/analytics/next';
import { GeistMono } from 'geist/font/mono';
import { GeistSans } from 'geist/font/sans';
import type { Metadata, Viewport } from 'next';
import { Instrument_Serif } from 'next/font/google';
import type { ReactNode } from 'react';
import { Footer } from '@/components/site/footer';
import { Header, type NavLink } from '@/components/site/header';
import { ThemeProvider } from '@/components/theme-provider';
import { getTrips } from '@/lib/content';
import { SITE_URL } from '@/lib/meta';
import { cn } from '@/lib/utils';
import './globals.css';

const instrumentSerif = Instrument_Serif({
	subsets: ['latin'],
	weight: '400',
	style: ['normal', 'italic'],
	variable: '--font-instrument-serif',
	display: 'swap',
});

export const metadata: Metadata = {
	metadataBase: new URL(SITE_URL),
	icons: { icon: { url: '/favicon.svg', type: 'image/svg+xml' } },
};

// Metadata cannot read CSS variables: these mirror paper-50 and paper-950 in globals.css, and
// design-tokens.test.ts checks that they match.
export const viewport: Viewport = {
	themeColor: [
		{ media: '(prefers-color-scheme: light)', color: 'oklch(0.972 0.006 85)' },
		{ media: '(prefers-color-scheme: dark)', color: 'oklch(0.165 0.01 50)' },
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
		<html
			lang="en"
			className={cn(GeistSans.variable, GeistMono.variable, instrumentSerif.variable)}
			suppressHydrationWarning
		>
			<body>
				<ThemeProvider>
					<a
						href="#main"
						className="sr-only rounded-full bg-foreground px-4 py-2 text-sm text-background focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-50"
					>
						Skip to content
					</a>
					<div className="mx-auto w-full max-w-[90rem] px-4 sm:px-8">
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
