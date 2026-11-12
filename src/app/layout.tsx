import { Analytics } from '@vercel/analytics/next';
import type { Metadata, Viewport } from 'next';
import { Fragment_Mono, Instrument_Sans, Source_Serif_4 } from 'next/font/google';
import type { ReactNode } from 'react';
import { Footer } from '@/components/site/footer';
import { Header } from '@/components/site/header';
import { ThemeProvider } from '@/components/theme-provider';
import { SITE_URL } from '@/lib/meta';
import { navLinks } from '@/lib/nav';
import { cn } from '@/lib/utils';
import './globals.css';

// Headings and interface. Italic too, so emphasis is drawn rather than slanted by the browser.
const instrumentSans = Instrument_Sans({
	subsets: ['latin', 'latin-ext'],
	style: ['normal', 'italic'],
	variable: '--font-instrument-sans',
	display: 'swap',
});

// Reading text. The optical-size axis tunes the same file for 17.5px essays and larger quotes.
const sourceSerif = Source_Serif_4({
	subsets: ['latin', 'latin-ext'],
	style: ['normal', 'italic'],
	axes: ['opsz'],
	variable: '--font-source-serif',
	display: 'swap',
});

// Meta, figures and code. One weight: the mono roles never set a heavier one.
const fragmentMono = Fragment_Mono({
	subsets: ['latin', 'latin-ext'],
	weight: '400',
	style: ['normal', 'italic'],
	variable: '--font-fragment-mono',
	display: 'swap',
});

export const metadata: Metadata = {
	metadataBase: new URL(SITE_URL),
	title: { default: 'Mathieu Saugier', template: '%s · Mathieu Saugier' },
	icons: { icon: { url: '/favicon.svg', type: 'image/svg+xml' } },
};

// Metadata cannot read CSS variables: these mirror neutral-50 and neutral-950 in globals.css, and
// design-tokens.test.ts checks that they match.
export const viewport: Viewport = {
	themeColor: [
		{ media: '(prefers-color-scheme: light)', color: 'oklch(0.972 0.006 250)' },
		{ media: '(prefers-color-scheme: dark)', color: 'oklch(0.185 0.016 250)' },
	],
};

export default function RootLayout({ children }: { children: ReactNode }) {
	const links = navLinks();
	return (
		<html
			lang="en"
			className={cn(instrumentSans.variable, sourceSerif.variable, fragmentMono.variable)}
			suppressHydrationWarning
		>
			<body>
				<ThemeProvider>
					<a
						href="#main"
						className="sr-only type-control rounded-control bg-foreground px-4 py-2 text-background focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-50"
					>
						Skip to content
					</a>
					{/* The header spans the window; its content and everything below share one container. */}
					<Header links={links} />
					<div className="mx-auto w-full max-w-wide px-4 sm:px-8">
						<main id="main">{children}</main>
						<Footer />
					</div>
				</ThemeProvider>
				<Analytics />
			</body>
		</html>
	);
}
