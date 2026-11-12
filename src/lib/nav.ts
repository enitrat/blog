import { getTrips } from '@/lib/content';

export type NavLink = { label: string; href: string };

/** The site's one nav list, read by the header and by home's sidebar. Places shows once a trip exists. */
export function navLinks(): NavLink[] {
	return [
		{ label: 'Writing', href: '/writing' },
		{ label: 'Reading', href: '/reading' },
		{ label: 'Work', href: '/work' },
		{ label: 'Places', href: '/places' },
		{ label: 'About', href: '/about' },
	].filter((l) => l.href !== '/places' || getTrips().length > 0);
}
