import { cn } from '@/lib/utils';

/** Newsreader 600 "M", centred in a 100 by 100 tile. `public/favicon.svg` draws the same path. */
const M =
	'M75.4 29.8L78.8 71.7L84.6 73.3L84.6 74.4L61.6 74.4L61.6 73.3L67.5 71.9L64.4 31.3L64.9 31.4L47.9 74.8L42.8 74.8L25.3 32.0L25.9 31.9L23.4 72.0L29.4 73.3L29.4 74.4L15.4 74.4L15.4 73.3L20.9 72.0L23.5 30.8L17.5 28.3L17.5 27.2L35.3 27.2L50.4 64.6L49.4 64.6L63.9 27.2L81.7 27.2L81.7 28.3L75.4 29.8Z';

/** The brand mark: a serif M on a tile in the primary role, so it follows the theme. */
export function Mark({ className }: { className?: string }) {
	return (
		<svg viewBox="0 0 100 100" aria-hidden className={cn('size-7 flex-none', className)}>
			<rect width="100" height="100" rx="24" className="fill-primary" />
			<path d={M} className="fill-primary-foreground" />
		</svg>
	);
}
