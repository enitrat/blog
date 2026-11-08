import { CONTACT_LINKS } from '@/consts';
import { Link } from '@/ds';

/** Contact links. */
export function Footer() {
	return (
		<footer className="footer pt-10 pb-8">
			<nav aria-label="Contact" className="flex flex-wrap gap-x-5 gap-y-2">
				{CONTACT_LINKS.map((c) => (
					<Link key={c.href} variant="quiet" href={c.href} rel="me noreferrer" target="_blank">
						{c.label}
					</Link>
				))}
			</nav>
		</footer>
	);
}
