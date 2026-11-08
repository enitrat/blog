import { CONTACT_LINKS } from '@/consts';

export function Footer() {
	return (
		<footer className="pt-10 pb-8">
			<nav aria-label="Contact" className="flex flex-wrap gap-x-4 gap-y-1.5 text-sm">
				{CONTACT_LINKS.map((c) => (
					<a
						key={c.href}
						href={c.href}
						rel="me noreferrer"
						target="_blank"
						className="text-gray-400 transition-colors hover:text-gray-600 dark:text-iron-400 dark:hover:text-iron-100"
					>
						{c.label}
					</a>
				))}
			</nav>
		</footer>
	);
}
