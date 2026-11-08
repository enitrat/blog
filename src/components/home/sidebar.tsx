import { Magnetic } from '@/components/motion/magnetic';
import { buttonVariants } from '@/components/ui/button';
import { CONTACT_LINKS } from '@/consts';
import { AgePlate } from './glyphs/age-plate';
import { Barbell } from './glyphs/barbell';
import { Book } from './glyphs/book';
import { Turntable } from './glyphs/turntable';
import { HomeLink } from './home-link';

const FACT =
	'group/glyph grid grid-cols-[--spacing(10)_1fr] items-center gap-3 py-0.5 text-gray-800 dark:text-iron-100';

/** Mathieu's sentence, role, and the three ways to reach him: a flat panel, not a card. */
export function Sidebar() {
	return (
		<aside
			aria-labelledby="bio-title"
			className="flex h-full flex-col justify-between gap-6 lg:overflow-y-auto lg:pr-6 lg:[scrollbar-width:none]"
		>
			<h1
				id="bio-title"
				className="grid gap-2 text-xl leading-tight font-semibold tracking-tight text-balance"
			>
				<span>Hi! I'm Mathieu.</span>
				<span>An engineer working in DeFi.</span>
			</h1>
			<div className="grid gap-2 text-sm leading-relaxed text-gray-600 dark:text-iron-300">
				<p>Fun facts:</p>
				<ul className="grid">
					<li className={FACT}>
						<AgePlate size={40} />
						<span>I'm 25</span>
					</li>
					<li className={FACT}>
						<Turntable size={40} />
						<span>I record DJ sets on soundcloud</span>
					</li>
					<li className={FACT}>
						<Barbell size={40} />
						<span>I'm on a quest to bench 100kg</span>
					</li>
					<li className={FACT}>
						<Book size={40} />
						<span>I enjoy classical literature a lot</span>
					</li>
				</ul>
			</div>
			<div className="grid gap-3 text-sm text-gray-600 dark:text-iron-300">
				<p>
					Senior Protocol Engineer at <HomeLink href="/work/zama/">Zama</HomeLink>
				</p>
				<nav aria-label="Contact" className="flex flex-wrap gap-1.5">
					{CONTACT_LINKS.map((c) => (
						<Magnetic key={c.href} intensity={0.2}>
							<a
								href={c.href}
								rel="me noreferrer"
								target="_blank"
								className={buttonVariants({ size: 'sm', className: 'font-mono font-normal' })}
							>
								{c.label}
							</a>
						</Magnetic>
					))}
				</nav>
				<HomeLink variant="quiet" href="/about/" className="justify-self-start">
					[Read more →]
				</HomeLink>
			</div>
		</aside>
	);
}
