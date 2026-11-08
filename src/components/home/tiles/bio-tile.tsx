import { CONTACT_LINKS } from '@/consts';
import { Link, Magnetic, Pill, Tile } from '@/ds';
import { AgePlate } from '../glyphs/age-plate';
import { Barbell } from '../glyphs/barbell';
import { Book } from '../glyphs/book';
import { Turntable } from '../glyphs/turntable';

const FACT = 'grid grid-cols-[--spacing(10)_1fr] items-center gap-4 py-2 text-foreground';

/** Mathieu's sentence, role, and the three ways to reach him. */
export function BioTile({ order }: { order?: number }) {
	return (
		<Tile
			span={{ md: 6, lg: 5 }}
			rows={{ lg: 4 }}
			order={order}
			as="section"
			aria-labelledby="bio-title"
		>
			<div className="flex h-full flex-col justify-between gap-8">
				<div>
					<h1
						id="bio-title"
						className="grid gap-3 text-title font-medium leading-tight tracking-tight text-balance"
					>
						<span>Hi! I'm Mathieu.</span>
						<span>An engineer working in DeFi.</span>
					</h1>
				</div>
				<div className="grid gap-3 text-ui leading-relaxed text-muted-foreground">
					<p>Fun facts:</p>
					<ul className="m-0 grid list-none p-0">
						<li className={FACT} data-glyph-host="">
							<AgePlate size={40} />
							<span>I'm 25</span>
						</li>
						<li className={FACT} data-glyph-host="">
							<Turntable size={40} />
							<span>I record DJ sets on soundcloud</span>
						</li>
						<li className={FACT} data-glyph-host="">
							<Barbell size={40} />
							<span>I'm on a quest to bench 100kg</span>
						</li>
						<li className={FACT} data-glyph-host="">
							<Book size={40} />
							<span>I enjoy classical literature a lot</span>
						</li>
					</ul>
				</div>
				<div className="grid gap-4 text-ui text-muted-foreground">
					<p>
						Senior Protocol Engineer at <Link href="/work/zama/">Zama</Link>
					</p>
					<nav aria-label="Contact" className="flex flex-wrap gap-2">
						{CONTACT_LINKS.map((c) => (
							<Magnetic key={c.href} strength={0.2}>
								<Pill href={c.href} rel="me noreferrer" target="_blank">
									{c.label}
								</Pill>
							</Magnetic>
						))}
					</nav>
					<Link variant="quiet" href="/about/">
						[Read more →]
					</Link>
				</div>
			</div>
		</Tile>
	);
}
