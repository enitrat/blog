'use client';

import { motion, useReducedMotion } from 'motion/react';
import { type SimpleIcon, siGithub, siSoundcloud, siTelegram, siX } from 'simple-icons';
import { useTileHeld } from '@/components/home/bento-tile';
import { CONTACT_LINKS, SET } from '@/consts';
import { spring } from '@/lib/motion';
import { cn } from '@/lib/utils';

type Brand = { icon: SimpleIcon; tile: string };

// Third-party brand colours stay literal. GitHub and X are black: on the dark card they take
// --lifted-black instead.
const BRANDS = {
	GitHub: { icon: siGithub, tile: `light-dark(#181717, var(--lifted-black))` },
	X: { icon: siX, tile: `light-dark(#000, var(--lifted-black))` },
	Telegram: { icon: siTelegram, tile: `#${siTelegram.hex}` },
	SoundCloud: { icon: siSoundcloud, tile: `#${siSoundcloud.hex}` },
} satisfies Record<(typeof CONTACT_LINKS)[number]['label'] | 'SoundCloud', Brand>;

const LINKS = [
	...CONTACT_LINKS.map((c) => ({ ...c, ...BRANDS[c.label] })),
	{ label: 'SoundCloud', href: SET.url, ...BRANDS.SoundCloud },
];

/**
 * Where else to find Mathieu: brand icons in squircles, each labelled. While the tile is held they
 * rise in a ripple, left to right; the one under the pointer rises and tilts further.
 */
export function BrandLinks() {
	const reduce = useReducedMotion();
	const held = useTileHeld() && !reduce;
	return (
		// Four across only when the tile itself has room for "SoundCloud" under each icon; otherwise two
		// by two. Measured on the tile, not the window: the bento's width depends on the sidebar.
		<div className="@container mt-3 flex min-h-0 flex-1">
			<ul className="grid w-full grid-cols-2 place-items-center content-center gap-x-1.5 gap-y-2 @[18rem]:flex @[18rem]:items-center @[18rem]:justify-between @[18rem]:gap-2">
				{LINKS.map((l, i) => (
					<li key={l.label} className="@[18rem]:flex-1">
						<motion.a
							href={l.href}
							target="_blank"
							rel="me noreferrer"
							initial="rest"
							animate={held ? 'awake' : 'rest'}
							whileHover={reduce ? undefined : 'hover'}
							whileTap={reduce ? undefined : 'press'}
							className="flex flex-col items-center gap-2 rounded-media type-small text-muted-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
						>
							<motion.span
								aria-hidden
								style={{ '--c': l.tile } as React.CSSProperties}
								variants={{
									rest: { y: 0, rotate: 0, scale: 1, transition: spring.fan },
									awake: {
										y: -3,
										rotate: 0,
										scale: 1,
										transition: { ...spring.fan, delay: i * 0.035 },
									},
									hover: { y: -5, rotate: -4, scale: 1.06, transition: spring.fan },
									press: { scale: 0.95, transition: spring.tap },
								}}
								className={cn(
									'grid size-10 place-items-center rounded-[28%] bg-(--c) shadow-chip @[18rem]:size-14',
								)}
							>
								<svg
									aria-hidden="true"
									viewBox="0 0 24 24"
									className="size-5 fill-white @[18rem]:size-6.5"
								>
									<path d={l.icon.path} />
								</svg>
							</motion.span>
							{l.label}
						</motion.a>
					</li>
				))}
			</ul>
		</div>
	);
}
