'use client';

import { AtSignIcon } from 'lucide-react';
import { motion, useReducedMotion } from 'motion/react';
import { type SimpleIcon, siGithub, siSoundcloud, siTelegram, siX } from 'simple-icons';
import { Card, CardHeader } from '@/components/ui/card';
import { CONTACT_LINKS, SET } from '@/consts';
import { spring } from '@/lib/motion';
import { cn } from '@/lib/utils';

type Brand = { icon: SimpleIcon; tile: string };

// Third-party brand colours stay literal. GitHub and X are black: on the dark card they take a
// lifted grey instead.
const LIFTED_BLACK = 'oklch(0.32 0.01 60)';
const BRANDS = {
	GitHub: { icon: siGithub, tile: `light-dark(#181717, ${LIFTED_BLACK})` },
	X: { icon: siX, tile: `light-dark(#000, ${LIFTED_BLACK})` },
	Telegram: { icon: siTelegram, tile: `#${siTelegram.hex}` },
	SoundCloud: { icon: siSoundcloud, tile: `#${siSoundcloud.hex}` },
} satisfies Record<(typeof CONTACT_LINKS)[number]['label'] | 'SoundCloud', Brand>;

const LINKS = [
	...CONTACT_LINKS.map((c) => ({ ...c, ...BRANDS[c.label] })),
	{ label: 'SoundCloud', href: SET.url, ...BRANDS.SoundCloud },
];

/** Where else to find Mathieu: brand icons in squircles, each labelled. */
export function ElsewhereTile({ className }: { className?: string }) {
	const reduce = useReducedMotion();
	return (
		<Card index={5} labelledBy="elsewhere-title" className={className}>
			<CardHeader id="elsewhere-title" icon={<AtSignIcon />} label="Elsewhere" />
			<ul className="mt-3 grid min-h-0 flex-1 grid-cols-2 place-items-center content-center gap-x-1.5 gap-y-4 min-[68.75rem]:flex min-[68.75rem]:items-center min-[68.75rem]:justify-between min-[68.75rem]:gap-2">
				{LINKS.map((l) => (
					<li key={l.label} className="min-[68.75rem]:flex-1">
						<motion.a
							href={l.href}
							target="_blank"
							rel="me noreferrer"
							initial="rest"
							whileHover={reduce ? undefined : 'hover'}
							whileTap={reduce ? undefined : 'press'}
							className="flex flex-col items-center gap-2 rounded-media text-xs text-muted-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
						>
							<motion.span
								aria-hidden
								style={{ '--c': l.tile } as React.CSSProperties}
								variants={{
									rest: { y: 0, rotate: 0, scale: 1, transition: spring.fan },
									hover: { y: -5, rotate: -4, scale: 1.06, transition: spring.fan },
									press: { scale: 0.9, transition: spring.tap },
								}}
								className={cn(
									'grid size-12 place-items-center rounded-[14px] bg-(--c) shadow-chip sm:size-14 sm:rounded-2xl',
								)}
							>
								<svg
									aria-hidden="true"
									viewBox="0 0 24 24"
									className="size-[22px] fill-white sm:size-[26px]"
								>
									<path d={l.icon.path} />
								</svg>
							</motion.span>
							{l.label}
						</motion.a>
					</li>
				))}
			</ul>
		</Card>
	);
}
