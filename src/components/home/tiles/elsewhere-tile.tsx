'use client';

import { AtSignIcon } from 'lucide-react';
import { motion, useReducedMotion } from 'motion/react';
import { type SimpleIcon, siGithub, siSoundcloud, siTelegram, siX } from 'simple-icons';
import { CONTACT_LINKS, SET } from '@/consts';
import { spring } from '@/lib/motion';
import { cn } from '@/lib/utils';
import { Card, CardHeader } from '../card';

// GitHub and X are black: on the dark card they take a lifted grey instead.
const ICONS: Record<(typeof CONTACT_LINKS)[number]['label'], { icon: SimpleIcon; tile: string }> = {
	GitHub: { icon: siGithub, tile: 'light-dark(#181717, oklch(0.32 0.01 60))' },
	X: { icon: siX, tile: 'light-dark(#000, oklch(0.32 0.01 60))' },
	Telegram: { icon: siTelegram, tile: `#${siTelegram.hex}` },
};
const LINKS = [
	...CONTACT_LINKS.map((c) => ({ ...c, ...ICONS[c.label] })),
	{ label: 'SoundCloud', href: SET.url, icon: siSoundcloud, tile: `#${siSoundcloud.hex}` },
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
									'grid size-12 place-items-center rounded-[14px] bg-(--c) sm:size-14 sm:rounded-2xl',
									'shadow-[inset_0_1px_0_oklch(1_0_0/0.25),inset_0_-2px_0_oklch(0_0_0/0.15),0_6px_14px_-6px_var(--c)]',
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
