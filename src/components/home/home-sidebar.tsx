import {
	BookOpenIcon,
	CakeIcon,
	Disc3Icon,
	DumbbellIcon,
	HandIcon,
	SparklesIcon,
} from 'lucide-react';
import type { StaticImageData } from 'next/image';
import Link from 'next/link';
import type { ReactNode } from 'react';
import { Eyebrow } from '@/components/ui/eyebrow';
import { GlowImage } from '@/components/ui/glow-image';
import { image } from '@/lib/media';

/** Born March 2001; the age turns over on 1 March, since the day is not recorded. */
function age(now = new Date()) {
	return now.getFullYear() - 2001 - (now.getMonth() < 2 ? 1 : 0);
}

const FACTS = [
	{ icon: CakeIcon, text: `I’m ${age()}` },
	{ icon: Disc3Icon, text: 'I record DJ sets on SoundCloud' },
	{ icon: DumbbellIcon, text: 'I’m on a quest to bench 100kg' },
	{ icon: BookOpenIcon, text: 'I enjoy classical literature a lot' },
];

const LINK =
	'rounded-inline text-foreground underline decoration-border-strong underline-offset-[3px] transition-colors hover:decoration-primary';

function Label({
	as,
	id,
	icon,
	children,
}: {
	as: 'p' | 'h2';
	id?: string;
	icon: ReactNode;
	children: ReactNode;
}) {
	return (
		<Eyebrow
			as={as}
			id={id}
			className="[&_svg]:stroke-[1.75]"
			icon={
				<span aria-hidden className="grid place-items-center">
					{icon}
				</span>
			}
		>
			{children}
		</Eyebrow>
	);
}

/** Mathieu: `src/assets/portrait.jpg`, his own square framing at 480px, for a 128px frame. Lit by
 *  its own glow, like every photo on the page. */
function Portrait({ src }: { src: StaticImageData }) {
	return (
		<GlowImage
			src={src}
			alt="Mathieu Saugier"
			sizes="8rem"
			eager
			className="size-32 flex-none"
			imageClassName="rounded-row"
		/>
	);
}

/**
 * Home's left column, rendered once. From `lg` it is a pinned column below the header: who Mathieu
 * is, then the fun facts. Its box reaches 2rem into the page gutter and pads back by the same: a
 * scroll box clips at its edge, and the portrait's glow needs that room to fade out. Below `lg` the
 * aside dissolves (`display: contents`) and its two blocks join home's grid on their own: the intro
 * before the bento, the facts after it, so the tiles start on the first screen.
 */
export async function HomeSidebar() {
	const portrait = await image('portrait.jpg');
	return (
		<aside className="contents lg:sticky lg:top-header lg:flex lg:h-[calc(100dvh-var(--spacing-header))] lg:flex-col lg:gap-6 lg:-ml-8 lg:overflow-y-auto lg:border-r lg:border-border lg:pt-8 lg:pr-8 lg:pb-6 lg:pl-8">
			<header className="order-1 flex flex-col gap-3 lg:order-none">
				<Label as="p" icon={<HandIcon />}>
					About
				</Label>
				<div className="flex flex-col items-start gap-4">
					<Portrait src={portrait} />
					<h1 className="type-title">
						Hi! I’m <span className="text-primary">Mathieu</span>.
					</h1>
				</div>
				<div className="mt-1">
					<p className="type-lead text-muted-foreground">An engineer working in DeFi.</p>
					<p className="mt-1 type-lead text-muted-foreground">
						Senior Protocol Engineer at{' '}
						<Link href="/work/zama/" className={LINK}>
							Zama
						</Link>
					</p>
				</div>
			</header>
			<section
				aria-labelledby="facts-title"
				className="order-3 border-t border-border pt-5 lg:order-none"
			>
				<Label as="h2" id="facts-title" icon={<SparklesIcon />}>
					Fun facts
				</Label>
				<ul className="mt-3 grid gap-2">
					{FACTS.map(({ icon: Icon, text }) => (
						<li key={text} className="flex items-start gap-2 type-body text-muted-foreground">
							{/* Centred on the first line (14px icon, 1.6 leading), so a fact that wraps keeps its icon at the top. */}
							<Icon aria-hidden className="mt-[0.3em] size-3.5 flex-none text-primary" />
							{text}
						</li>
					))}
				</ul>
			</section>
		</aside>
	);
}
