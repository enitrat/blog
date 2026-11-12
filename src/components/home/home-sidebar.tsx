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
import { cn } from '@/lib/utils';

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

const FOCUS = 'outline-none focus-visible:ring-2 focus-visible:ring-ring';
const LINK = `rounded-inline text-foreground underline decoration-border-strong underline-offset-[3px] transition-colors hover:decoration-primary ${FOCUS}`;

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

/**
 * A stand-in portrait until Mathieu supplies a photo: his initials set in the sans on the accent,
 * so it reads as a mark, never as someone's face.
 */
function Monogram() {
	return (
		<span
			aria-hidden
			className="relative grid size-20 flex-none place-items-center rounded-row bg-primary type-title text-primary-foreground shadow-glow"
		>
			MS
		</span>
	);
}

/** Who Mathieu is: the page's kicker, title and lead. Its label is a kicker, so the h1 comes first. */
function Intro() {
	return (
		<header className="flex flex-col gap-3">
			<Label as="p" icon={<HandIcon />}>
				About
			</Label>
			<div className="flex items-center gap-4">
				<Monogram />
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
	);
}

function Facts({ id }: { id: string }) {
	return (
		<section aria-labelledby={id} className="border-t border-border pt-5">
			<Label as="h2" id={id} icon={<SparklesIcon />}>
				Fun facts
			</Label>
			<ul className="mt-3 grid gap-2">
				{FACTS.map(({ icon: Icon, text }) => (
					<li key={text} className="flex items-start gap-2.5 type-lead">
						{/* On the first line, so a fact that wraps keeps its icon at the top. */}
						<Icon aria-hidden className="mt-[0.4em] size-4 flex-none text-primary" />
						{text}
					</li>
				))}
			</ul>
		</section>
	);
}

function Room({ src, sizes }: { src: StaticImageData; sizes: string }) {
	return (
		<figure className="group/photo">
			<GlowImage
				src={src}
				sizes={sizes}
				className="aspect-4/3 w-full"
				imageClassName="rounded-row"
			/>
			<figcaption className="mt-2 type-caption text-faint">The room, built in Blender</figcaption>
		</figure>
	);
}

/**
 * Home's left column, under the site header. From `lg` it is pinned below the header, full height:
 * who Mathieu is, the fun facts and the room he built in Blender. Below `lg` it holds only the
 * intro, so the tiles start on the first screen; `HomeFactsBelow` carries the rest after them.
 */
export async function HomeSidebar() {
	const room = await image('living-room.png');
	return (
		<aside className="flex flex-col gap-6 lg:sticky lg:top-header lg:h-[calc(100dvh-var(--spacing-header))] lg:overflow-y-auto lg:border-r lg:border-border lg:pt-8 lg:pr-8 lg:pb-6">
			<Intro />
			<div className="hidden flex-col gap-6 lg:flex">
				<Facts id="facts-title" />
				<Room src={room} sizes="18rem" />
			</div>
		</aside>
	);
}

/** Below `lg`: the fun facts and the room, after the tiles. */
export async function HomeFactsBelow({ className }: { className?: string }) {
	const room = await image('living-room.png');
	return (
		<div className={cn('grid items-start gap-6 sm:grid-cols-2', className)}>
			<Facts id="facts-below-title" />
			<Room src={room} sizes="(min-width: 40rem) 50vw, 100vw" />
		</div>
	);
}
