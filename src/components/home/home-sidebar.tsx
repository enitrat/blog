import { HandIcon, HistoryIcon } from 'lucide-react';
import type { StaticImageData } from 'next/image';
import Link from 'next/link';
import type { ReactNode } from 'react';
import { Lifeline } from '@/components/home/lifeline';
import { Eyebrow } from '@/components/ui/eyebrow';
import { GlowImage } from '@/components/ui/glow-image';
import { lifeline } from '@/data/lifeline';
import { image } from '@/lib/media';

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

/** Mathieu: `src/assets/portrait.jpg`, his own square framing at 480px, for an 80px frame beside
 *  the name on a phone and a 128px one above it from `lg`. Lit by its own glow, like every photo. */
function Portrait({ src }: { src: StaticImageData }) {
	return (
		<GlowImage
			src={src}
			alt="Mathieu Saugier"
			sizes="8rem"
			eager
			className="size-20 flex-none lg:size-32"
			imageClassName="rounded-row"
		/>
	);
}

/**
 * Home's left column, rendered once. From `lg` it is a pinned column below the header: who Mathieu
 * is, then the lifeline. Its box reaches 2rem into the page gutter and pads back by the same: a
 * scroll box clips at its edge, and the portrait's glow needs that room to fade out. Below `lg` the
 * aside dissolves (`display: contents`) and both blocks come before the bento, so the page reads who
 * Mathieu is, then what he makes, and Elsewhere still signs off. To keep that short on a phone, the
 * portrait sits beside the name and the lifeline's rail runs at a fixed 2px a year.
 */
export async function HomeSidebar() {
	const portrait = await image('portrait.jpg');
	return (
		<aside className="contents lg:sticky lg:top-header lg:flex lg:h-[calc(100dvh-var(--spacing-header))] lg:flex-col lg:gap-6 lg:-ml-8 lg:overflow-y-auto lg:border-r lg:border-border lg:pt-8 lg:pr-8 lg:pb-6 lg:pl-8">
			<header className="order-1 flex flex-col gap-3 lg:order-none">
				<Label as="p" icon={<HandIcon />}>
					About
				</Label>
				<div className="flex items-center gap-4 lg:flex-col lg:items-start">
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
				aria-labelledby="lifeline-title"
				className="order-2 border-t border-border pt-5 max-lg:[--spacing-lifeline-year:0.125rem] lg:order-none"
			>
				<Label as="h2" id="lifeline-title" icon={<HistoryIcon />}>
					Me in 10s
				</Label>
				<div className="mt-3">
					<Lifeline entries={lifeline} />
				</div>
			</section>
		</aside>
	);
}
