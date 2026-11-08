import { BookOpenIcon, CakeIcon, Disc3Icon, DumbbellIcon, HandIcon } from 'lucide-react';
import Link from 'next/link';
import { Card, CardHeader } from '@/components/ui/card';

const FACTS = [
	{ icon: CakeIcon, text: "I'm 25" },
	{ icon: Disc3Icon, text: 'I record DJ sets on SoundCloud' },
	{ icon: DumbbellIcon, text: "I'm on a quest to bench 100kg" },
	{ icon: BookOpenIcon, text: 'I enjoy classical literature a lot' },
];

const LINK =
	'rounded-sm font-medium text-foreground underline decoration-border-strong underline-offset-[3px] outline-none hover:decoration-primary focus-visible:ring-2 focus-visible:ring-ring';

/** Mathieu's greeting, fun facts and role. */
export function AboutTile({ className }: { className?: string }) {
	return (
		<Card index={0} labelledBy="about-title" className={className}>
			<CardHeader id="about-title" icon={<HandIcon />} label="About" />
			<div className="mt-3 flex min-h-0 flex-1 flex-col">
				<h1 className="font-display text-4xl leading-[1.02] min-[68.75rem]:text-display tracking-[-0.015em]">
					Hi! I'm <em className="text-primary">Mathieu</em>.
					<span className="mt-1 block font-sans text-lg leading-[1.3] tracking-normal text-muted-foreground">
						An engineer working in DeFi.
					</span>
				</h1>
				<ul className="mt-auto grid gap-0.5 pt-4">
					{FACTS.map(({ icon: Icon, text }) => (
						<li
							key={text}
							className="flex h-[30px] items-center gap-2.5 border-t border-border text-sm text-muted-foreground first:border-t-0"
						>
							<Icon aria-hidden className="size-4 flex-none text-primary" />
							{text}
						</li>
					))}
				</ul>
				<p className="mt-3 border-t border-border pt-3 text-sm text-muted-foreground">
					Senior Protocol Engineer at{' '}
					<Link href="/work/zama/" className={LINK}>
						Zama
					</Link>
				</p>
			</div>
		</Card>
	);
}
