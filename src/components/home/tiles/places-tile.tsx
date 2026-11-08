import 'server-only';
import { ArrowUpRightIcon, MapPinIcon } from 'lucide-react';
import Link from 'next/link';
import { Card, CardHeader } from '@/components/ui/card';
import type { Trip } from '@/lib/content';

/** The latest trips as rows. The page renders it only when a trip exists. */
export function PlacesTile({
	trips,
	index,
	className,
}: {
	trips: readonly Pick<Trip, 'id' | 'data'>[];
	index: number;
	className?: string;
}) {
	return (
		<Card index={index} labelledBy="places-title" className={className}>
			<CardHeader id="places-title" icon={<MapPinIcon />} label="Places">
				<Link
					href="/places/"
					className="inline-flex items-center gap-0.5 rounded-sm outline-none transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
				>
					All places
					<ArrowUpRightIcon aria-hidden className="size-3.5" />
				</Link>
			</CardHeader>
			<ul className="mt-3 flex min-h-0 flex-1 flex-col">
				{trips.slice(0, 3).map((t) => (
					<li key={t.id} className="flex min-h-0 flex-1 border-t border-border first:border-t-0">
						<Link
							href="/places/"
							className="-mx-2 flex min-w-0 flex-1 items-center rounded-media px-2 font-display text-xl outline-none transition-colors hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring"
						>
							<span className="truncate">{t.data.title}</span>
						</Link>
					</li>
				))}
			</ul>
		</Card>
	);
}
