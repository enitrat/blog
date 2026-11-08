import { SparklesIcon } from 'lucide-react';
import { now } from '@/data/now';
import { Card, CardHeader } from '../card';

/** A few short items Mathieu writes himself (`src/data/now.ts`). The page renders it only when the list has items. */
export function NowTile({ index, className }: { index: number; className?: string }) {
	return (
		<Card index={index} labelledBy="now-title" className={className}>
			<CardHeader id="now-title" icon={<SparklesIcon />} label="Now" />
			<ul className="mt-3 flex min-h-0 flex-1 flex-col">
				{now.map((item) => (
					<li
						key={item.what}
						className="flex min-h-0 flex-1 items-center justify-between gap-2 border-t border-border first:border-t-0"
					>
						<span className="truncate text-sm">{item.what}</span>
						<time dateTime={item.since} className="flex-none font-mono text-xs text-faint">
							Since {item.since}
						</time>
					</li>
				))}
			</ul>
		</Card>
	);
}
