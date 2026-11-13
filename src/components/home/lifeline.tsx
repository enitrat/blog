import type { LifelineEntry } from '@/data/lifeline';
import { cn } from '@/lib/utils';

/** The rail through a row: from the dot down, through it, or into it, so the line runs unbroken. */
function Rail({ from, to }: { from: 'top' | 'dot'; to: 'bottom' | 'dot' }) {
	return (
		<span
			aria-hidden
			className={cn(
				'absolute left-1/2 w-px -translate-x-1/2 bg-border-strong',
				from === 'top' ? 'top-0' : 'top-[0.5lh]',
				to === 'bottom' ? 'bottom-0' : 'h-[0.5lh]',
			)}
		/>
	);
}

/**
 * A life as a time axis: years in a mono column, a hairline rail with quiet dots, the entry in the
 * body role. The rail is to scale: before each entry it runs `--spacing-lifeline-year` per year
 * since the last, so a childhood reads long and the last two years short. Dots and rail are the
 * neutral roles, so the accent stays with the name above.
 */
export function Lifeline({ entries }: { entries: LifelineEntry[] }) {
	const last = entries.length - 1;
	return (
		<ol>
			{entries.map((e, i) => {
				const years = i === 0 ? 0 : e.year - entries[i - 1].year;
				return (
					<li key={e.year} className="grid grid-cols-[2.25rem_0.5rem_1fr] gap-x-3">
						{years > 0 && (
							// A row of its own above the entry, so the rail runs through the gap.
							<span
								aria-hidden
								className="col-span-3 grid grid-cols-subgrid"
								style={{ height: `calc(var(--spacing-lifeline-year) * ${years})` }}
							>
								<span className="relative col-start-2">
									<Rail from="top" to="bottom" />
								</span>
							</span>
						)}
						<time dateTime={String(e.year)} className="pt-[0.2rem] text-end type-meta text-faint">
							{e.year}
						</time>
						<span className="relative type-body">
							<Rail from={i === 0 ? 'dot' : 'top'} to={i === last ? 'dot' : 'bottom'} />
							<span
								aria-hidden
								className="absolute top-[0.5lh] left-1/2 size-1.5 -translate-1/2 rounded-full bg-faint"
							/>
						</span>
						<p className="type-body text-pretty text-muted-foreground">{e.text}</p>
					</li>
				);
			})}
		</ol>
	);
}
