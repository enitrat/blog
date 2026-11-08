import 'server-only';
import NextLink from 'next/link';
import { Tile } from '@/components/ui/tile';
import { getWorkIndex, workUrl } from '@/utils/work';
import { HomeLink } from '../home-link';

/** The work collection, compact. */
export function WorkTile() {
	const reports = getWorkIndex();
	if (reports.length === 0) return null;
	return (
		<Tile
			compact
			label="Work"
			labelAs="h2"
			action={
				<HomeLink variant="arrow" href="/work/">
					All work
				</HomeLink>
			}
		>
			<ul className="flex h-full flex-col justify-between">
				{reports.map((r) => (
					<li
						key={r.id}
						className="min-h-0 border-t border-gray-200 first:border-t-0 dark:border-white/10"
					>
						<NextLink
							href={workUrl(r)}
							className="group flex flex-col rounded-md px-2 py-1 transition-colors outline-none hover:bg-gray-100/50 focus-visible:ring-2 focus-visible:ring-brand-500/60 dark:hover:bg-white/5"
						>
							<span className="flex items-baseline justify-between gap-2">
								<h3 className="truncate text-sm font-semibold text-gray-800 dark:text-iron-100">
									{r.data.title}
								</h3>
								<span className="flex-none text-xs font-medium text-gray-400 tabular-nums dark:text-iron-500">
									{r.data.period ?? 'Project'}
								</span>
							</span>
							<span className="truncate text-xs text-gray-500 dark:text-iron-400">
								{r.data.summary}
							</span>
						</NextLink>
					</li>
				))}
			</ul>
		</Tile>
	);
}
