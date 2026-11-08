import 'server-only';
import { Link, ListRow, ListRows, Tile } from '@/ds';
import { getWorkIndex, workUrl } from '@/utils/work';

/** The work collection, compact. */
export function WorkTile({ order }: { order?: number }) {
	const reports = getWorkIndex();
	if (reports.length === 0) return null;
	return (
		<Tile
			label="Work"
			labelAs="h2"
			span={{ md: 6, lg: 12 }}
			order={order}
			as="section"
			action={
				<Link variant="arrow" href="/work/">
					All work
				</Link>
			}
		>
			<ListRows variant="tile">
				{reports.map((r) => (
					<ListRow
						key={r.id}
						href={workUrl(r)}
						title={r.data.title}
						description={r.data.summary}
						date={r.data.period ?? 'Project'}
					/>
				))}
			</ListRows>
		</Tile>
	);
}
