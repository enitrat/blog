import { ListRow, ListRows } from '@/components/ui/list-row';
import { Tile } from '@/components/ui/tile';
import { now } from '@/data/now';

/** A few short items Mathieu writes himself (`src/data/now.ts`). Renders nothing while empty. */
export function NowTile() {
	if (now.length === 0) return null;
	return (
		<Tile size="compact" label="Now" labelAs="h2" dot>
			<ListRows>
				{now.map((item) => (
					<ListRow
						key={item.what}
						title={item.what}
						date={`Since ${item.since}`}
						datetime={item.since}
					/>
				))}
			</ListRows>
		</Tile>
	);
}
