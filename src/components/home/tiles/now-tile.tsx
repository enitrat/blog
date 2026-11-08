import { now } from '@/data/now';
import { ListRow, ListRows, Tile } from '@/ds';

/** A few short items Mathieu writes himself (`src/data/now.ts`). Renders nothing while empty. */
export function NowTile({ span, order }: { span?: { md?: number; lg?: number }; order?: number }) {
	if (now.length === 0) return null;
	return (
		<Tile label="Now" labelAs="h2" dot span={span} order={order} as="section">
			<ListRows variant="tile">
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
