import 'server-only';
import { Link, ListRow, ListRows, Tile } from '@/ds';
import { getTrips, type Trip } from '@/lib/content';

/** The latest trips. Renders nothing until the first trip exists. */
export function PlacesTile({
	trips = getTrips(),
	span,
	order,
}: {
	/** Defaults to every trip, newest first. */
	trips?: readonly Pick<Trip, 'id' | 'data'>[];
	span?: { md?: number; lg?: number };
	order?: number;
}) {
	if (trips.length === 0) return null;
	return (
		<Tile
			label="Places"
			labelAs="h2"
			dot
			dotTone="muted"
			span={span}
			order={order}
			as="section"
			action={
				<Link variant="arrow" href="/places/">
					All places
				</Link>
			}
		>
			<ListRows variant="tile">
				{trips.slice(0, 3).map((t) => (
					<ListRow key={t.id} href="/places/" title={t.data.title ?? t.id} />
				))}
			</ListRows>
		</Tile>
	);
}
