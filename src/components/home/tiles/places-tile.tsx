import 'server-only';
import { ListRow, ListRows } from '@/components/ui/list-row';
import { Tile } from '@/components/ui/tile';
import { getTrips, type Trip } from '@/lib/content';
import { HomeLink } from '../home-link';

/** The latest trips. Renders nothing until the first trip exists. */
export function PlacesTile({
	trips = getTrips(),
}: {
	/** Defaults to every trip, newest first. */
	trips?: readonly Pick<Trip, 'id' | 'data'>[];
}) {
	if (trips.length === 0) return null;
	return (
		<Tile
			compact
			label="Places"
			labelAs="h2"
			dot
			action={
				<HomeLink variant="arrow" href="/places/">
					All places
				</HomeLink>
			}
		>
			<ListRows>
				{trips.slice(0, 3).map((t) => (
					<ListRow key={t.id} href="/places/" title={t.data.title ?? t.id} />
				))}
			</ListRows>
		</Tile>
	);
}
