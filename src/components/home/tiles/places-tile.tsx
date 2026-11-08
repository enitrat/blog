import 'server-only';
import { ListRow, ListRows } from '@/components/ui/list-row';
import { TextLink } from '@/components/ui/text-link';
import { Tile } from '@/components/ui/tile';
import { getTrips, type Trip } from '@/lib/content';

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
			size="compact"
			label="Places"
			labelAs="h2"
			dot
			action={
				<TextLink variant="action" icon="arrow" href="/places/">
					All places
				</TextLink>
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
