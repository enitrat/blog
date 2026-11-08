import 'server-only';
import { SET } from '@/consts';
import { Equalizer, Tile } from '@/ds';
import { Readout } from '@/ds/components/objects/readout';
import { image } from '@/lib/media';
import { SetTilePlayer } from './set-tile-player';

/**
 * The last set recorded, and the tile is the player. It drives the same SoundCloud widget as the
 * room's record player (`src/lib/set-player.ts`), so playing from either place updates both. The
 * widget loads on the first press, not on page load. While playing, the artwork becomes a record
 * (square to round, turning slowly) and the header equalizer moves; under reduced motion it turns
 * round but does not spin. The tile is not a link, so it does not lift.
 */
export function SetTile({
	span = { md: 3, lg: 4 },
	order,
}: {
	span?: { md?: number; lg?: number };
	order?: number;
}) {
	const date = new Date(SET.recorded).toLocaleDateString('en-GB', {
		day: 'numeric',
		month: 'short',
		year: 'numeric',
		timeZone: 'UTC',
	});
	return (
		<Tile
			className="set-tile"
			label="Last set"
			labelAs="h2"
			span={span}
			order={order}
			as="section"
			data-set-tile=""
			action={
				<span className="inline-flex items-center gap-2">
					<span className="elapsed" aria-hidden="true">
						<Readout mode="char" value="0:00" />
					</span>
					<Equalizer />
				</span>
			}
		>
			<SetTilePlayer art={image('set/bouncyx-seven.jpg')} date={date} />
		</Tile>
	);
}
