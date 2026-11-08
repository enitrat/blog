import 'server-only';
import { Tile } from '@/components/ui/tile';
import { SET } from '@/consts';
import { image } from '@/lib/media';
import { SetTilePlayer, SetTileStatus } from './set-tile-player';

/**
 * The last set recorded, and the tile is the player. It drives the same SoundCloud widget as the
 * room's record player (`src/lib/set-player.ts`), so playing from either place updates both. The
 * widget loads on the first press, not on page load. While playing, the artwork becomes a record
 * (square to round, turning slowly) and the header equalizer moves; under reduced motion it turns
 * round but does not spin. The tile is not a link, so it does not lift.
 */
export async function SetTile() {
	const date = new Date(SET.recorded).toLocaleDateString('en-GB', {
		day: 'numeric',
		month: 'short',
		year: 'numeric',
		timeZone: 'UTC',
	});
	const { src, width, height, blurDataURL } = await image('set/bouncyx-seven.jpg');
	return (
		<Tile compact label="Last set" labelAs="h2" action={<SetTileStatus />}>
			<SetTilePlayer art={{ src, width, height, blurDataURL }} date={date} />
		</Tile>
	);
}
