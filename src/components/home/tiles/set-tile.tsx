import 'server-only';
import { SET } from '@/consts';
import { image } from '@/lib/media';
import { SetPlayer } from './set-player';

/**
 * The last set recorded, and the card is the player. It drives the same SoundCloud widget as the
 * room's record player (`src/lib/set-player.ts`), so playing from either place updates both.
 */
export async function SetTile({ className }: { className?: string }) {
	const date = new Date(SET.recorded).toLocaleDateString('en-GB', {
		day: 'numeric',
		month: 'short',
		year: 'numeric',
		timeZone: 'UTC',
	});
	const { src, width, height, blurDataURL } = await image('set/bouncyx-seven.jpg');
	return <SetPlayer art={{ src, width, height, blurDataURL }} date={date} className={className} />;
}
