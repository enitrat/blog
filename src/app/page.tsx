import { RoomTile } from '@/components/home/room-tile';
import { BioTile } from '@/components/home/tiles/bio-tile';
import { NowTile } from '@/components/home/tiles/now-tile';
import { PlacesTile } from '@/components/home/tiles/places-tile';
import { ReadingTile } from '@/components/home/tiles/reading-tile';
import { SetTile } from '@/components/home/tiles/set-tile';
import { WorkTile } from '@/components/home/tiles/work-tile';
import { WritingTile } from '@/components/home/tiles/writing-tile';
import { SITE_DESCRIPTION } from '@/consts';
import { now } from '@/data/now';
import { Bench } from '@/ds';
import { getTrips } from '@/lib/content';
import { pageMeta } from '@/lib/meta';

export const metadata = pageMeta({
	title: 'Mathieu Saugier',
	description: SITE_DESCRIPTION,
	path: '/',
});

export default function Home() {
	const trips = getTrips();
	// Row three shares the width between whichever of Now and Places exist.
	const both = now.length > 0 && trips.length > 0;
	const row3 = both ? { md: 3, lg: 6 } : { md: 6, lg: 12 };
	return (
		<Bench settle>
			<BioTile order={0} />
			<RoomTile order={8} />
			<ReadingTile order={4} />
			<SetTile order={2} />
			<WritingTile order={3} />
			<NowTile span={row3} order={5} />
			<PlacesTile trips={trips} span={row3} order={6} />
			<WorkTile order={7} />
		</Bench>
	);
}
