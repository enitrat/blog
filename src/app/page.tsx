import { Cell } from '@/components/home/cell';
import { RoomTile } from '@/components/home/room-tile';
import { Sidebar } from '@/components/home/sidebar';
import { NowTile } from '@/components/home/tiles/now-tile';
import { PlacesTile } from '@/components/home/tiles/places-tile';
import { ReadingTile } from '@/components/home/tiles/reading-tile';
import { SetTile } from '@/components/home/tiles/set-tile';
import { TkTile } from '@/components/home/tiles/tk-tile';
import { WorkTile } from '@/components/home/tiles/work-tile';
import { WritingTile } from '@/components/home/tiles/writing-tile';
import { SITE_DESCRIPTION } from '@/consts';
import { now } from '@/data/now';
import { getTrips } from '@/lib/content';
import { pageMeta } from '@/lib/meta';

export const metadata = pageMeta({
	title: 'Mathieu Saugier',
	description: SITE_DESCRIPTION,
	path: '/',
});

/** A cell on phones and tablets; on lg a fixed slot of the 4 x 4 grid. */
const CELL = 'min-h-60 lg:min-h-0';

/**
 * From lg the home is one viewport: the sidebar on the left, a 4 x 4 grid of cards on the right.
 * `--chrome` is the height of the header and footer around `main`; the grid takes the rest.
 * Phones stack and scroll.
 */
export default function Home() {
	const trips = getTrips();
	return (
		<div className="flex flex-col gap-6 pb-4 [--chrome:11.25rem] lg:h-[calc(100dvh-var(--chrome))] lg:flex-row lg:gap-0 lg:pb-0">
			<div className="lg:w-72 lg:flex-none lg:border-r lg:border-gray-200 lg:pb-1.5 xl:w-80 dark:lg:border-white/10">
				<Cell index={0} hover={false} className="h-full">
					<Sidebar />
				</Cell>
			</div>
			<div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2 lg:flex-1 lg:grid-cols-[repeat(4,minmax(0,1fr))] lg:grid-rows-[repeat(4,minmax(0,1fr))] lg:pb-1.5 lg:pl-1.5">
				<Cell index={1} className={`${CELL} lg:row-span-2`}>
					<SetTile />
				</Cell>
				<Cell index={2} className={`${CELL} sm:col-span-2 lg:col-span-2 lg:row-span-2`}>
					<WritingTile />
				</Cell>
				<Cell index={3} hover={false} className={`${CELL} lg:row-span-2`}>
					<ReadingTile />
				</Cell>
				<Cell index={4} className={`${CELL} sm:col-span-2 lg:col-span-2 lg:row-span-2`}>
					<WorkTile />
				</Cell>
				<Cell index={5} drag className={`${CELL} lg:row-span-2`}>
					{trips.length > 0 ? <PlacesTile trips={trips} /> : <TkTile what="photo" />}
				</Cell>
				<Cell index={6} drag className={CELL}>
					<RoomTile />
				</Cell>
				<Cell index={7} drag className={CELL}>
					{now.length > 0 ? <NowTile /> : <TkTile what="currently listening" />}
				</Cell>
			</div>
		</div>
	);
}
