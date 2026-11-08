import { AboutTile } from '@/components/home/tiles/about-tile';
import { BenchTile } from '@/components/home/tiles/bench-tile';
import { ElsewhereTile } from '@/components/home/tiles/elsewhere-tile';
import { NowTile } from '@/components/home/tiles/now-tile';
import { PlacesTile } from '@/components/home/tiles/places-tile';
import { ReadingTile } from '@/components/home/tiles/reading-tile';
import { SetTile } from '@/components/home/tiles/set-tile';
import { WorkTile } from '@/components/home/tiles/work-tile';
import { WritingTile } from '@/components/home/tiles/writing-tile';
import { SITE_DESCRIPTION } from '@/consts';
import { bench } from '@/data/lifts';
import { now } from '@/data/now';
import { getTrips } from '@/lib/content';
import { pageMeta } from '@/lib/meta';

export const metadata = {
	...pageMeta({ title: 'Mathieu Saugier', description: SITE_DESCRIPTION, path: '/' }),
	title: { absolute: 'Mathieu Saugier' },
};

/**
 * The bento. Rows have a fixed height, so the grid keeps its proportions on any window and the
 * spare height sits below it. Phones: 2 columns of 176px rows; tablets: 4 of 160px; from 1100px,
 * 8 columns of 194px rows, which fill a 1440x900 window under the header:
 *
 *   About (2 x 2)   | Writing (6 x 2)
 *   Elsewhere       | Set (4 x 1)    | Reading
 *   Work (6 x 1)                     | (2 x 2)
 *
 * Places, Now and Bench, once they have data, share a fifth row, widths 3 : 2 : 1.
 */
export default function Home() {
	const trips = getTrips();
	const extras = [
		trips.length > 0 && <PlacesTile key="places" trips={trips} index={6} className="grow-3" />,
		now.length > 0 && <NowTile key="now" index={7} className="grow-2" />,
		bench && <BenchTile key="bench" {...bench} index={8} className="grow" />,
	].filter(Boolean);
	return (
		<div className="grid grid-flow-dense auto-rows-[11rem] grid-cols-2 gap-gutter pb-6 sm:auto-rows-[10rem] sm:grid-cols-4 min-[68.75rem]:auto-rows-[12.125rem] min-[68.75rem]:grid-cols-8">
			<AboutTile className="col-span-2 row-span-2 min-[68.75rem]:col-[1/3] min-[68.75rem]:row-[1/3]" />
			<WritingTile className="col-span-2 row-span-2 min-[68.75rem]:col-[3/9] min-[68.75rem]:row-[1/3]" />
			<SetTile className="col-span-2 min-[68.75rem]:col-[3/7] min-[68.75rem]:row-3" />
			<ReadingTile className="row-span-2 min-[68.75rem]:col-[7/9] min-[68.75rem]:row-[3/5]" />
			<ElsewhereTile className="row-span-2 min-[68.75rem]:col-[1/3] min-[68.75rem]:row-3" />
			<WorkTile className="col-span-2 min-[68.75rem]:col-[1/7] min-[68.75rem]:row-4" />
			{extras.length > 0 && (
				<div className="contents *:col-span-2 min-[68.75rem]:col-span-full min-[68.75rem]:flex min-[68.75rem]:gap-gutter min-[68.75rem]:*:basis-0">
					{extras}
				</div>
			)}
		</div>
	);
}
