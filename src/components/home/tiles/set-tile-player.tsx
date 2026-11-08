'use client';

import { useEffect, useRef } from 'react';
import { SET } from '@/consts';
import { Button, Icon, Link, Magnetic, Tilt } from '@/ds';
import { Plate } from '@/ds/components/objects/plate';
import type { Img } from '@/lib/media';

/**
 * The set tile's body: the artwork disc, the title, and the play button. The controller loads in
 * the browser on mount, so the SoundCloud widget code never runs on the server. Without JS the
 * SoundCloud link is the fallback.
 */
export function SetTilePlayer({ art, date }: { art: Img; date: string }) {
	const ref = useRef<HTMLDivElement>(null);
	useEffect(() => {
		const tile = ref.current?.closest<HTMLElement>('[data-set-tile]');
		if (!tile) return;
		let teardown: (() => void) | undefined;
		let gone = false;
		import('./set-tile-controller').then(({ initSetTile }) => {
			if (!gone) teardown = initSetTile(tile);
		});
		return () => {
			gone = true;
			teardown?.();
		};
	}, []);
	return (
		<div ref={ref} className="set-body">
			<div className="set-disc">
				<Tilt>
					<div className="set-spin">
						<Plate src={art} alt="" ratio="1/1" sizes="(min-width: 40rem) 16rem, 8rem" />
					</div>
				</Tilt>
			</div>
			<div className="grid min-w-0 gap-3">
				<div>
					<p className="m-0 text-ui font-semibold leading-normal">{SET.title}</p>
					<p className="set-meta">
						{SET.artist} · <time dateTime={SET.recorded}>{date}</time>
					</p>
				</div>
				<div className="flex flex-wrap items-center gap-x-(--gap-bench) gap-y-2">
					<Magnetic strength={0.15}>
						<Button
							variant="inverse"
							pressed={false}
							aria-label={`Play ${SET.title}`}
							data-set-toggle=""
						>
							<span className="when-stopped">
								<Icon name="play" />
								Play
							</span>
							<span className="when-playing">
								<Icon name="pause" />
								Pause
							</span>
						</Button>
					</Magnetic>
					<Link variant="external" href={SET.url}>
						SoundCloud
					</Link>
				</div>
			</div>
		</div>
	);
}
