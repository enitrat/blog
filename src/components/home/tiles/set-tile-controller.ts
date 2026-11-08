/**
 * The set tile is the player: it drives the shared SoundCloud widget (`lib/set-player.ts`), so
 * playing here or from the room updates both. While playing the artwork becomes a turning record
 * and the header shows the elapsed time. Imported only in the browser: `set-player` touches
 * `document` when it loads.
 */
import { animate, cancelFrame, frame, motionValue } from 'motion';
import { SET } from '@/consts';
import { setReadout } from '@/ds/components/objects/readout-controller';
import { prefersReducedMotion, SPRING_RECORD } from '@/lib/motion';
import { setPlayer } from '@/lib/set-player';

/** One turn per 9.6s, calm. */
const CRUISE = 360 / 9.6;

/** Wires the `[data-set-tile]` tile; returns the teardown. */
export function initSetTile(tile: HTMLElement): () => void {
	const elapsed = tile.querySelector<HTMLElement>('.elapsed [data-readout]');
	const button = tile.querySelector<HTMLButtonElement>('[data-set-toggle]');
	const spin = tile.querySelector<HTMLElement>('.set-spin');

	// The record has inertia: `speed` (deg/s) eases up to cruise or down to zero on a spring, and a
	// frame loop integrates it into `angle`. A paused record stays where it stopped.
	const speed = motionValue(0);
	let angle = 0;
	let last = 0;
	let running: { stop: () => void } | null = null;
	const step = ({ timestamp }: { timestamp: number }) => {
		// clamp: a tab coming back from hidden must not jump the record
		angle = (angle + (speed.get() * Math.min(timestamp - last, 50)) / 1000) % 360;
		last = timestamp;
		if (spin) spin.style.transform = `rotate(${angle}deg)`;
	};
	const turn = (on: boolean) => {
		running?.stop();
		if (prefersReducedMotion()) {
			// also reached when reduced motion is switched on mid-play: stop turning, keep the angle
			speed.set(0);
			cancelFrame(step);
			return;
		}
		if (on && !speed.get()) {
			last = performance.now();
			frame.update(step, true);
		}
		running = animate(speed, on ? CRUISE : 0, {
			...SPRING_RECORD,
			onComplete: () => {
				if (!on) cancelFrame(step);
			},
		});
	};

	const reduce = matchMedia('(prefers-reduced-motion: reduce)');
	const onReduce = () => turn(tile.hasAttribute('data-playing'));
	reduce.addEventListener('change', onReduce);

	const toggle = () => setPlayer.toggle();
	button?.addEventListener('click', toggle);

	// Same rule as the room: it turns from `cueing` on, since the press has been heard.
	const offState = setPlayer.subscribe((state) => {
		const on = state !== 'stopped';
		if (on !== tile.hasAttribute('data-playing')) turn(on);
		tile.toggleAttribute('data-playing', on);
		button?.setAttribute('aria-pressed', String(on));
		button?.setAttribute('aria-label', `${on ? 'Pause' : 'Play'} ${SET.title}`);
		if (!on && elapsed) setReadout(elapsed, '0:00');
	});
	const offPosition = setPlayer.subscribePosition((seconds) => {
		if (elapsed)
			setReadout(elapsed, `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`);
	});

	return () => {
		offState();
		offPosition();
		button?.removeEventListener('click', toggle);
		reduce.removeEventListener('change', onReduce);
		running?.stop();
		cancelFrame(step);
		speed.destroy();
	};
}
