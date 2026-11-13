'use client';

import NumberFlow, { NumberFlowGroup } from '@number-flow/react';
import { ArrowUpRightIcon, PauseIcon, PlayIcon } from 'lucide-react';
import { motion, useMotionValue, useReducedMotion, useSpring } from 'motion/react';
import Image, { type StaticImageData } from 'next/image';
import { useEffect, useSyncExternalStore } from 'react';
import { useTileAwake } from '@/components/home/bento-tile';
import { SET } from '@/consts';
import { spring } from '@/lib/motion';
import { setPlayer } from '@/lib/set-player';
import { cn } from '@/lib/utils';

/** One turn per 2.4s, as the prototype's record. */
const CRUISE = 360 / 2.4;
const { type: _, ...speedSpring } = spring.layout;

// More bars than fit: the waveform clips them, so a wide card shows more bars, not fatter ones.
const BARS = Array.from({ length: 96 }, (_, k) =>
	Number((0.25 + 0.75 * Math.abs(Math.sin(k * 1.7) * Math.cos(k * 0.43))).toFixed(2)),
);

/** On from `cueing`, like the room: the press has been heard. */
function usePlaying() {
	const state = useSyncExternalStore(
		setPlayer.subscribe,
		() => setPlayer.state,
		() => 'stopped' as const,
	);
	return state !== 'stopped';
}

function Elapsed() {
	const seconds = useSyncExternalStore(
		setPlayer.subscribePosition,
		() => setPlayer.position,
		() => 0,
	);
	return (
		<span aria-hidden="true" className="inline-flex min-w-[4ch] type-meta text-faint">
			<NumberFlowGroup>
				<NumberFlow value={Math.floor(seconds / 60)} />
				<NumberFlow
					value={seconds % 60}
					prefix=":"
					digits={{ 1: { max: 5 } }}
					format={{ minimumIntegerDigits: 2 }}
				/>
			</NumberFlowGroup>
		</span>
	);
}

/**
 * The bars dance while playing and stand still otherwise, and under reduced motion. They beat in
 * CSS, each offset by a negative delay, so 96 bars cost the main thread nothing.
 */
function Waveform({ playing }: { playing: boolean }) {
	return (
		<div
			aria-hidden="true"
			className="my-1.5 flex min-h-0 flex-1 items-center gap-0.5 overflow-hidden"
		>
			{BARS.map((h, k) => (
				<i
					key={k.toString()}
					className={cn(
						'block w-1 flex-none rounded-full transition-colors duration-200',
						playing ? 'animate-bar bg-primary' : 'bg-border-strong',
					)}
					style={{ height: `${h * 100}%`, animationDelay: `-${(k * 73) % 900}ms` }}
				/>
			))}
		</div>
	);
}

/**
 * The last set, inside its home tile: cover, title, waveform, play button. It drives the shared
 * SoundCloud widget, which loads on the first press. While the tile is held and the set stopped,
 * the record slides out of its sleeve. While playing the cover becomes a record (square to round, turning
 * slowly, a label and grooves fading in) and the waveform moves; under reduced motion the cover
 * turns round and the bars light, but nothing spins. The spin is the one motion here that is not an
 * animation (a per-frame angle), so it is the one that checks reduced motion itself.
 */
export function LastSet({ art }: { art: StaticImageData }) {
	const playing = usePlaying();
	const reduce = useReducedMotion();
	// Held and stopped, the record slides halfway out of its sleeve and the sleeve tips back.
	const peek = useTileAwake() && !playing;

	// The record has inertia: `speed` (deg/s) springs up to cruise or down to zero, and each frame
	// adds it to `angle`. A paused record stays where it stopped.
	const speed = useSpring(0, speedSpring);
	const angle = useMotionValue(0);
	useEffect(() => {
		speed.set(playing && !reduce ? CRUISE : 0);
	}, [playing, reduce, speed]);
	// The frame loop runs only while the record turns or spins down, not for the page's lifetime.
	useEffect(() => {
		let frame = 0;
		let last = 0;
		const tick = (now: number) => {
			const v = speed.get();
			// clamp: a tab coming back from hidden must not jump the record
			angle.set((angle.get() + (v * Math.min(now - last, 50)) / 1000) % 360);
			last = now;
			frame = v || speed.isAnimating() ? requestAnimationFrame(tick) : 0;
		};
		const start = () => {
			if (frame) return;
			last = performance.now();
			frame = requestAnimationFrame(tick);
		};
		const stop = speed.on('change', start);
		return () => {
			stop();
			cancelAnimationFrame(frame);
		};
	}, [speed, angle]);

	const shape = { borderRadius: playing ? '50%' : 'var(--radius-media)' };
	const move = spring.fan;
	return (
		<div className="mt-3 flex min-h-0 flex-1 gap-5">
			<div className="relative aspect-square w-[42%] flex-none self-center">
				<motion.div
					aria-hidden="true"
					className="absolute inset-[6%] overflow-hidden ambient-glow"
					initial={false}
					animate={{ scale: playing ? 1.3 : 1.08, opacity: playing ? 0.9 : 0.75 }}
					transition={move}
				>
					<Image src={art} alt="" sizes="10rem" className="size-full object-cover" />
				</motion.div>
				<motion.span
					aria-hidden="true"
					className="absolute inset-[4%] rounded-full shadow-art record"
					initial={false}
					animate={{ x: peek ? '20%' : '0%', rotate: peek ? 30 : 0 }}
					transition={move}
				/>
				<motion.div
					className="relative size-full overflow-hidden shadow-art"
					initial={false}
					animate={{ ...shape, scale: playing ? 0.94 : 1, rotate: peek ? -3 : 0 }}
					transition={move}
				>
					<motion.div className="size-full" style={{ rotate: angle }}>
						<Image src={art} alt="" sizes="10rem" className="size-full object-cover" />
						<span
							aria-hidden="true"
							className={cn(
								'absolute inset-0 rounded-full record-grooves transition-opacity duration-300',
								playing ? 'opacity-100' : 'opacity-0',
							)}
						/>
					</motion.div>
				</motion.div>
			</div>
			<div className="relative z-10 flex min-w-0 flex-1 flex-col">
				<p className="line-clamp-2 type-subtitle text-balance">{SET.title}</p>
				<p className="mt-0.5 type-meta text-faint">{SET.artist}</p>
				<Waveform playing={playing} />
				<div className="flex items-center gap-2">
					<motion.button
						type="button"
						aria-pressed={playing}
						aria-label={`${playing ? 'Pause' : 'Play'} ${SET.title}`}
						onClick={() => setPlayer.toggle()}
						whileHover={{ scale: 1.08, transition: spring.pop }}
						whileTap={{ scale: 0.95, transition: spring.tap }}
						className="relative grid size-9 flex-none cursor-pointer place-items-center rounded-full bg-primary text-primary-foreground shadow-glow hit-area"
					>
						{playing ? (
							<PauseIcon aria-hidden className="size-4 fill-current stroke-none" />
						) : (
							<PlayIcon aria-hidden className="size-4 translate-x-px fill-current stroke-none" />
						)}
					</motion.button>
					<Elapsed />
					<a
						href={SET.url}
						target="_blank"
						rel="noopener noreferrer"
						aria-label={`Open ${SET.title} on SoundCloud`}
						className="relative ml-auto grid size-7 place-items-center rounded-media text-faint transition-colors hit-area hover:bg-muted hover:text-foreground"
					>
						<ArrowUpRightIcon aria-hidden className="size-4" />
					</a>
				</div>
			</div>
		</div>
	);
}
