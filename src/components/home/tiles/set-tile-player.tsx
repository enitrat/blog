'use client';

import NumberFlow, { NumberFlowGroup } from '@number-flow/react';
import { PauseIcon, PlayIcon } from 'lucide-react';
import {
	motion,
	useAnimationFrame,
	useMotionValue,
	useReducedMotion,
	useSpring,
} from 'motion/react';
import Image, { type StaticImageData } from 'next/image';
import { useEffect, useSyncExternalStore } from 'react';
import { Equalizer } from '@/components/motion/equalizer';
import { Magnetic } from '@/components/motion/magnetic';
import { Tilt } from '@/components/motion/tilt';
import { Button } from '@/components/ui/button';
import { TextLink } from '@/components/ui/text-link';
import { SET } from '@/consts';
import { spring } from '@/lib/motion';
import { setPlayer } from '@/lib/set-player';

/** One turn per 9.6s, calm. */
const CRUISE = 360 / 9.6;
const { type: _, ...speedSpring } = spring.layout;

/** On from `cueing`, like the room: the press has been heard. */
function usePlaying() {
	const state = useSyncExternalStore(
		setPlayer.subscribe,
		() => setPlayer.state,
		() => 'stopped' as const,
	);
	return state !== 'stopped';
}

/** The header's elapsed time and equalizer. */
export function SetTileStatus() {
	const playing = usePlaying();
	const seconds = useSyncExternalStore(
		setPlayer.subscribePosition,
		() => setPlayer.position,
		() => 0,
	);
	return (
		<span className="inline-flex items-center gap-2 text-xs text-gray-400 dark:text-iron-400">
			{playing && (
				<span
					aria-hidden="true"
					className="inline-flex min-w-[5ch] justify-end font-mono tabular-nums"
				>
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
			)}
			<Equalizer playing={playing} />
		</span>
	);
}

/**
 * The set tile's body: the artwork disc, the title, and the play button. It drives the shared
 * SoundCloud widget, which loads on the first press. While playing the artwork becomes a record
 * (square to round, turning slowly); under reduced motion it turns round but does not spin.
 */
export function SetTilePlayer({ art, date }: { art: StaticImageData; date: string }) {
	const playing = usePlaying();
	const reduce = useReducedMotion();

	// The record has inertia: `speed` (deg/s) springs up to cruise or down to zero, and each frame
	// adds it to `angle`. A paused record stays where it stopped.
	const speed = useSpring(0, speedSpring);
	const angle = useMotionValue(0);
	useEffect(() => {
		speed.set(playing && !reduce ? CRUISE : 0);
	}, [playing, reduce, speed]);
	useAnimationFrame((_, delta) => {
		const v = speed.get();
		// clamp: a tab coming back from hidden must not jump the record
		if (v) angle.set((angle.get() + (v * Math.min(delta, 50)) / 1000) % 360);
	});

	const shape = { borderRadius: playing ? '50%' : '0.375rem' };
	return (
		<div className="flex items-center gap-4 sm:flex-col sm:items-stretch lg:gap-2">
			<Tilt
				rotationFactor={7.5}
				className="relative isolate w-30 flex-none sm:w-full lg:mx-auto lg:w-24 xl:w-28"
			>
				<motion.div
					aria-hidden="true"
					className="absolute inset-0 -z-10 translate-y-2 overflow-hidden opacity-70 blur-lg saturate-[6] dark:opacity-15 dark:saturate-[12]"
					initial={false}
					animate={shape}
					transition={spring.layout}
				>
					<Image
						src={art}
						alt=""
						sizes="(min-width: 40rem) 16rem, 8rem"
						className="size-full object-cover"
					/>
				</motion.div>
				<motion.div
					className="aspect-square overflow-hidden border border-iron-950/10 bg-gray-100 shadow-xs dark:border-white/10 dark:bg-iron-800"
					initial={false}
					animate={shape}
					transition={spring.layout}
				>
					<motion.div className="size-full" style={{ rotate: angle }}>
						<Image
							src={art}
							alt=""
							sizes="(min-width: 40rem) 16rem, 8rem"
							className="size-full object-cover"
						/>
					</motion.div>
				</motion.div>
			</Tilt>
			<div className="grid min-w-0 gap-3 lg:gap-2">
				<div>
					<p className="text-sm font-semibold">{SET.title}</p>
					<p className="font-mono text-xs text-gray-500 dark:text-iron-400">
						{SET.artist} · <time dateTime={SET.recorded}>{date}</time>
					</p>
				</div>
				<div className="flex flex-wrap items-center gap-x-1.5 gap-y-2">
					<Magnetic intensity={0.15}>
						<Button
							aria-pressed={playing}
							aria-label={`${playing ? 'Pause' : 'Play'} ${SET.title}`}
							onClick={() => setPlayer.toggle()}
						>
							{playing ? <PauseIcon aria-hidden /> : <PlayIcon aria-hidden />}
							{playing ? 'Pause' : 'Play'}
						</Button>
					</Magnetic>
					<TextLink icon="external" href={SET.url}>
						SoundCloud
					</TextLink>
				</div>
			</div>
		</div>
	);
}
