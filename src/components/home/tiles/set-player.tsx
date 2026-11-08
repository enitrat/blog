'use client';

import NumberFlow, { NumberFlowGroup } from '@number-flow/react';
import { ArrowUpRightIcon, Disc3Icon, PauseIcon, PlayIcon } from 'lucide-react';
import {
	motion,
	useAnimationFrame,
	useMotionValue,
	useReducedMotion,
	useSpring,
} from 'motion/react';
import Image, { type StaticImageData } from 'next/image';
import { useEffect, useSyncExternalStore } from 'react';
import { SET } from '@/consts';
import { spring } from '@/lib/motion';
import { setPlayer } from '@/lib/set-player';
import { cn } from '@/lib/utils';
import { Card, CardHeader } from '../card';

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
		<span
			aria-hidden="true"
			className="inline-flex min-w-[4ch] font-mono text-xs tracking-[-0.01em] text-faint tabular-nums"
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
	);
}

/** The bars dance while playing and stand still otherwise, and under reduced motion. */
function Waveform({ playing }: { playing: boolean }) {
	const reduce = useReducedMotion();
	const dance = playing && !reduce;
	return (
		<div
			aria-hidden="true"
			className="my-1.5 flex min-h-0 flex-1 items-center gap-0.5 overflow-hidden"
		>
			{BARS.map((h, k) => (
				<motion.i
					key={k.toString()}
					className={cn(
						'block w-1 flex-none rounded-[2px] transition-colors duration-200',
						playing ? 'bg-primary' : 'bg-border-strong',
					)}
					style={{ height: `${h * 100}%` }}
					initial={false}
					animate={dance ? { scaleY: [0.45, 1] } : { scaleY: 1 }}
					transition={
						dance
							? {
									duration: 0.9,
									repeat: Number.POSITIVE_INFINITY,
									repeatType: 'reverse',
									ease: 'easeInOut',
									delay: -((k * 73) % 900) / 1000,
								}
							: { duration: 0.2 }
					}
				/>
			))}
		</div>
	);
}

/**
 * The set card: cover, title, waveform, play button. It drives the shared SoundCloud widget, which
 * loads on the first press. While playing the cover becomes a record (square to round, turning
 * slowly, a label and grooves fading in) and the waveform moves; under reduced motion the cover
 * turns round and the bars light, but nothing spins.
 */
export function SetPlayer({
	art,
	date,
	className,
}: {
	art: StaticImageData;
	date: string;
	className?: string;
}) {
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

	const shape = { borderRadius: playing ? '50%' : 'var(--radius-media)' };
	const move = reduce ? { duration: 0 } : spring.fan;
	return (
		<Card index={2} labelledBy="set-title" className={className}>
			<CardHeader id="set-title" icon={<Disc3Icon />} label="Last set">
				<time dateTime={SET.recorded} className="font-mono text-xs tracking-[-0.01em] tabular-nums">
					{date}
				</time>
			</CardHeader>
			<div className="mt-3 flex min-h-0 flex-1 gap-3">
				<div className="relative aspect-square h-full flex-none">
					<motion.div
						aria-hidden="true"
						className="absolute inset-[8%] overflow-hidden blur-[22px] saturate-[1.6]"
						initial={false}
						animate={{ scale: playing ? 1.3 : 1.1, opacity: playing ? 0.9 : 0.55 }}
						transition={move}
					>
						<Image src={art} alt="" sizes="10rem" className="size-full object-cover" />
					</motion.div>
					<motion.div
						className="relative size-full overflow-hidden shadow-[0_6px_16px_-6px_oklch(0_0_0/0.4)]"
						initial={false}
						animate={{ ...shape, scale: playing ? 0.94 : 1 }}
						transition={move}
					>
						<motion.div className="size-full" style={{ rotate: angle }}>
							<Image src={art} alt="" sizes="10rem" className="size-full object-cover" />
							<span
								aria-hidden="true"
								className={cn(
									'absolute inset-0 rounded-full transition-opacity duration-300',
									playing ? 'opacity-100' : 'opacity-0',
								)}
								style={{
									background:
										'radial-gradient(circle, var(--card) 0 5%, oklch(0.1 0 0 / 0.85) 5.5% 7%, transparent 7.5% 30%), repeating-radial-gradient(circle, transparent 0 3px, oklch(0 0 0 / 0.08) 3px 4px)',
								}}
							/>
						</motion.div>
					</motion.div>
				</div>
				<div className="flex min-w-0 flex-1 flex-col">
					<p className="truncate text-lg leading-[1.2] font-semibold tracking-[-0.015em]">
						{SET.title}
					</p>
					<p className="mt-0.5 font-mono text-xs tracking-[-0.01em] text-faint">{SET.artist}</p>
					<Waveform playing={playing} />
					<div className="flex items-center gap-2">
						<motion.button
							type="button"
							aria-pressed={playing}
							aria-label={`${playing ? 'Pause' : 'Play'} ${SET.title}`}
							onClick={() => setPlayer.toggle()}
							whileHover={reduce ? undefined : { scale: 1.08, transition: spring.pop }}
							whileTap={reduce ? undefined : { scale: 0.9, transition: spring.tap }}
							className="grid size-9 flex-none cursor-pointer place-items-center rounded-full bg-primary text-primary-foreground shadow-[0_4px_12px_-4px_var(--primary)] outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-card"
						>
							{playing ? (
								<PauseIcon aria-hidden className="size-4 fill-current stroke-none" />
							) : (
								<PlayIcon aria-hidden className="size-4 fill-current stroke-none" />
							)}
						</motion.button>
						<Elapsed />
						<a
							href={SET.url}
							target="_blank"
							rel="noopener noreferrer"
							aria-label="SoundCloud"
							className="ml-auto grid size-7 place-items-center rounded-media text-faint outline-none transition-colors hover:bg-muted hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
						>
							<ArrowUpRightIcon aria-hidden className="size-4" />
						</a>
					</div>
				</div>
			</div>
		</Card>
	);
}
