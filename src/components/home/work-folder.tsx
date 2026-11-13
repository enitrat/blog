'use client';

import { ArrowUpRightIcon } from 'lucide-react';
import { motion } from 'motion/react';
import Image, { type StaticImageData } from 'next/image';
import Link from 'next/link';
import { useTileAwake } from '@/components/home/bento-tile';
import { spring } from '@/lib/motion';

export type FolderSheet = { title: string; role?: string; logo?: StaticImageData };

/**
 * A folder holding the jobs as sheets, the latest in front. While the tile is held the sheets rise
 * out of it; the whole folder opens /work.
 */
export function WorkFolder({ sheets }: { sheets: FolderSheet[] }) {
	const awake = useTileAwake();
	const shown = sheets.slice(0, 3);
	return (
		<Link
			href="/work/"
			className="relative mt-3 flex min-h-0 flex-1 items-end justify-center rounded-media"
		>
			<span className="relative h-[90%] w-[88%] max-w-72">
				<span
					aria-hidden
					className="absolute inset-x-0 top-[10%] bottom-0 rounded-row bg-primary-soft"
				/>
				<span
					aria-hidden
					className="absolute top-[3%] left-0 h-[12%] w-[38%] rounded-t-row bg-primary-soft"
				/>
				{shown.map((s, i) => (
					<motion.span
						key={s.title}
						aria-hidden
						className="absolute inset-x-[8%] top-[16%] flex h-[60%] items-start gap-2 rounded-chip bg-card p-2.5 shadow-card"
						style={{ zIndex: shown.length - i }}
						initial={false}
						animate={{
							y: awake ? -18 - i * 12 : -i * 7,
							rotate: awake ? (i - 1) * 3 : 0,
						}}
						transition={spring.fan}
					>
						{s.logo && (
							<Image src={s.logo} alt="" sizes="16px" className="size-4 flex-none rounded-inline" />
						)}
						<span className="min-w-0">
							<span className="block truncate type-control-sm">{s.title}</span>
							<span className="block truncate type-small text-faint">{s.role}</span>
						</span>
					</motion.span>
				))}
				<span
					aria-hidden
					className="absolute inset-x-0 bottom-0 z-10 h-[55%] rounded-row bg-primary shadow-raised"
				/>
				<span className="absolute bottom-3 left-3.5 z-10 flex items-center gap-1 type-control text-primary-foreground">
					All work
					<ArrowUpRightIcon
						aria-hidden
						className="size-3.5 transition-transform duration-150 ease-out group-data-[held]/tile:translate-x-0.5 group-data-[held]/tile:-translate-y-0.5"
					/>
				</span>
			</span>
		</Link>
	);
}
