'use client';

import { ArrowRightIcon } from 'lucide-react';
import { motion, useReducedMotion } from 'motion/react';
import Link from 'next/link';
import type { ReactNode } from 'react';
import { hoverScale, spring } from '@/lib/motion';

/** One work entry: logo, title, role, period and summary. The logo and arrow spring on hover. */
export function WorkRow({
	href,
	logo,
	title,
	role,
	period,
	summary,
}: {
	href: string;
	logo: ReactNode;
	title: string;
	role?: string;
	period?: string;
	summary?: string;
}) {
	const reduce = useReducedMotion();
	const meta = [role, period].filter(Boolean).join(' · ');
	return (
		<motion.li
			initial="rest"
			whileHover="hover"
			whileFocus="hover"
			className="border-t border-border first:border-t-0"
		>
			<Link
				href={href}
				className="-mx-3 flex items-start gap-4 rounded-xl px-3 py-4 outline-none transition-colors hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring"
			>
				<motion.span
					className="flex flex-none"
					variants={{ rest: { scale: 1 }, hover: { scale: reduce ? 1 : hoverScale.small } }}
					transition={spring.pop}
				>
					{logo}
				</motion.span>
				<span className="min-w-0 flex-1">
					<h3 className="text-sm font-semibold text-balance text-foreground">{title}</h3>
					{meta && (
						<span className="mt-0.5 block text-xs font-medium text-faint tabular-nums">{meta}</span>
					)}
					{summary && (
						<span className="mt-1.5 block text-sm leading-relaxed text-pretty text-muted-foreground">
							{summary}
						</span>
					)}
				</span>
				<motion.span
					aria-hidden
					className="mt-3 hidden flex-none text-faint sm:block"
					variants={{
						rest: { opacity: 0, x: reduce ? 0 : -4 },
						hover: { opacity: 1, x: 0 },
					}}
					transition={spring.enter}
				>
					<ArrowRightIcon className="size-4" />
				</motion.span>
			</Link>
		</motion.li>
	);
}
