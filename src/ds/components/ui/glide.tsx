'use client';

import { animate, type MotionValue, motion, useMotionTemplate, useMotionValue } from 'motion/react';
import {
	type ComponentProps,
	type FocusEvent,
	type PointerEvent,
	useEffect,
	useRef,
	useState,
} from 'react';
import { cn } from '@/ds/lib/cn';
import { canHover, SPRING_GLIDE, transition } from '@/lib/motion';

const ITEM = '[data-glide-item]';
const TAGS = { div: motion.div, ul: motion.ul, nav: motion.nav } as const;
type Insets = [top: number, right: number, bottom: number, left: number];

/**
 * A group of items (each `data-glide-item`) sharing ONE highlight that glides to the hovered or
 * keyboard-focused item. The highlight is the group's `::before`, clipped to the active item by
 * four insets (`--gt/--gr/--gb/--gl`) that spring on `SPRING_GLIDE`, so only clip-path and opacity
 * move. The first hover appears in place; a move glides; reduced motion jumps but still fades.
 * Without JS the items keep their own hover styles.
 */
export function Glide({
	as = 'div',
	shape = 'pill',
	className,
	style,
	...props
}: Omit<
	ComponentProps<'div'>,
	'onAnimationStart' | 'onDrag' | 'onDragEnd' | 'onDragStart' | 'onDragOver'
> & {
	as?: keyof typeof TAGS;
	shape?: 'pill' | 'row';
}) {
	const Tag = TAGS[as] as typeof motion.div;
	const ref = useRef<HTMLDivElement>(null);
	const [ready, setReady] = useState(false);
	const [on, setOn] = useState(false);
	const current = useRef<Element | null>(null);
	// The group's rect is read once per hover; dropped on leave, scroll, resize and transitions.
	const box = useRef<DOMRect | null>(null);
	const insets: MotionValue<number>[] = [
		useMotionValue(0),
		useMotionValue(0),
		useMotionValue(0),
		useMotionValue(0),
	];
	const [gt, gr, gb, gl] = insets;
	const vars = {
		'--gt': useMotionTemplate`${gt}px`,
		'--gr': useMotionTemplate`${gr}px`,
		'--gb': useMotionTemplate`${gb}px`,
		'--gl': useMotionTemplate`${gl}px`,
	};

	const measure = (item: Element): Insets => {
		const group = ref.current as HTMLElement;
		box.current ??= group.getBoundingClientRect();
		const g = box.current;
		const i = item.getBoundingClientRect();
		return [i.top - g.top, g.right - i.right, g.bottom - i.bottom, i.left - g.left];
	};
	const jump = (item: Element) => {
		box.current = null;
		measure(item).forEach((v, k) => {
			insets[k].stop();
			insets[k].set(v);
		});
	};
	const show = (item: Element) => {
		if (item === current.current) return;
		const wasOn = current.current !== null;
		current.current = item;
		setOn(true);
		if (!wasOn) return jump(item);
		measure(item).forEach((v, k) => {
			animate(insets[k], v, transition(SPRING_GLIDE));
		});
	};
	const hide = () => {
		current.current = null;
		box.current = null;
		setOn(false);
	};
	const focused = () => ref.current?.querySelector(`${ITEM}:focus-visible`) ?? null;

	// biome-ignore lint/correctness/useExhaustiveDependencies: `jump` reads only refs and stable motion values
	useEffect(() => {
		const group = ref.current;
		if (!group) return;
		const drop = () => {
			box.current = null;
		};
		addEventListener('scroll', drop, { passive: true, capture: true });
		addEventListener('resize', drop, { passive: true });
		addEventListener('transitionend', drop, { capture: true });
		// fonts and wrapping move items under a visible highlight
		const resize = new ResizeObserver(() => current.current && jump(current.current));
		resize.observe(group);
		for (const item of group.querySelectorAll(ITEM)) resize.observe(item);
		setReady(true);
		return () => {
			resize.disconnect();
			removeEventListener('scroll', drop, { capture: true });
			removeEventListener('resize', drop);
			removeEventListener('transitionend', drop, { capture: true });
		};
	}, []);

	return (
		<Tag
			ref={ref}
			data-glide=""
			data-shape={shape}
			data-glide-ready={ready ? '' : undefined}
			data-glide-on={on ? '' : undefined}
			className={cn('glide glide-highlight', className)}
			style={{ ...style, ...vars }}
			onPointerOver={(e: PointerEvent) => {
				if (!canHover()) return;
				const item = (e.target as Element).closest(ITEM);
				if (item && ref.current?.contains(item)) show(item);
			}}
			onPointerLeave={() => {
				const f = focused();
				f ? show(f) : hide();
			}}
			onFocus={(e: FocusEvent) => {
				if (e.target.matches(`${ITEM}:focus-visible`)) show(e.target);
			}}
			onBlur={(e: FocusEvent) => {
				const next = e.relatedTarget as Node | null;
				if (next && ref.current?.contains(next)) return;
				if (!ref.current?.matches(':hover')) hide();
			}}
			{...props}
		/>
	);
}
