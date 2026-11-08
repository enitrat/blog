/**
 * Glide: one controller per `[data-glide]` group. The group's highlight is its `::before`,
 * clipped to the active item with four custom properties (`--gt/--gr/--gb/--gl`, insets in px)
 * that glide on `SPRING_GLIDE`. Opacity is a CSS transition. The first hover appears in place;
 * a move glides; reduced motion jumps but still fades. Keyboard focus works without a hover pointer.
 */
import { animate } from 'motion';
import { canHover, SPRING_GLIDE, transition } from '@/lib/motion';

const ITEM = '[data-glide-item]';

/** Starts the glide on `group`; returns the teardown. */
export function initGlide(group: HTMLElement): () => void {
	let current: Element | null = null;
	let running: { stop: () => void } | null = null;

	// The group's rect is read once per hover; dropped on leave, scroll, resize and transitions.
	let box: DOMRect | null = null;
	const drop = () => {
		box = null;
	};
	const insets = (item: Element) => {
		box ??= group.getBoundingClientRect();
		const g = box;
		const i = item.getBoundingClientRect();
		return {
			'--gt': `${i.top - g.top}px`,
			'--gr': `${g.right - i.right}px`,
			'--gb': `${g.bottom - i.bottom}px`,
			'--gl': `${i.left - g.left}px`,
		};
	};
	const jump = (item: Element) => {
		running?.stop();
		drop();
		for (const [k, v] of Object.entries(insets(item))) group.style.setProperty(k, v);
	};
	const show = (item: Element) => {
		if (item === current) return;
		const on = group.hasAttribute('data-glide-on');
		current = item;
		if (!on) {
			jump(item);
			group.setAttribute('data-glide-on', '');
			return;
		}
		running?.stop();
		running = animate(group, insets(item), transition(SPRING_GLIDE));
	};
	const hide = () => {
		current = null;
		drop();
		group.removeAttribute('data-glide-on');
	};
	const focused = () => group.querySelector(`${ITEM}:focus-visible`);

	const onOver = (e: PointerEvent) => {
		if (!canHover()) return;
		const item = (e.target as Element).closest(ITEM);
		if (item && group.contains(item)) show(item);
	};
	const onLeave = () => {
		const f = focused();
		f ? show(f) : hide();
	};
	const onFocusIn = (e: FocusEvent) => {
		const t = e.target as Element;
		if (t.matches(`${ITEM}:focus-visible`)) show(t);
	};
	const onFocusOut = (e: FocusEvent) => {
		const next = e.relatedTarget as Node | null;
		if (next && group.contains(next)) return;
		if (!group.matches(':hover')) hide();
	};
	group.addEventListener('pointerover', onOver);
	group.addEventListener('pointerleave', onLeave);
	group.addEventListener('focusin', onFocusIn);
	group.addEventListener('focusout', onFocusOut);
	addEventListener('scroll', drop, { passive: true, capture: true });
	addEventListener('resize', drop, { passive: true });
	addEventListener('transitionend', drop, { capture: true });
	// fonts and wrapping move items under a visible highlight
	const resize = new ResizeObserver(() => current && jump(current));
	resize.observe(group);
	for (const item of group.querySelectorAll(ITEM)) resize.observe(item);

	group.setAttribute('data-glide-ready', '');
	return () => {
		running?.stop();
		resize.disconnect();
		group.removeEventListener('pointerover', onOver);
		group.removeEventListener('pointerleave', onLeave);
		group.removeEventListener('focusin', onFocusIn);
		group.removeEventListener('focusout', onFocusOut);
		removeEventListener('scroll', drop, { capture: true });
		removeEventListener('resize', drop);
		removeEventListener('transitionend', drop, { capture: true });
		group.removeAttribute('data-glide-ready');
		group.removeAttribute('data-glide-on');
	};
}
