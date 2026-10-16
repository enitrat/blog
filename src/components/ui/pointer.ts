/**
 * Pointer primitives: one script for every `Tilt` and `Magnetic` on the page. Springs come
 * from `lib/motion.ts`. Inert without a hover pointer, and under reduced motion at the moment
 * of each event (the preference can change while the page is open).
 */
import { animate } from 'motion';
import { canHover, prefersReducedMotion, SPRING_MOUSE } from '../../lib/motion';

const live = () => canHover() && !prefersReducedMotion();
const spring = SPRING_MOUSE;

/** Bumped when layout may have moved under a still pointer; every cached rect compares against it. */
let epoch = 0;
const bump = () => {
	epoch++;
};

function initTilt(el: HTMLElement) {
	const max = Number(el.dataset.tiltMax ?? 7.5);
	// The box is the parent's: `el` rotates, so its own rect would shift under the pointer.
	const box = el.parentElement ?? el;
	const to = (rx: number, ry: number) =>
		animate(el, { rotateX: rx, rotateY: ry, transformPerspective: 1000 }, spring);
	// One read per hover: measured on the first move after entering, dropped on scroll, resize and transitions.
	let r: DOMRect | null = null;
	let seen = -1;
	box.addEventListener('pointerenter', () => {
		r = null;
	});
	box.addEventListener('pointermove', (e) => {
		if (e.pointerType !== 'mouse' || !live()) return;
		if (!r || seen !== epoch) {
			r = box.getBoundingClientRect();
			seen = epoch;
		}
		const px = (e.clientX - r.left) / r.width;
		const py = (e.clientY - r.top) / r.height;
		el.dataset.tiltActive = '';
		el.style.setProperty('--glare-x', `${px * 100}%`);
		el.style.setProperty('--glare-y', `${py * 100}%`);
		to((0.5 - py) * 2 * max, (px - 0.5) * 2 * max);
	});
	box.addEventListener('pointerleave', () => {
		r = null;
		if (!('tiltActive' in el.dataset)) return;
		delete el.dataset.tiltActive;
		to(0, 0);
	});
}

const MARGIN = 16;

/** The wrapper never moves; its first child is what drifts, so the wrapper's rect is the home box. */
function initMagnetic(els: HTMLElement[]) {
	const magnets = els.map((el) => ({
		el,
		item: (el.firstElementChild as HTMLElement | null) ?? el,
		cx: 0,
		cy: 0,
		hw: 0,
		hh: 0,
		on: false,
	}));
	let dirty = true;
	let queued = false;
	let last: PointerEvent | null = null;

	// x/y, not a transform string: a retarget then keeps the live velocity and reads no computed style.
	const drift = (s: (typeof magnets)[number], x: number, y: number) =>
		animate(s.item, { x, y }, spring);
	const reset = () => {
		last = null;
		for (const s of magnets) {
			if (!s.on) continue;
			s.on = false;
			drift(s, 0, 0);
		}
	};
	const measure = () => {
		for (const s of magnets) {
			const r = s.el.getBoundingClientRect();
			s.cx = r.left + r.width / 2;
			s.cy = r.top + r.height / 2;
			s.hw = r.width / 2 + MARGIN;
			s.hh = r.height / 2 + MARGIN;
		}
		dirty = false;
	};
	const run = () => {
		queued = false;
		if (!last) return;
		if (dirty) measure();
		const { clientX: x, clientY: y } = last;
		for (const s of magnets) {
			const inside = Math.abs(x - s.cx) <= s.hw && Math.abs(y - s.cy) <= s.hh;
			if (!inside && !s.on) continue;
			s.on = inside;
			const k = Number(s.el.dataset.magneticStrength ?? 0.35);
			drift(s, inside ? (x - s.cx) * k : 0, inside ? (y - s.cy) * k : 0);
		}
	};

	document.addEventListener('pointermove', (e) => {
		if (e.pointerType !== 'mouse' || !live()) return;
		last = e;
		if (queued) return;
		queued = true;
		requestAnimationFrame(run);
	});
	// Layout moves under a still pointer: re-measure lazily, and let go meanwhile.
	const invalidate = () => {
		dirty = true;
		reset();
	};
	addEventListener('scroll', invalidate, { passive: true, capture: true });
	addEventListener('resize', invalidate, { passive: true });
	new ResizeObserver(() => {
		dirty = true;
	}).observe(document.documentElement);
	document.documentElement.addEventListener('pointerleave', reset);
}

if (typeof document !== 'undefined') {
	addEventListener('scroll', bump, { passive: true, capture: true });
	addEventListener('resize', bump, { passive: true });
	addEventListener('transitionend', bump, { capture: true });
	document.querySelectorAll<HTMLElement>('[data-tilt]').forEach(initTilt);
	const magnets = [...document.querySelectorAll<HTMLElement>('[data-magnetic]')];
	if (magnets.length) initMagnetic(magnets);
}
