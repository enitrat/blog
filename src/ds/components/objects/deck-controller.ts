/**
 * Deck controller: one per `[data-deck]`. At rest the deck is stacked; while open (host hovered,
 * or keyboard focus inside) it fans or pulls, and one card is active. The active card comes from
 * the pointer's position over the deck's fixed footprint, never from `:hover` on a moving card,
 * so motion never changes the choice. Springs from `lib/motion.ts`; enter staggers, exit squares
 * the pile up with one knock. Under reduced motion the poses still change, but instantly.
 */
import { animate, frame, type MotionValue, motionValue } from 'motion';
import {
	canHover,
	DEAL_CROSS,
	GATHER,
	prefersReducedMotion,
	SPRING_DEAL,
	SPRING_LAYOUT,
	SPRING_PRESS,
	SPRING_SETTLE,
	STAGGER,
} from '@/lib/motion';
import {
	ACTIVE_EVENT,
	type DeckActive,
	type DeckOptions,
	DIM,
	type Hold,
	KNOCK,
	openPose,
	type Pose,
	pullZone,
	restPose,
	transformOf,
} from './deck-poses';

type Channel = keyof Pose;
type Run = { stop(): void };

/** Starts the deck on `root`; returns the teardown. */
export function initDeck(root: HTMLElement): () => void {
	const cards = [...root.querySelectorAll<HTMLElement>('[data-deck-card]')];
	const n = cards.length;
	if (n === 0) return () => {};
	const cleanups: (() => void)[] = [];
	const on = <K extends keyof HTMLElementEventMap>(
		target: HTMLElement,
		type: K,
		fn: (e: HTMLElementEventMap[K]) => void,
	) => {
		target.addEventListener(type, fn);
		cleanups.push(() => target.removeEventListener(type, fn));
	};

	// Cards behind the front one are lazy images: where a pointer can fan the hand, decode them
	// while idle, so the first fan does not pay for the decode in one long frame. Touch never
	// fans, so it keeps lazy loading.
	if (canHover()) {
		const decode = () => {
			for (const img of root.querySelectorAll('img')) {
				img.loading = 'eager';
				img.decode().catch(() => {});
			}
		};
		if (window.requestIdleCallback) {
			const id = window.requestIdleCallback(decode);
			cleanups.push(() => window.cancelIdleCallback(id));
		} else {
			const id = window.setTimeout(decode, 200);
			cleanups.push(() => clearTimeout(id));
		}
	}

	const o: DeckOptions = {
		spread: Number(root.dataset.spread),
		angle: Number(root.dataset.angle),
		lift: Number(root.dataset.lift),
		hold: root.dataset.hold as Hold,
	};
	const host =
		(root.dataset.host === 'ancestor' && root.closest<HTMLElement>('[data-deck-host]')) || root;
	// One motion value per pose channel per card: a retarget keeps the live velocity, so sweeping
	// zones mid-spring stays continuous (animating a transform string would restart from rest).
	const channels: Record<Channel, MotionValue<number>>[] = cards.map((_, i) => {
		const p = restPose(i, n, o.hold);
		return {
			x: motionValue(p.x),
			y: motionValue(p.y),
			r: motionValue(p.r),
			ry: motionValue(p.ry),
			s: motionValue(p.s),
			d: motionValue(p.d),
		};
	});
	const runs: Partial<Record<Channel, Run>>[] = cards.map(() => ({}));
	const dims = cards.map((card) => card.querySelector<HTMLElement>('.deck-dim'));
	const dimmed = channels.map((c) => c.d.get());
	const draw = cards.map((card, i) => () => {
		const c = channels[i];
		card.style.transform = transformOf({
			x: c.x.get(),
			y: c.y.get(),
			r: c.r.get(),
			ry: c.ry.get(),
			s: c.s.get(),
		});
		// the dim is its own element: only it is touched, and only when the value moved
		const d = c.d.get();
		const dim = dims[i];
		if (d !== dimmed[i] && dim) {
			dimmed[i] = d;
			dim.style.opacity = String(d * DIM);
		}
	});
	channels.forEach((c, i) => {
		for (const v of Object.values(c)) cleanups.push(v.on('change', () => frame.render(draw[i])));
	});
	cleanups.push(() => {
		for (const r of runs) for (const run of Object.values(r)) run.stop();
		for (const c of channels) for (const v of Object.values(c)) v.destroy();
	});

	let open = false;
	/** Hand position of the active card (0 is the front card at rest). */
	let active = 0;
	/** Item on top; dealing rotates it, so card i sits at hand position (i - top + n) % n. */
	let top = 0;
	const pos = (i: number) => (i - top + n) % n;
	let knock = 0;
	cleanups.push(() => clearTimeout(knock));

	const target = (i: number): Pose =>
		open ? openPose(pos(i), n, o, active) : restPose(pos(i), n, o.hold);
	const zOf = (i: number) =>
		open ? (pos(i) === active ? n + 1 : n - Math.abs(pos(i) - active)) : n - pos(i);

	// The held card (the front one at rest) is announced so a Readout can follow it with no tile script.
	let held = 0;
	const announce = () => {
		const item = open ? cards.findIndex((_, i) => pos(i) === active) : top;
		if (item === held) return;
		held = item;
		const caption: DeckActive['caption'] = JSON.parse(cards[item].dataset.caption || '{}');
		root.dispatchEvent(
			new CustomEvent<DeckActive>(ACTIVE_EVENT, { bubbles: true, detail: { item, caption } }),
		);
	};

	/** Spring each card's channels to its target; `stagger` seconds per card; `raise` lifts the whole pile (% of height). */
	function settle(
		spring: typeof SPRING_LAYOUT | typeof SPRING_PRESS | typeof SPRING_DEAL,
		stagger: number,
		raise = 0,
	) {
		clearTimeout(knock);
		const still = prefersReducedMotion();
		cards.forEach((card, i) => {
			for (const r of Object.values(runs[i])) r.stop();
			// z-order and shadow are instant and never animated
			card.style.zIndex = String(zOf(i));
			if (open && pos(i) === active) card.setAttribute('data-active', '');
			else card.removeAttribute('data-active');
			const to = target(i);
			const c = channels[i];
			if (still) {
				// reduced motion: the hand still moves, it just does not travel
				for (const k of Object.keys(c) as Channel[]) c[k].set(to[k]);
				return;
			}
			to.y -= raise;
			const opts = { ...spring, delay: pos(i) * stagger };
			for (const k of Object.keys(c) as Channel[]) runs[i][k] = animate(c[k], to[k], opts);
		});
		if (open) root.setAttribute('data-open', '');
		else root.removeAttribute('data-open');
		root.dataset.active = String(active);
		announce();
	}

	/** Set down: the cards gather with the pile held a little high, then it drops with one small knock. */
	function squareUp() {
		settle(SPRING_PRESS, 0, KNOCK);
		if (prefersReducedMotion()) return;
		knock = window.setTimeout(() => {
			if (open) return;
			cards.forEach((_, i) => {
				runs[i].y?.stop();
				runs[i].y = animate(channels[i].y, target(i).y, SPRING_SETTLE);
			});
		}, GATHER * 1000);
	}

	const setOpen = (next: boolean, at = active) => {
		if (next === open && at === active) return;
		const was = open;
		open = next;
		// a closed deck holds its front card: nothing stale for the next focus or readout
		active = next ? at : 0;
		// enter staggers, exit squares up, active changes are immediate
		if (!was && next) settle(SPRING_LAYOUT, STAGGER);
		else if (was && !next) squareUp();
		else settle(SPRING_PRESS, 0);
	};

	// One rect per pointer session: the root never moves, only its cards do, so it stays valid until
	// the page scrolls or resizes. Nothing else in pointermove reads layout.
	let rect: DOMRect | null = null;
	const forget = () => {
		rect = null;
	};
	// the host can move under the pointer (bench settle, tile lift): re-measure when its motion ends
	// and on the first move after entering, not on every move
	let fresh = false;
	on(host, 'transitionend', forget);
	on(host, 'animationend', forget);
	addEventListener('resize', forget);
	addEventListener('scroll', forget, { passive: true, capture: true });
	cleanups.push(() => {
		removeEventListener('resize', forget);
		removeEventListener('scroll', forget, { capture: true });
	});
	/** The zone under the pointer: a fan splits its fixed footprint by x; a pulled pile is hit-tested by box (`pullZone`). */
	const zone = (x: number, y: number) => {
		rect ??= root.getBoundingClientRect();
		if (o.hold === 'pull')
			return pullZone(
				((x - rect.left) / rect.width) * 100,
				((y - rect.top) / rect.height) * 100,
				n,
			);
		const width = rect.width * (1 + o.spread * (n - 1));
		const left = rect.left - (rect.width * o.spread * (n - 1)) / 2;
		return Math.min(n - 1, Math.max(0, Math.floor(((x - left) / width) * n)));
	};
	// The host may itself be the focused element (a linked Tile), so test it as well as its descendants.
	const keyboardFocus = () =>
		host.matches(':focus-visible') || !!host.querySelector(':focus-visible');

	// Ambient deal: the front card goes to the back. Ends at the first pointer or keyboard
	// interaction, for good (a mouse click's focus is not interaction worth stopping for).
	const every = Number(root.dataset.cycle);
	if (every > 0 && n > 1 && canHover() && !prefersReducedMotion()) {
		let inView = false;
		const crossings = new Set<number>();
		const view = new IntersectionObserver(([e]) => {
			inView = e.isIntersecting;
		});
		const stop = () => {
			clearInterval(timer);
			view.disconnect();
			host.removeEventListener('pointerenter', stop);
			host.removeEventListener('focusin', onFocus);
		};
		const onFocus = (e: FocusEvent) => {
			if (e.target instanceof Element && e.target.matches(':focus-visible')) stop();
		};
		view.observe(root);
		const timer = window.setInterval(() => {
			if (!inView || open || document.hidden || prefersReducedMotion()) return;
			const dealt = top;
			top = (top + 1) % n;
			// keep DOM, tab and reading order the same as the visual one: the front card comes first
			root.append(
				...cards
					.map((_, i) => i)
					.sort((a, b) => pos(a) - pos(b))
					.map((i) => cards[i]),
			);
			settle(SPRING_DEAL, 0);
			// the dealt card crosses the pile before it drops behind
			cards[dealt].style.zIndex = String(n + 1);
			const id = window.setTimeout(() => {
				crossings.delete(id);
				if (!open) cards[dealt].style.zIndex = String(zOf(dealt));
			}, DEAL_CROSS * 1000);
			crossings.add(id);
		}, every * 1000);
		host.addEventListener('pointerenter', stop);
		host.addEventListener('focusin', onFocus);
		cleanups.push(stop, () => {
			for (const id of crossings) clearTimeout(id);
		});
	}
	if (canHover()) {
		const track = (e: PointerEvent) => {
			if (e.pointerType !== 'touch') setOpen(true, zone(e.clientX, e.clientY));
		};
		on(host, 'pointerenter', (e) => {
			forget();
			track(e);
			fresh = true;
		});
		on(host, 'pointermove', (e) => {
			if (fresh) {
				fresh = false;
				forget();
			}
			track(e);
		});
		on(host, 'pointerleave', () => {
			if (!keyboardFocus()) setOpen(false);
		});
	}
	on(host, 'focusin', (e) => {
		const i = cards.findIndex((c) => c.contains(e.target as Node));
		if (e.target instanceof Element && e.target.matches(':focus-visible'))
			setOpen(true, i < 0 ? active : pos(i));
	});
	on(host, 'focusout', (e) => {
		if (!host.contains(e.relatedTarget as Node | null) && !host.matches(':hover')) setOpen(false);
	});

	return () => {
		for (const fn of cleanups) fn();
	};
}
