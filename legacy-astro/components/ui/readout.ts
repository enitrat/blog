/**
 * Readout client: text that reports a held object's state and changes by rolling. The old value
 * leaves up and fades, the new one enters from below. `char` mode rolls only the characters that
 * changed (aligned from the right, like place value); `text` mode rolls the whole string.
 * Interruptible: a change mid-roll retargets from where the glyph is; an outgoing glyph is never
 * cut short, it finishes fading on its own. Reduced motion is a crossfade. Only transform and opacity animate.
 */
import { animate, frame, type MotionValue, motionValue } from 'motion';
import { prefersReducedMotion, ROLL_FADE, SPRING_ROLL } from '../../lib/motion';
import type { DeckActive } from './deck';
import { ACTIVE_EVENT } from './deck';

type State = { stage: HTMLElement; slots: HTMLElement[]; text: string; char: boolean };
const states = new WeakMap<HTMLElement, State>();
/** A glyph's live position and opacity. Motion values, not computed style: a retarget keeps its velocity and reads no layout. */
const live = new WeakMap<HTMLElement, { y: MotionValue<number>; o: MotionValue<number> }>();

/** Travel of a rolling glyph, % of its own height. */
const TRAVEL = 60;

const glyph = (c: string) =>
	Object.assign(document.createElement('span'), { className: 'readout-glyph', textContent: c });
const slotOf = () => Object.assign(document.createElement('span'), { className: 'readout-slot' });

/** The glyph's motion values, created on first use at (y, o). Every change is painted on the next frame. */
function track(g: HTMLElement, y = 0, o = 1) {
	let m = live.get(g);
	if (!m) {
		const mv = { y: motionValue(y), o: motionValue(o) };
		const draw = () => {
			g.style.transform = `translateY(${mv.y.get()}%)`;
			g.style.opacity = String(mv.o.get());
		};
		mv.y.on('change', () => frame.render(draw));
		mv.o.on('change', () => frame.render(draw));
		live.set(g, mv);
		m = mv;
	}
	return m;
}

function roll(g: HTMLElement, y: number, o: number, spring: typeof SPRING_ROLL | typeof ROLL_FADE) {
	const m = track(g);
	animate(m.y, y, spring);
	return animate(m.o, o, spring);
}

function swap(slot: HTMLElement, c: string, instant: boolean) {
	const old = slot.querySelector<HTMLElement>('.readout-glyph:not([data-leaving])');
	const next = glyph(c);
	slot.append(next);
	if (instant) return old?.remove();
	const still = prefersReducedMotion();
	const spring = still ? ROLL_FADE : SPRING_ROLL;
	if (old) {
		old.dataset.leaving = '';
		roll(old, still ? 0 : -TRAVEL, 0, spring).then(() => old.remove());
	}
	// enters from below (reduced motion: from where it sits), invisible
	const m = track(next, still ? 0 : TRAVEL, 0);
	next.style.opacity = '0';
	next.style.transform = `translateY(${m.y.get()}%)`;
	roll(next, 0, 1, spring);
}

function stateOf(el: HTMLElement): State {
	let st = states.get(el);
	if (st) return st;
	const stage = el.querySelector<HTMLElement>('.readout-stage');
	if (!stage) throw new Error('Readout: missing .readout-stage');
	const text = stage.textContent ?? '';
	const char = el.dataset.mode === 'char';
	st = { stage, slots: [], text, char };
	for (const c of char ? [...text] : [text]) {
		const slot = slotOf();
		slot.append(glyph(c));
		st.slots.push(slot);
	}
	stage.replaceChildren(...st.slots);
	states.set(el, st);
	return st;
}

/** Roll `el` (a `Readout`) to `text`. Safe to call on every tick: an unchanged value does nothing. */
export function setReadout(el: HTMLElement, text: string) {
	const st = stateOf(el);
	if (text === st.text) return;
	st.text = text;
	const sr = el.querySelector('.readout-sr');
	if (sr) sr.textContent = text;
	const chars = st.char ? [...text] : [text];
	while (st.slots.length < chars.length) {
		const slot = slotOf();
		st.stage.prepend(slot);
		st.slots.unshift(slot);
		swap(slot, '', true);
	}
	while (st.slots.length > chars.length) st.slots.shift()?.remove();
	const shown = (slot: HTMLElement) =>
		slot.querySelector('.readout-glyph:not([data-leaving])')?.textContent;
	st.slots.forEach((slot, i) => {
		if (shown(slot) !== chars[i]) swap(slot, chars[i], false);
	});
}

/** Wire every `Readout` that follows a Deck (`data-readout-follow`) so it needs no script of its own. */
export function initReadouts(root: ParentNode = document) {
	for (const el of root.querySelectorAll<HTMLElement>('[data-readout-follow]')) {
		const field = el.dataset.readoutField ?? '';
		document.getElementById(el.dataset.readoutFollow ?? '')?.addEventListener(ACTIVE_EVENT, (e) => {
			setReadout(el, (e as CustomEvent<DeckActive>).detail.caption[field] ?? '');
		});
	}
}
