// Keth bench: the live page. One loop, writes attributes on refs, stops offscreen.
import { COMB_REST, L, RAM_REST, readoutOf, SIZES, timeline } from './model.mjs';

/**
 * @param root element holding the svg, the range input and the readout
 * @param ctx { P, translateAlong, say(text), motion(): boolean true when motion is allowed }
 */
export function mount(root, { P, translateAlong, say, motion, follow }) {
	const svg = root.querySelector('svg');
	const input = root.querySelector('input[type=range]');
	const all = (sel) => [...svg.querySelectorAll(sel)];
	const one = (sel) => svg.querySelector(sel);
	const gantry = all('[data-g="gantry"]');
	const blade = one('[data-g="blade"]');
	const stampF = one('[data-g="stampF"]');
	const stampB = one('[data-g="stampB"]');
	const comb = one('[data-g="comb"]');
	const rams = all('[data-g="ram"]');
	const cards = new Map(all('[data-card]').map((el) => [el.dataset.card, el]));
	const slots = all('[data-slot]');
	const tokens = all('[data-token]');
	const keys = all('[data-key]');
	const pins = all('[data-pin]');
	const fades = all('[data-fade]:not([data-proof])');
	const proof = one('[data-proof]');
	const lit = [...all('[data-proof] .iso-solid'), ...all('[data-lamp] .iso-solid:last-child')];
	const up = (dz) => translateAlong(P, [0, 0, dz]);
	const set = (el, name, value) => {
		if (el.getAttribute(name) !== value) el.setAttribute(name, value);
	};
	const show = (el, on) => {
		if (el.style.display !== (on ? '' : 'none')) el.style.display = on ? '' : 'none';
	};

	let txs = SIZES[Number(input.value)];
	let tl = timeline(txs);
	let t = 0;
	let pending = null;
	let mul = 1;
	let touring = true;
	let tourAt = 0;
	let idleTimer = 0;
	let frame = 0;
	let last = 0;
	let visible = false;
	let saidText = '';
	let held = false; // capture scripts freeze the page with pause()

	const builtX = (index) =>
		index === 0
			? L.X0
			: index === 't'
				? L.X0 + L.INIT_W + L.BODY_W
				: L.X0 + L.INIT_W + L.BODY_W * (index - 1);

	function draw() {
		const s = tl.at(t);
		const f = s.fade * mul;
		for (const el of gantry) set(el, 'transform', translateAlong(P, [s.gantry, 0, 0]));
		set(blade, 'transform', up((1 - s.blade) * L.LIFT_BLADE));
		set(stampF, 'transform', up((1 - s.stampF) * L.LIFT_STAMP));
		set(stampB, 'transform', up((1 - s.stampB) * L.LIFT_STAMP));
		set(comb, 'transform', up((1 - s.comb) * COMB_REST));
		for (const el of rams) set(el, 'transform', up((1 - s.press) * RAM_REST));
		for (const [key, el] of cards) {
			const i = key === 't' ? tl.K - 1 : Number(key);
			const on = key === 't' || i <= tl.B;
			show(el, on);
			if (!on) continue;
			const joined = L.X0 + (i === 0 ? 0 : L.INIT_W + L.BODY_W * (i - 1));
			const home = key === 't' ? L.X0 + L.INIT_W + L.BODY_W * tl.B : joined;
			set(el, 'transform', translateAlong(P, [home - builtX(key) + s.cardShift[i], 0, 0]));
			const imp = el.querySelector('[data-imp]');
			set(imp, 'opacity', String(s.stamped[i].toFixed(2)));
		}
		for (const el of slots) {
			set(
				el.querySelector('[data-imp]'),
				'opacity',
				String(s.slotStamped[el.dataset.slot].toFixed(2)),
			);
		}
		for (const el of tokens) {
			const tok = s.tokens.find((x) => x.k === Number(el.dataset.token));
			show(el, Boolean(tok));
			if (!tok) continue;
			const inner = el.firstElementChild;
			set(inner, 'transform', translateAlong(P, tok.pos));
			set(el, 'opacity', tok.alpha.toFixed(2));
		}
		keys.forEach((el, p) => {
			show(el, p < tl.total - 1);
			set(el, 'opacity', s.keys.toFixed(2));
		});
		pins.forEach((el, p) => {
			show(el, p < tl.total - 1);
		});
		for (const el of fades) set(el, 'opacity', f.toFixed(2));
		set(proof, 'opacity', (s.proof * f).toFixed(2));
		const on = s.verified > 0.5;
		for (const el of lit) {
			if (on) el.setAttribute('data-lit', '');
			else el.removeAttribute('data-lit');
		}
		if (follow) follow(s.focus);
		const text = readoutOf(txs, s.stage);
		if (text !== saidText) {
			saidText = text;
			say(text);
		}
	}

	function restart(n) {
		txs = n;
		tl = timeline(n);
		t = motion() ? 0 : tl.duration - 0.6;
		input.value = String(SIZES.indexOf(n));
		input.setAttribute('aria-valuetext', `${n} transactions`);
	}

	function tick(now) {
		const dt = last ? Math.min((now - last) / 1000, 1 / 30) : 1 / 60;
		last = now;
		mul += ((pending ? 0 : 1) - mul) * (1 - Math.exp(-dt / 0.09));
		if (pending && mul < 0.03) {
			restart(pending);
			pending = null;
		}
		t += dt;
		if (t >= tl.duration) {
			if (touring) {
				tourAt = (tourAt + 1) % TOUR.length;
				restart(TOUR[tourAt]);
			} else {
				t = 0;
			}
		}
		draw();
		frame = visible && motion() ? requestAnimationFrame(tick) : 0;
	}
	const TOUR = [30, 50, 20, 40, 10];
	const run = () => {
		if (!frame && visible && motion() && !held) {
			last = 0;
			frame = requestAnimationFrame(tick);
		}
	};
	function pick(n, user) {
		if (user) {
			touring = false;
			clearTimeout(idleTimer);
			idleTimer = setTimeout(() => {
				touring = motion();
				run();
			}, 9000);
		}
		if (!motion()) {
			restart(n);
			draw();
			return;
		}
		if (n !== txs) pending = n;
		run();
	}
	input.addEventListener('input', () => pick(SIZES[Number(input.value)], true));
	new IntersectionObserver(
		(entries) => {
			visible = entries[entries.length - 1].isIntersecting;
			if (visible) run();
		},
		{ rootMargin: '120px 0px' },
	).observe(svg);
	const calmQuery = matchMedia('(prefers-reduced-motion: reduce)');
	calmQuery.addEventListener('change', () => {
		touring = motion();
		restart(txs);
		draw();
		run();
	});
	if (motion()) {
		touring = true;
		restart(TOUR[0]);
	} else {
		touring = false;
		restart(txs);
	}
	draw();
	new ResizeObserver(() => {
		follow?.(tl.at(t).focus, true); // a new width keeps the camera where it was looking
	}).observe(svg);
	// handles for the capture scripts
	root.__keth = {
		seek(n, time) {
			touring = false;
			restart(n);
			t = time;
			for (let i = 0; i < 40; i++) draw(); // lets an eased camera settle
		},
		pause() {
			held = true;
			cancelAnimationFrame(frame);
			frame = 0;
			visible = false;
		},
	};
}
