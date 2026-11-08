/**
 * Shelf: prev/next buttons, ArrowLeft/ArrowRight inside the track, and mouse drag with momentum.
 * Touch and pen keep native scrolling. Without this the track is a plain overflow scroller.
 */

/** Share of the visible width one prev/next step scrolls. */
const PAGE = 0.8;
/** Pixels a press must travel before it counts as a drag (and swallows the click). */
const DRAG_MIN = 4;
/** Per-frame decay of the released velocity, and the speed at which the glide stops. */
const FRICTION = 0.94;
const REST = 0.4;

/** Starts the shelf on `root` (`[data-shelf]`); returns the teardown. */
export function initShelf(root: HTMLElement): () => void {
	const track = root.querySelector<HTMLElement>('[data-shelf-track]');
	const nav = root.querySelector<HTMLElement>('[data-shelf-nav]');
	const [prev, next] = nav ? [...nav.querySelectorAll<HTMLButtonElement>('button')] : [];
	if (!track || !nav || !prev || !next) return () => {};
	const reduced = matchMedia('(prefers-reduced-motion: reduce)');
	const ac = new AbortController();
	const on = { signal: ac.signal };
	root.dataset.ready = '';
	nav.hidden = false;

	const sync = () => {
		prev.disabled = track.scrollLeft <= 1;
		next.disabled = track.scrollLeft >= track.scrollWidth - track.clientWidth - 1;
	};
	const page = (dir: number) =>
		track.scrollBy({
			left: dir * track.clientWidth * PAGE,
			behavior: reduced.matches ? 'auto' : 'smooth',
		});
	prev.addEventListener('click', () => page(-1), on);
	next.addEventListener('click', () => page(1), on);
	track.addEventListener('scroll', sync, { passive: true, signal: ac.signal });
	const ro = new ResizeObserver(sync);
	ro.observe(track);
	sync();

	track.addEventListener(
		'keydown',
		(e) => {
			if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
			e.preventDefault();
			page(e.key === 'ArrowRight' ? 1 : -1);
		},
		on,
	);

	let down = false;
	let moved = 0;
	let startX = 0;
	let startLeft = 0;
	let lastX = 0;
	let velocity = 0;
	let raf = 0;
	track.addEventListener(
		'pointerdown',
		(e) => {
			if (e.pointerType !== 'mouse' || e.button !== 0) return;
			down = true;
			moved = 0;
			startX = lastX = e.clientX;
			startLeft = track.scrollLeft;
			velocity = 0;
			cancelAnimationFrame(raf);
		},
		on,
	);
	window.addEventListener(
		'pointermove',
		(e) => {
			if (!down) return;
			moved = Math.max(moved, Math.abs(e.clientX - startX));
			if (moved > DRAG_MIN) track.dataset.dragging = '';
			track.scrollLeft = startLeft - (e.clientX - startX);
			velocity = lastX - e.clientX;
			lastX = e.clientX;
		},
		on,
	);
	window.addEventListener(
		'pointerup',
		() => {
			if (!down) return;
			down = false;
			delete track.dataset.dragging;
			if (reduced.matches) return;
			const glide = () => {
				if (Math.abs(velocity) < REST) return;
				track.scrollLeft += velocity;
				velocity *= FRICTION;
				raf = requestAnimationFrame(glide);
			};
			glide();
		},
		on,
	);
	// A drag must not activate the item under the cursor.
	track.addEventListener(
		'click',
		(e) => {
			if (moved > DRAG_MIN) {
				e.preventDefault();
				e.stopPropagation();
				moved = 0;
			}
		},
		{ capture: true, signal: ac.signal },
	);

	return () => {
		ac.abort();
		ro.disconnect();
		cancelAnimationFrame(raf);
		delete root.dataset.ready;
		nav.hidden = true;
	};
}
