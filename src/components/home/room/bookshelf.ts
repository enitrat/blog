import { CABINET, ordered, rowName, rows, type ShelfFrame, type Slot } from './shelves';

export type { ShelfFrame } from './shelves';

type Point = { x: number; y: number; z: number };
type Project = (x: number, y: number, z: number) => Point;
type Section = { books: Slot[]; frame: ShelfFrame };
/** The four levels the visitor moves through, one arm each. A book is its own
 *  level rather than a field on the row, so `kind` alone answers every question
 *  the controls ask. */
type Location =
	| { kind: 'cabinet' }
	| { kind: 'row'; anchor: Slot }
	| { kind: 'book'; anchor: Slot; book: Slot };
/** The two levels that stand in front of a particular row. */
type Framed = Extract<Location, { anchor: Slot }>;

/** The room, cabinet, row, and book are successive levels of native navigation. */
export function mountBookshelf(host: HTMLElement) {
	const required = <T extends Element>(selector: string): T => {
		const element = host.querySelector<T>(selector);
		if (!element) throw new Error(`Missing bookshelf element: ${selector}`);
		return element;
	};
	const stage = required<HTMLElement>('[data-room-stage]');
	const canvas = required<HTMLCanvasElement>('.living-room__canvas');
	const targets = required<HTMLElement>('[data-book-targets]');
	const previous = required<HTMLButtonElement>('[data-shelf-previous]');
	const next = required<HTMLButtonElement>('[data-shelf-next]');
	const previousRow = required<HTMLButtonElement>('[data-row-previous]');
	const nextRow = required<HTMLButtonElement>('[data-row-next]');
	const leave = required<HTMLButtonElement>('[data-shelf-exit]');
	const leaveLabel = required<HTMLElement>('[data-shelf-exit-label]');
	const status = required<HTMLElement>('[data-shelf-status]');
	// One caption for the whole room, and one writer: the hotspots name objects
	// through `label()`, browsing names rows and books here.
	const caption = required<HTMLElement>('.living-room__caption');
	const catalog = required<HTMLDetailsElement>('[data-shelf-catalog]');
	const dialog = required<HTMLDialogElement>('[data-room-book]');
	const body = required<HTMLElement>('[data-open-book]');
	const entrance = required<HTMLAnchorElement>('[data-anchor="bookshelf"]');
	const events = new AbortController();
	const records = new Map(
		[...host.querySelectorAll<HTMLElement>('[data-book]')].map((record) => [
			record.dataset.book,
			record,
		]),
	);
	const links = new Map<string, HTMLElement>();
	const rowLinks = new Map<number, HTMLAnchorElement>();
	const rowAnchors = new Map(rows.map(({ row, books }) => [row, books[0]]));
	// This transparent band catches near misses between spines without zooming out.
	const gaps = document.createElement('div');
	gaps.className = 'room-library__gaps';
	gaps.setAttribute('aria-hidden', 'true');
	gaps.hidden = true;
	targets.append(gaps);
	let sections: Section[] = [];
	let location: Location | null = null;
	let disposed = false;
	// The renderer's listener doubles as the answer to "is the room live?".
	let frameChanged: ((frame: ShelfFrame | null) => void) | undefined;
	/** Which volume is under the pointer. The shelves know; the room draws it. */
	let pullChanged: ((isbn: string | null) => void) | undefined;
	// Which record is currently on loan to the dialog. A DOM bookkeeping detail:
	// the level is `location.kind`, never this.
	let mounted: { slot: Slot; source: HTMLElement; content: HTMLElement } | undefined;
	let closing: Animation | undefined;
	let reachTimer: ReturnType<typeof setTimeout> | undefined;
	let returningBook = false;
	const motion = matchMedia('(prefers-reduced-motion: reduce)');
	let returnFocus: HTMLElement | undefined;
	let lastHash = '';
	let goingBack = false;
	// A swipe synthesises a click on whatever it started over, and that click
	// follows within a few hundred milliseconds. A keyboard activation minutes
	// later must not still be suppressed, so this records a moment, not a mode.
	let swipedAt = -1;
	const fromSwipe = () => performance.now() - swipedAt < 500;
	let touch: { x: number; y: number } | undefined;

	/** Both of these are read back on every resize tick, and `status` is a live
	 *  region: writing the value it already holds would announce it again. */
	const write = (element: HTMLElement, value: string) => {
		if (element.textContent !== value) element.textContent = value;
	};

	function readLocation(): Location | null {
		const [name, anchorIsbn, bookIsbn, extra] = window.location.hash.slice(1).split('/');
		if (name !== 'bookshelf' || extra !== undefined) return null;
		if (!anchorIsbn) return bookIsbn === undefined ? { kind: 'cabinet' } : null;
		const anchor = ordered.find((slot) => slot.isbn === anchorIsbn);
		if (!anchor) return null;
		if (!bookIsbn) return { kind: 'row', anchor };
		const book = ordered.find((slot) => slot.isbn === bookIsbn);
		if (!book || book.row !== anchor.row) return null;
		return records.get(book.isbn)?.hasAttribute('data-noted')
			? { kind: 'book', anchor, book }
			: { kind: 'row', anchor: book };
	}

	function address(anchor: Slot, book?: Slot) {
		return `#bookshelf/${anchor.isbn}${book ? `/${book.isbn}` : ''}`;
	}

	function navigate(hash: string, replace = false) {
		if (hash === window.location.hash) return;
		const url = new URL(window.location.href);
		url.hash = hash;
		// Moving within this level preserves its parent history entry.
		if (replace) history.replaceState(history.state, '', url);
		else history.pushState({ ...history.state, roomBookFrom: window.location.hash }, '', url);
		sync();
	}

	function back() {
		if (!location || goingBack) return;
		const parent =
			location.kind === 'cabinet'
				? ''
				: location.kind === 'book'
					? address(location.anchor)
					: '#bookshelf';
		if (history.state?.roomBookFrom === parent) {
			goingBack = true;
			history.back();
		} else navigate(parent, true);
	}

	/* The record is moved, not copied: notes carry their own enhanced elements
	   (copy buttons on code blocks), and a clone would leave those dead. */
	function restoreBook() {
		if (!mounted) return;
		closing?.cancel();
		closing = undefined;
		dialog.removeAttribute('data-closing');
		mounted.source.append(mounted.content);
		if (frameChanged) {
			returnFocus = links.get(mounted.slot.isbn);
			returningBook = true;
		} else {
			// Without a renderer the list is the bookshelf. Open it back to where
			// the reader was: focus inside a closed <details> is dropped silently.
			catalog.open = true;
			if (mounted.source instanceof HTMLDetailsElement) mounted.source.open = true;
			returnFocus = mounted.source.querySelector('summary') ?? undefined;
		}
		mounted = undefined;
		if (dialog.open) dialog.close();
	}

	function showBook(slot: Slot) {
		const spatial = !!frameChanged && !motion.matches;
		if (spatial && (!dialog.open || !dialog.hasAttribute('data-spatial'))) {
			stage.scrollIntoView({ block: 'center', behavior: 'instant' });
			body.style.opacity = '0';
			body.inert = true;
		}
		dialog.toggleAttribute('data-spatial', spatial);
		if (!spatial) {
			body.removeAttribute('style');
			body.inert = false;
		}
		if (mounted?.slot.isbn === slot.isbn) {
			// Reduced motion arrived or left while this book was open. Only the
			// modality has to change; what is mounted is already right.
			if (dialog.open && dialog.matches(':modal') === spatial) {
				dialog.close();
				open(spatial);
			}
			return;
		}
		restoreBook();
		const source = records.get(slot.isbn);
		const content = source?.querySelector<HTMLElement>('[data-book-content]');
		if (!source?.hasAttribute('data-noted') || !content) return;
		mounted = { slot, source, content };
		dialog.style.setProperty('--binding', source.style.getPropertyValue('--binding'));
		dialog.toggleAttribute('data-noted', source.hasAttribute('data-noted'));
		dialog.setAttribute('aria-label', `${source.dataset.title}, ${source.dataset.author}`);
		body.replaceChildren(content);
		open(spatial);
		dialog.scrollTop = 0;
	}

	/** A book standing open in the room is an object on a shelf, not a sheet laid
	 *  over the page: the reader keeps scrolling the page behind it, and a click
	 *  on clear background reaches the canvas and steps back a level. A modal
	 *  dialog makes the rest of the document inert and takes both away, so only
	 *  the conventional record — reduced motion, or no renderer — is modal.
	 *  Escape is the browser's `cancel` there, and the room's own key handler here. */
	function open(spatial: boolean) {
		if (spatial) dialog.show();
		else dialog.showModal();
	}

	function closeBook() {
		if (closing) return;
		const style = getComputedStyle(dialog);
		const animation = dialog.animate(
			[
				{ opacity: style.opacity, transform: style.transform },
				{
					opacity: 0,
					transform:
						motion.matches || dialog.hasAttribute('data-spatial')
							? 'none'
							: 'translateY(8px) scale(0.98)',
				},
			],
			{
				duration: motion.matches || dialog.hasAttribute('data-spatial') ? 100 : 180,
				easing: 'cubic-bezier(0.23, 1, 0.32, 1)',
				fill: 'forwards',
			},
		);
		closing = animation;
		dialog.setAttribute('data-closing', '');
		void animation.finished.then(
			() => {
				if (closing === animation) back();
			},
			() => {},
		);
	}

	function captionFor(slot?: Slot) {
		const record = slot && records.get(slot.isbn);
		if (!record) return 'Choose a row to look closer.\nGlowing books hold notes.';
		return `${record.dataset.title}\n${record.dataset.author} · ${record.hasAttribute('data-noted') ? `Read notes${record.hasAttribute('data-reading') ? ' · Reading now' : ''}` : record.dataset.facts}`;
	}

	const describe = (slot?: Slot) => write(caption, captionFor(slot));

	function layout() {
		const width = Math.max(
			0.06,
			Math.min(0.48, (canvas.clientWidth * 0.3) / Math.max(1, canvas.clientHeight)),
		);
		sections = [];
		for (const { books } of rows) {
			let start = 0;
			while (start < books.length) {
				const first = books[start];
				const left = first.x - first.width / 2 - 0.012;
				let end = start + 1;
				while (end < books.length && books[end].x + books[end].width / 2 <= left + width - 0.012)
					end++;
				const visible = books.slice(start, end);
				const right = visible[visible.length - 1];
				sections.push({
					books: visible,
					frame: {
						view: 'row',
						x: (first.x - first.width / 2 + right.x + right.width / 2) / 2,
						y: first.y + Math.max(...visible.map((slot) => slot.height)) / 2,
						z: first.z,
						width,
						height: 0.32,
					},
				});
				if (end === books.length) break;
				start = end - start > 2 ? end - 1 : end;
			}
		}
	}

	/** Which run of books is on screen. Derived, never stored: `layout()` rebuilds
	 *  `sections` on every resize and this stays correct without being told. */
	function currentSection(place: Framed) {
		const desired = place.kind === 'book' ? place.book : place.anchor;
		const exact = sections.findIndex(
			(section) =>
				section.books[0].isbn === place.anchor.isbn &&
				section.books.some((slot) => slot.isbn === desired.isbn),
		);
		if (exact >= 0) return exact;
		const holding = sections.findIndex((section) =>
			section.books.some((slot) => slot.isbn === desired.isbn),
		);
		// Every book in `rows` falls in exactly one section. A miss means the
		// manifest and `layout()` disagree, which must not be quietly framed as
		// whichever row happens to be first.
		if (holding < 0) throw new Error(`No shelf section holds ${desired.isbn}`);
		return holding;
	}

	/** Where the camera should be looking at this level. */
	function frameFor(place: Location | null): ShelfFrame | null {
		if (!place) return null;
		return place.kind === 'cabinet' ? CABINET : sections[currentSection(place)].frame;
	}

	/** What the chrome should say at this level, decided before anything is
	 *  written, so the writing below has no branches of its own. */
	function chromeFor(place: Location | null) {
		if (!place)
			return { level: null, exit: 'Back to the room', status: '', caption: '', pan: null, row: -1 };
		if (place.kind === 'cabinet')
			return {
				level: 'cabinet',
				exit: 'Back to the room',
				status: 'Bookshelf',
				caption: captionFor(),
				pan: null,
				row: -1,
			};
		const index = currentSection(place);
		return {
			level: 'row',
			exit: 'Back to the bookshelf',
			status: rowName(place.anchor.row),
			caption: captionFor(place.kind === 'book' ? place.book : place.anchor),
			pan: {
				previous: sections[index - 1]?.books[0].row === place.anchor.row,
				next: sections[index + 1]?.books[0].row === place.anchor.row,
			},
			row: rows.findIndex(({ row }) => row === place.anchor.row),
		};
	}

	function applyChrome(chrome: ReturnType<typeof chromeFor>) {
		host.toggleAttribute('data-browsing', chrome.level !== null);
		targets.hidden = chrome.level === null;
		// CSS shows the row and spine targets per level; a book still frames its
		// row, so the camera and the controls stay at row level behind the dialog.
		if (chrome.level) host.dataset.shelfLevel = chrome.level;
		else host.removeAttribute('data-shelf-level');
		write(leaveLabel, chrome.exit);
		write(status, chrome.status);
		write(caption, chrome.caption);
		previous.disabled = !chrome.pan?.previous;
		next.disabled = !chrome.pan?.next;
		// A row that fits in one frame has nowhere to pan to.
		previous.hidden = !chrome.pan?.previous && !chrome.pan?.next;
		next.hidden = previous.hidden;
		previousRow.disabled = chrome.row <= 0;
		nextRow.disabled = chrome.row < 0 || chrome.row === rows.length - 1;
	}

	/** Focus follows the level, but has to wait for the targets to be placed,
	 *  so it is only ever recorded here and applied by `place()`. */
	function parkFocus(before: Location | null, changed: boolean) {
		if (!location) {
			if (before) returnFocus = entrance;
			return;
		}
		if (!changed) return;
		if (location.kind === 'cabinet')
			returnFocus = rowLinks.get(
				before && before.kind !== 'cabinet' ? before.anchor.row : rows[0]?.row,
			);
		else if (location.kind === 'row') returnFocus ??= links.get(location.anchor.isbn);
	}

	function applyFocus() {
		if (!returnFocus || mounted || host.hasAttribute('data-book-moving')) return;
		if (returnFocus.hidden || returnFocus.closest('[hidden]')) return;
		returnFocus.focus({ preventScroll: true });
		returnFocus = undefined;
		if (returningBook) {
			returningBook = false;
			pullChanged?.(null);
		}
	}

	function sync() {
		if (disposed) return;
		clearTimeout(reachTimer);
		const before = location;
		location = readLocation();
		if (location?.kind === 'row' && window.location.hash.split('/').length === 3) {
			const url = new URL(window.location.href);
			url.hash = address(location.anchor);
			history.replaceState(history.state, '', url);
		}
		const changed = lastHash !== window.location.hash;
		lastHash = window.location.hash;
		goingBack = false;
		// Without a renderer the shelves are just the list on the page: no camera
		// to move, no chrome to show, and nothing to be stranded inside. The
		// dialog is the one thing that works either way.
		const shown = frameChanged ? location : null;
		// Whether focus is about to be left on a target this level hides. Read
		// before anything moves; acted on once the chrome is actually on screen.
		const stranded =
			changed &&
			shown !== null &&
			(targets.contains(document.activeElement) || document.activeElement === entrance);
		if (location?.kind === 'book') {
			showBook(location.book);
		} else {
			if (before?.kind === 'book' && !mounted && frameChanged)
				returnFocus = links.get(before.book.isbn);
			restoreBook();
		}
		// `place()` reads this when the camera settles, so it is set first.
		if (shown && shown.kind !== 'cabinet') rowAnchors.set(shown.anchor.row, shown.anchor);
		applyChrome(chromeFor(shown));
		// Arriving at a row reaches for nothing yet; arriving at a book took it out.
		pullChanged?.(shown?.kind === 'book' ? shown.book.isbn : null);
		if (frameChanged) {
			frameChanged(frameFor(shown));
			// The way out is display:none until the camera publishes its view, so
			// parking focus there any earlier puts it on the floor instead.
			if (stranded && !mounted) leave.focus({ preventScroll: true });
			parkFocus(before, changed);
		} else applyFocus();
	}

	function enter() {
		if (!location) navigate('#bookshelf');
	}

	function focusRow(row: number) {
		const anchor = rowAnchors.get(row);
		if (!anchor || location?.kind === 'book') return;
		returnFocus = links.get(anchor.isbn);
		navigate(address(anchor), location?.kind === 'row');
	}

	function moveRow(delta: number) {
		if (location?.kind !== 'row') return;
		const anchor = location.anchor;
		const index = rows.findIndex(({ row }) => row === anchor.row);
		const row = rows[index + delta];
		if (row) focusRow(row.row);
	}

	function move(delta: number) {
		if (location?.kind !== 'row') return;
		const section = sections[currentSection(location) + delta];
		if (!section || section.books[0].row !== location.anchor.row) return;
		returnFocus = links.get(section.books[0].isbn);
		navigate(address(section.books[0]), true);
	}

	function plainClick(event: MouseEvent) {
		return (
			!event.metaKey && !event.ctrlKey && !event.shiftKey && !event.altKey && event.button === 0
		);
	}

	for (const { row, books } of rows) {
		const link = document.createElement('a');
		link.className = 'room-library__row';
		link.href = address(books[0]);
		link.hidden = true;
		link.setAttribute(
			'aria-label',
			`Look closer at the ${rowName(row).toLowerCase()}, ${books.length} books`,
		);
		const describeRow = () => {
			const notes = books.filter((slot) =>
				records.get(slot.isbn)?.hasAttribute('data-noted'),
			).length;
			write(
				caption,
				`${rowName(row)} · ${books.length} books\nLook closer${notes ? ` · ${notes} with notes inside` : ''}`,
			);
		};
		link.addEventListener('pointerenter', describeRow, { signal: events.signal });
		link.addEventListener('focus', describeRow, { signal: events.signal });
		link.addEventListener(
			'click',
			(event) => {
				if (!plainClick(event)) return;
				event.preventDefault();
				if (location?.kind === 'cabinet') focusRow(row);
			},
			{ signal: events.signal },
		);
		rowLinks.set(row, link);
		targets.append(link);
	}

	for (const slot of ordered) {
		const record = records.get(slot.isbn);
		if (!record) continue;
		const noted = record.hasAttribute('data-noted');
		const link: HTMLElement = document.createElement(noted ? 'a' : 'span');
		link.className = 'room-library__spine';
		if (link instanceof HTMLAnchorElement) {
			link.href = address(slot, slot);
			link.setAttribute('aria-haspopup', 'dialog');
		} else {
			link.tabIndex = 0;
			link.setAttribute('role', 'img');
		}
		link.hidden = true;
		link.setAttribute(
			'aria-label',
			`${record.dataset.title}, ${record.dataset.author}. ${noted ? `Read notes. ${record.dataset.facts}` : record.dataset.facts}`,
		);
		link.toggleAttribute('data-noted', record.hasAttribute('data-noted'));
		link.toggleAttribute('data-reading', record.hasAttribute('data-reading'));
		const reach = () => {
			clearTimeout(reachTimer);
			describe(slot);
			pullChanged?.(noted ? slot.isbn : null);
		};
		// A volume you are no longer reaching for goes back, unless it is the one
		// you already took down: an open record keeps its book out of the row.
		const release = () => {
			clearTimeout(reachTimer);
			pullChanged?.(location?.kind === 'book' ? location.book.isbn : null);
		};
		link.addEventListener(
			'pointerenter',
			(event) => {
				describe(slot);
				clearTimeout(reachTimer);
				if (event.pointerType === 'mouse') reachTimer = setTimeout(reach, 100);
			},
			{ signal: events.signal },
		);
		link.addEventListener('focus', reach, { signal: events.signal });
		link.addEventListener('pointerleave', release, { signal: events.signal });
		link.addEventListener('blur', release, { signal: events.signal });
		link.addEventListener(
			'click',
			(event) => {
				if (!plainClick(event)) return;
				event.preventDefault();
				if (fromSwipe() || location?.kind !== 'row') return;
				if (noted) navigate(address(location.anchor, slot));
				else {
					link.focus({ preventScroll: true });
					reach();
				}
			},
			{ signal: events.signal },
		);
		links.set(slot.isbn, link);
		targets.append(link);
	}

	entrance.addEventListener(
		'click',
		(event) => {
			// A room that is merely still loading is a room: browsing waits for it.
			// Only a device that cannot render one falls through to the anchor,
			// which jumps to the list rather than to a mode with no way out.
			if (!plainClick(event)) return;
			if (!frameChanged && host.dataset.canRender === undefined) return;
			event.preventDefault();
			enter();
		},
		{ signal: events.signal },
	);
	previous.addEventListener('click', () => move(-1), { signal: events.signal });
	next.addEventListener('click', () => move(1), { signal: events.signal });
	previousRow.addEventListener('click', () => moveRow(-1), { signal: events.signal });
	nextRow.addEventListener('click', () => moveRow(1), { signal: events.signal });
	leave.addEventListener('click', back, { signal: events.signal });
	dialog.addEventListener(
		'cancel',
		(event) => {
			event.preventDefault();
			closeBook();
		},
		{ signal: events.signal },
	);
	dialog.querySelector('form')?.addEventListener(
		'submit',
		(event) => {
			event.preventDefault();
			closeBook();
		},
		{ signal: events.signal },
	);
	// A non-modal book does not contain the tab order, so focus can legitimately
	// be anywhere on the page while one stands open. Escape has to reach it from
	// there too; the room's own handler below runs first and marks the event.
	document.addEventListener(
		'keydown',
		(event) => {
			if (event.defaultPrevented || event.key !== 'Escape') return;
			if (location?.kind !== 'book' || dialog.matches(':modal')) return;
			event.preventDefault();
			back();
		},
		{ signal: events.signal },
	);
	host.addEventListener(
		'keydown',
		(event) => {
			if (event.target instanceof Node && catalog.contains(event.target)) return;
			if (location?.kind === 'book' && !dialog.matches(':modal') && event.key === 'Escape') {
				event.preventDefault();
				back();
				return;
			}
			if (!location || location.kind === 'book') return;
			if (event.altKey || event.metaKey || event.ctrlKey) return;
			if (event.key === 'Escape') {
				event.preventDefault();
				event.stopPropagation();
				back();
				return;
			}
			if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
				event.preventDefault();
				const delta = event.key === 'ArrowDown' ? 1 : -1;
				if (location.kind === 'row') moveRow(delta);
				else {
					const index = rows.findIndex(({ row }) => rowLinks.get(row) === document.activeElement);
					const row = rows[index < 0 ? 0 : Math.max(0, Math.min(rows.length - 1, index + delta))];
					if (row) rowLinks.get(row.row)?.focus({ preventScroll: true });
				}
			}
			if (location.kind !== 'row' || (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft'))
				return;
			event.preventDefault();
			const focused = [...links].find(([, link]) => link === document.activeElement);
			const delta = event.key === 'ArrowRight' ? 1 : -1;
			if (!focused) {
				move(delta);
				return;
			}
			const adjacent = ordered[ordered.findIndex((slot) => slot.isbn === focused[0]) + delta];
			if (!adjacent || adjacent.row !== location.anchor.row) return;
			const link = links.get(adjacent.isbn);
			if (link && !link.hidden) link.focus({ preventScroll: true });
			else {
				returnFocus = link;
				navigate(address(adjacent), true);
			}
		},
		{ signal: events.signal },
	);
	stage.addEventListener(
		'pointerdown',
		(event) => {
			swipedAt = -1;
			if (location?.kind === 'row' && event.pointerType === 'touch')
				touch = { x: event.clientX, y: event.clientY };
		},
		{ signal: events.signal },
	);
	stage.addEventListener(
		'pointerup',
		(event) => {
			if (!touch) return;
			const dx = event.clientX - touch.x;
			const dy = event.clientY - touch.y;
			touch = undefined;
			if (Math.abs(dx) > 45 && Math.abs(dx) > Math.abs(dy) * 1.5) {
				swipedAt = performance.now();
				move(dx < 0 ? 1 : -1);
			}
		},
		{ signal: events.signal },
	);
	stage.addEventListener(
		'pointercancel',
		() => {
			touch = undefined;
		},
		{ signal: events.signal },
	);
	const historyChanged = () => {
		if (window.location.hash !== lastHash) sync();
	};
	window.addEventListener('popstate', historyChanged, { signal: events.signal });
	window.addEventListener('hashchange', historyChanged, { signal: events.signal });
	const resize = new ResizeObserver(() => {
		layout();
		sync();
	});
	resize.observe(canvas);
	motion.addEventListener('change', sync, { signal: events.signal });
	layout();
	sync();

	function placeBox(
		element: HTMLElement,
		project: Project,
		box: { left: number; bottom: number; right: number; top: number; z: number },
		clip = false,
	) {
		const corners = [
			project(box.left, box.top, box.z),
			project(box.right, box.top, box.z),
			project(box.left, box.bottom, box.z),
			project(box.right, box.bottom, box.z),
		];
		const a = {
			x: Math.min(...corners.map((p) => p.x)),
			y: Math.max(...corners.map((p) => p.y)),
			z: Math.min(...corners.map((p) => p.z)),
		};
		const b = {
			x: Math.max(...corners.map((p) => p.x)),
			y: Math.min(...corners.map((p) => p.y)),
			z: Math.max(...corners.map((p) => p.z)),
		};
		// `hidden` here means one thing only: no valid position on screen. Which
		// level a target belongs to is CSS's business, via data-shelf-level.
		element.hidden =
			a.z < -1 ||
			a.z > 1 ||
			b.z < -1 ||
			b.z > 1 ||
			(clip
				? b.x < -1 || a.x > 1 || b.y > 1 || a.y < -1
				: a.x < -1 || b.x > 1 || a.y > 1 || b.y < -1);
		element.style.left = `${(a.x + 1) * 50}%`;
		element.style.top = `${(1 - a.y) * 50}%`;
		element.style.width = `${(b.x - a.x) * 50}%`;
		element.style.height = `${(a.y - b.y) * 50}%`;
	}

	return {
		enter,
		/** Align live HTML with the exposed Blender pages in viewport coordinates. */
		positionReader(
			rect: { left: number; top: number; width: number; height: number; controlsTop: number },
			opacity: number,
		) {
			if (!dialog.hasAttribute('data-spatial')) return;
			body.style.left = `${rect.left}px`;
			body.style.top = `${rect.top}px`;
			body.style.width = `${rect.width}px`;
			body.style.height = `${rect.height}px`;
			body.style.opacity = String(opacity);
			body.inert = opacity < 1;
			dialog.style.setProperty('--reader-controls-top', `${rect.controlsTop}px`);
		},
		background() {
			if (frameChanged && !fromSwipe() && location) back();
		},
		get openBook() {
			return location?.kind === 'book' ? location.book.isbn : null;
		},
		get active() {
			return location !== null;
		},
		/** The room's own hotspots name themselves through here, so the caption
		 *  has a single owner: while browsing, the shelves keep it. */
		label(text: string) {
			if (!location) write(caption, text);
		},
		/** The room comes to collect the volume being reached for. */
		pulls(listener: (isbn: string | null) => void) {
			pullChanged = listener;
		},
		/** Only annotated volumes invite opening and receive the scene's warm glow. */
		hasNotes(isbn: string) {
			return records.get(isbn)?.hasAttribute('data-noted') ?? false;
		},
		/** Reading status supplies the cloth bookmark independently of authored notes. */
		isReading(isbn: string) {
			return records.get(isbn)?.hasAttribute('data-reading') ?? false;
		},
		connect(listener: (frame: ShelfFrame | null) => void) {
			frameChanged = listener;
			sync();
		},
		fallback() {
			frameChanged = undefined;
			// The room proved it cannot render one. The entrance stops intercepting
			// its own click, so it falls through to the list rather than to a level
			// the URL is already on and `enter()` therefore refuses to re-enter.
			delete host.dataset.canRender;
			sync();
		},
		/** Project every target once, when the camera has arrived. The shelves are
		 *  a function of where the camera is going, not of the frames it draws on
		 *  the way: the settled camera stays still and parallax is off while
		 *  browsing, so re-running this per frame recomputed identical numbers. */
		place(project: Project) {
			for (const { row, books } of rows) {
				const link = rowLinks.get(row);
				if (!link) continue;
				// The target is the run of books, not the whole board, so the
				// centred dot lands on the row it stands for.
				const first = books[0];
				const last = books[books.length - 1];
				placeBox(link, project, {
					left: first.x - first.width / 2 - 0.02,
					bottom: first.y - 0.012,
					right: last.x + last.width / 2 + 0.02,
					top: first.y + 0.3,
					z: first.z,
				});
				link.href = address(rowAnchors.get(row) ?? books[0]);
			}
			// Spines outside the framed row have no position to hold, and would
			// otherwise keep the one they had before the camera moved.
			const framed = location && location.kind !== 'cabinet' ? location.anchor.row : -1;
			for (const slot of ordered)
				if (slot.row !== framed) {
					const link = links.get(slot.isbn);
					if (link) link.hidden = true;
				}
			if (location && location.kind !== 'cabinet') {
				const books = ordered.filter((slot) => slot.row === framed);
				const first = books[0];
				const last = books[books.length - 1];
				placeBox(
					gaps,
					project,
					{
						left: first.x - first.width / 2 - 0.004,
						bottom: first.y - 0.004,
						right: last.x + last.width / 2 + 0.004,
						top: first.y + Math.max(...books.map((slot) => slot.height)) + 0.004,
						z: first.z,
					},
					true,
				);
				for (const slot of books) {
					const link = links.get(slot.isbn);
					if (!link) continue;
					placeBox(link, project, {
						left: slot.x - slot.width / 2,
						bottom: slot.y,
						right: slot.x + slot.width / 2,
						top: slot.y + slot.height,
						z: slot.z,
					});
					if (link instanceof HTMLAnchorElement) link.href = address(location.anchor, slot);
				}
			}
			applyFocus();
		},
		dispose() {
			disposed = true;
			clearTimeout(reachTimer);
			events.abort();
			resize.disconnect();
			restoreBook();
			targets.replaceChildren();
		},
	};
}

export type Bookshelf = ReturnType<typeof mountBookshelf>;
