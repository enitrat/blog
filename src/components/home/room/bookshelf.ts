import slots from '../../../assets/room/book-slots.json';

type Slot = (typeof slots)[number];
export type ShelfFrame = {
	view: 'cabinet' | 'row';
	x: number;
	y: number;
	z: number;
	width: number;
	height: number;
};
type Point = { x: number; y: number; z: number };
type Project = (x: number, y: number, z: number) => Point;
type Section = { books: Slot[]; frame: ShelfFrame };
type Location = { kind: 'cabinet' } | { kind: 'row'; anchor: Slot; book: Slot | undefined };
const CABINET: ShelfFrame = {
	view: 'cabinet',
	x: -1.12,
	y: 0.87,
	z: -1.068,
	width: 1.38,
	height: 1.94,
};
const ROW_NAMES = ['Bottom row', 'Middle row', 'Top row'];

/** The room, cabinet, row, and book are successive levels of native navigation. */
export function mountBookshelf(host: HTMLElement) {
	const required = <T extends Element>(selector: string): T => {
		const element = host.querySelector<T>(selector);
		if (!element) throw new Error(`Missing bookshelf element: ${selector}`);
		return element;
	};
	const stage = required<HTMLElement>('[data-room-stage]');
	const targets = required<HTMLElement>('[data-book-targets]');
	const previous = required<HTMLButtonElement>('[data-shelf-previous]');
	const next = required<HTMLButtonElement>('[data-shelf-next]');
	const previousRow = required<HTMLButtonElement>('[data-row-previous]');
	const nextRow = required<HTMLButtonElement>('[data-row-next]');
	const leave = required<HTMLButtonElement>('[data-shelf-exit]');
	const leaveLabel = required<HTMLElement>('[data-shelf-exit-label]');
	const status = required<HTMLElement>('[data-shelf-status]');
	// One caption for the whole room: the hotspots name objects with it, and
	// browsing names rows and books with it.
	const caption = required<HTMLElement>('.living-room__caption');
	const catalog = required<HTMLDetailsElement>('[data-shelf-catalog]');
	const dialog = required<HTMLDialogElement>('[data-room-book]');
	const body = required<HTMLElement>('[data-open-book]');
	const entrance = required<HTMLAnchorElement>('[data-anchor="bookshelf"]');
	const events = new AbortController();
	const ordered = [...slots].sort((a, b) => b.row - a.row || a.x - b.x);
	const rows = [2, 1, 0]
		.map((row) => ({ row, books: ordered.filter((slot) => slot.row === row) }))
		.filter(({ books }) => books.length > 0);
	const records = new Map(
		[...host.querySelectorAll<HTMLDetailsElement>('[data-book]')].map((record) => [
			record.dataset.book,
			record,
		]),
	);
	const links = new Map<string, HTMLAnchorElement>();
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
	let sectionIndex = 0;
	let live = false;
	let disposed = false;
	let openedFallback = false;
	let frameChanged: ((frame: ShelfFrame | null) => void) | undefined;
	let opened: { slot: Slot; source: HTMLDetailsElement; content: HTMLElement } | undefined;
	let returnFocus: HTMLElement | undefined;
	let lastHash = '';
	let ready = false;
	let goingBack = false;
	let swiped = false;
	let touch: { x: number; y: number } | undefined;

	function readLocation(): Location | null {
		const [name, anchorIsbn, bookIsbn, extra] = window.location.hash.slice(1).split('/');
		if (name !== 'bookshelf' || extra !== undefined) return null;
		if (!anchorIsbn) return bookIsbn === undefined ? { kind: 'cabinet' } : null;
		const anchor = ordered.find((slot) => slot.isbn === anchorIsbn);
		const book = bookIsbn ? ordered.find((slot) => slot.isbn === bookIsbn) : undefined;
		return anchor && (!bookIsbn || book?.row === anchor.row) ? { kind: 'row', anchor, book } : null;
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
			location.kind === 'cabinet' ? '' : location.book ? address(location.anchor) : '#bookshelf';
		if (history.state?.roomBookFrom === parent) {
			goingBack = true;
			history.back();
		} else navigate(parent, true);
	}

	function restoreBook() {
		if (!opened) return;
		opened.source.append(opened.content);
		returnFocus = live
			? links.get(opened.slot.isbn)
			: (opened.source.querySelector('summary') ?? undefined);
		opened = undefined;
		if (dialog.open) dialog.close();
	}

	function showBook(slot: Slot) {
		if (opened?.slot.isbn === slot.isbn) return;
		restoreBook();
		const source = records.get(slot.isbn);
		const content = source?.querySelector<HTMLElement>('[data-book-content]');
		if (!source || !content) return;
		opened = { slot, source, content };
		dialog.style.setProperty('--binding', source.style.getPropertyValue('--binding'));
		dialog.toggleAttribute('data-noted', source.hasAttribute('data-noted'));
		dialog.setAttribute('aria-label', `${source.dataset.title}, ${source.dataset.author}`);
		body.replaceChildren(content);
		dialog.showModal();
		dialog.scrollTop = 0;
	}

	function describe(slot?: Slot) {
		const record = slot && records.get(slot.isbn);
		caption.textContent = record
			? `${record.dataset.title} · ${record.dataset.author} · ${record.hasAttribute('data-noted') ? 'Read notes' : 'View record'}`
			: 'Choose a row to look closer.';
	}

	function layout() {
		const narrowest = Math.min(...slots.map((slot) => slot.width));
		const width = Math.max(0.06, Math.min(0.48, (stage.clientWidth * narrowest) / 48));
		host.style.setProperty(
			'--shelf-stage-height',
			`${Math.ceil((stage.clientWidth * 0.3) / width)}px`,
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
						height: 0.3,
					},
				});
				if (end === books.length) break;
				start = end - start > 2 ? end - 1 : end;
			}
		}
	}

	function sync() {
		if (disposed) return;
		const before = location;
		location = readLocation();
		const changed = lastHash !== window.location.hash;
		lastHash = window.location.hash;
		goingBack = false;
		ready = false;
		host.toggleAttribute('data-browsing', location !== null);
		if (location) host.dataset.shelfLevel = location.kind;
		else delete host.dataset.shelfLevel;
		targets.hidden = !location || !live;
		if (
			location &&
			(targets.contains(document.activeElement) || document.activeElement === entrance)
		) {
			if (!changed && document.activeElement instanceof HTMLElement) {
				returnFocus ??= document.activeElement;
			}
			// Park focus on the visible way out rather than on a live region.
			leave.focus({ preventScroll: true });
		}
		targets.inert = true;
		const focusedRow = location?.kind === 'row' ? location : null;
		for (const button of [previous, next, previousRow, nextRow])
			button.hidden = !focusedRow || !live;
		leaveLabel.textContent = focusedRow ? 'Back to the bookshelf' : 'Back to the room';
		if (!location) {
			caption.textContent = '';
			restoreBook();
			frameChanged?.(null);
			if (before) returnFocus = entrance;
		} else {
			if (!live && !catalog.open) {
				catalog.open = true;
				openedFallback = true;
			}
			if (live && openedFallback) {
				catalog.open = false;
				openedFallback = false;
			}
			if (location.kind === 'cabinet') {
				restoreBook();
				status.textContent = 'Bookshelf';
				describe();
				frameChanged?.(CABINET);
				if (changed)
					returnFocus = rowLinks.get(before?.kind === 'row' ? before.anchor.row : rows[0]?.row);
			} else {
				const { anchor, book } = location;
				const desired = book ?? anchor;
				const exact = sections.findIndex(
					(section) =>
						section.books[0].isbn === anchor.isbn &&
						section.books.some((slot) => slot.isbn === desired.isbn),
				);
				sectionIndex =
					exact >= 0
						? exact
						: Math.max(
								0,
								sections.findIndex((section) =>
									section.books.some((slot) => slot.isbn === desired.isbn),
								),
							);
				const section = sections[sectionIndex];
				rowAnchors.set(anchor.row, anchor);
				previous.disabled = sections[sectionIndex - 1]?.books[0].row !== anchor.row;
				next.disabled = sections[sectionIndex + 1]?.books[0].row !== anchor.row;
				previous.hidden = !live || (previous.disabled && next.disabled);
				next.hidden = previous.hidden;
				const rowIndex = rows.findIndex(({ row }) => row === anchor.row);
				previousRow.disabled = rowIndex === 0;
				nextRow.disabled = rowIndex === rows.length - 1;
				status.textContent = ROW_NAMES[anchor.row];
				frameChanged?.(section.frame);
				if (book) showBook(book);
				else restoreBook();
				describe(book ?? anchor);
				if (changed && !book) returnFocus ??= links.get(anchor.isbn);
			}
		}
		if (!live && returnFocus && !opened) {
			if (!targets.contains(returnFocus)) returnFocus.focus({ preventScroll: true });
			returnFocus = undefined;
		}
	}

	function enter() {
		if (!location) navigate('#bookshelf');
	}

	function focusRow(row: number) {
		const anchor = rowAnchors.get(row);
		if (!anchor || opened) return;
		returnFocus = links.get(anchor.isbn);
		navigate(address(anchor), location?.kind === 'row');
	}

	function moveRow(delta: number) {
		if (location?.kind !== 'row' || opened) return;
		const index = rows.findIndex(
			({ row }) => location?.kind === 'row' && row === location.anchor.row,
		);
		const row = rows[index + delta];
		if (row) focusRow(row.row);
	}

	function move(delta: number) {
		const section = sections[sectionIndex + delta];
		if (location?.kind !== 'row' || opened || section?.books[0].row !== location.anchor.row) return;
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
			`Look closer at the ${ROW_NAMES[row].toLowerCase()}, ${books.length} books`,
		);
		const describeRow = () => {
			caption.textContent = `${ROW_NAMES[row]} · ${books.length} books · Look closer`;
		};
		link.addEventListener('pointerenter', describeRow, { signal: events.signal });
		link.addEventListener('focus', describeRow, { signal: events.signal });
		link.addEventListener(
			'click',
			(event) => {
				if (!plainClick(event)) return;
				event.preventDefault();
				if (ready && location?.kind === 'cabinet') focusRow(row);
			},
			{ signal: events.signal },
		);
		rowLinks.set(row, link);
		targets.append(link);
	}

	for (const slot of ordered) {
		const record = records.get(slot.isbn);
		if (!record) continue;
		const link = document.createElement('a');
		link.className = 'room-library__spine';
		link.href = address(slot, slot);
		link.hidden = true;
		link.setAttribute('aria-haspopup', 'dialog');
		link.setAttribute(
			'aria-label',
			`${record.dataset.title}, ${record.dataset.author}. ${record.hasAttribute('data-noted') ? 'Read notes' : 'View record'}`,
		);
		link.toggleAttribute('data-noted', record.hasAttribute('data-noted'));
		link.addEventListener('pointerenter', () => describe(slot), { signal: events.signal });
		link.addEventListener('focus', () => describe(slot), { signal: events.signal });
		link.addEventListener(
			'click',
			(event) => {
				if (!plainClick(event)) return;
				event.preventDefault();
				if (!swiped && ready && location?.kind === 'row') navigate(address(location.anchor, slot));
			},
			{ signal: events.signal },
		);
		links.set(slot.isbn, link);
		targets.append(link);
	}

	entrance.addEventListener(
		'click',
		(event) => {
			if (!plainClick(event)) return;
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
			back();
		},
		{ signal: events.signal },
	);
	dialog.querySelector('form')?.addEventListener(
		'submit',
		(event) => {
			event.preventDefault();
			back();
		},
		{ signal: events.signal },
	);
	host.addEventListener(
		'keydown',
		(event) => {
			if (event.target instanceof Node && catalog.contains(event.target)) return;
			if (!location || opened || event.altKey || event.metaKey || event.ctrlKey) return;
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
			swiped = false;
			if (location?.kind === 'row' && ready && event.pointerType === 'touch')
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
				swiped = true;
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
	resize.observe(stage);
	layout();
	sync();

	function place(
		element: HTMLElement,
		project: Project,
		left: number,
		bottom: number,
		right: number,
		top: number,
		z: number,
		clip = false,
	) {
		const a = project(left, top, z);
		const b = project(right, bottom, z);
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
		background() {
			if (live && ready && !swiped && !opened) back();
		},
		get active() {
			return location !== null;
		},
		connect(listener: (frame: ShelfFrame | null) => void) {
			frameChanged = listener;
			live = true;
			sync();
		},
		fallback() {
			live = false;
			frameChanged = undefined;
			sync();
		},
		draw(project: Project, settled: boolean) {
			ready = settled;
			targets.inert = !settled || !!opened;
			for (const { row, books } of rows) {
				const link = rowLinks.get(row);
				if (!link) continue;
				link.hidden = !live || !settled || location?.kind !== 'cabinet';
				if (link.hidden) continue;
				// The target is the run of books, not the whole board, so the
				// centred dot lands on the row it stands for.
				const first = books[0];
				const last = books[books.length - 1];
				place(
					link,
					project,
					first.x - first.width / 2 - 0.02,
					first.y - 0.012,
					last.x + last.width / 2 + 0.02,
					first.y + 0.3,
					first.z,
				);
				link.href = address(rowAnchors.get(row) ?? books[0]);
			}
			gaps.hidden = !live || !settled || location?.kind !== 'row';
			if (!gaps.hidden && location?.kind === 'row') {
				const books = ordered.filter(
					(slot) => location?.kind === 'row' && slot.row === location.anchor.row,
				);
				const first = books[0];
				const last = books[books.length - 1];
				place(
					gaps,
					project,
					first.x - first.width / 2 - 0.004,
					first.y - 0.004,
					last.x + last.width / 2 + 0.004,
					first.y + Math.max(...books.map((slot) => slot.height)) + 0.004,
					first.z,
					true,
				);
			}
			for (const slot of ordered) {
				const link = links.get(slot.isbn);
				if (!link) continue;
				link.hidden =
					!live || !settled || location?.kind !== 'row' || slot.row !== location.anchor.row;
				if (link.hidden) continue;
				place(
					link,
					project,
					slot.x - slot.width / 2,
					slot.y,
					slot.x + slot.width / 2,
					slot.y + slot.height,
					slot.z,
				);
				if (location?.kind === 'row') link.href = address(location.anchor, slot);
			}
			if (
				settled &&
				returnFocus &&
				!opened &&
				!returnFocus.hidden &&
				!returnFocus.closest('[hidden]')
			) {
				returnFocus.focus({ preventScroll: true });
				returnFocus = undefined;
			}
		},
		dispose() {
			disposed = true;
			events.abort();
			resize.disconnect();
			restoreBook();
			targets.replaceChildren();
		},
	};
}

export type Bookshelf = ReturnType<typeof mountBookshelf>;
