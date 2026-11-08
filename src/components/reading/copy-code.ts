/**
 * Adds a Copy button to every code block under `root`, cloned from `template`. Without JS the
 * code is still selectable and copyable by hand.
 */

/** Milliseconds "Copied!" stays before the button reads "Copy" again. */
const RESET = 2000;

/** Starts the copy buttons; returns the teardown, which removes them. */
export function initCopyCode(root: HTMLElement, template: HTMLElement): () => void {
	const source = template.firstElementChild;
	if (!source) return () => {};
	const ac = new AbortController();
	const added: Element[] = [];
	const observers: ResizeObserver[] = [];
	const timers = new Set<number>();
	// One counter per language, so every block has a name that tells the reader which sample
	// they are on, rather than repeating "Copy".
	const seen = new Map<string, number>();

	for (const block of root.querySelectorAll<HTMLElement>('pre:has(> code)')) {
		if (block.querySelector('.copy-btn')) continue;
		const language = /language-(\S+)/.exec(block.querySelector('code')?.className ?? '')?.[1];
		const kind = language && language !== 'plaintext' && language !== 'text' ? language : 'code';
		const ordinal = (seen.get(kind) ?? 0) + 1;
		seen.set(kind, ordinal);
		const name = `${kind} sample ${ordinal}`;

		block.dataset.copyCode = '';
		block.setAttribute('role', 'region');
		block.setAttribute('aria-label', name);

		// A scroll container is reachable by keyboard only if it is focusable, but a block with
		// nothing to scroll would be a tab stop that does nothing, so the stop is added only while
		// the code overflows.
		const syncFocusable = () => {
			if (block.scrollWidth > block.clientWidth) block.setAttribute('tabindex', '0');
			else block.removeAttribute('tabindex');
		};
		syncFocusable();
		const ro = new ResizeObserver(syncFocusable);
		ro.observe(block);
		observers.push(ro);

		// The visible word starts the accessible name, so the spoken name and the label match.
		const button = source.cloneNode(true) as HTMLButtonElement;
		button.setAttribute('aria-label', `Copy ${name}`);

		const status = document.createElement('span');
		status.className = 'sr-only';
		status.setAttribute('role', 'status');

		button.addEventListener(
			'click',
			async () => {
				// The markdown fence leaves a closing newline in the code; the copy leaves it out.
				const text = (block.querySelector('code')?.textContent ?? '').replace(/\n$/, '');
				try {
					await navigator.clipboard.writeText(text);
					button.textContent = 'Copied!';
					button.dataset.copied = '';
					status.textContent = 'Copied to clipboard';
				} catch {
					button.textContent = 'Failed';
					status.textContent = 'Copy failed';
				}
				const timer = window.setTimeout(() => {
					timers.delete(timer);
					button.textContent = 'Copy';
					delete button.dataset.copied;
					status.textContent = '';
				}, RESET);
				timers.add(timer);
			},
			{ signal: ac.signal },
		);

		block.append(button, status);
		added.push(button, status);
	}

	return () => {
		ac.abort();
		for (const ro of observers) ro.disconnect();
		for (const t of timers) clearTimeout(t);
		for (const el of added) el.remove();
	};
}
