import { SET } from '../consts';

/** One state, not two booleans that could disagree. `cueing`: asked to play,
 *  but the mix has not started yet. */
export type Playback = 'stopped' | 'cueing' | 'playing';

type Widget = {
	play(): void;
	pause(): void;
	setVolume(volume: number): void;
	getPosition(callback: (ms: number) => void): void;
	bind(event: string, handler: () => void): void;
};
type SoundCloud = ((frame: Element) => Widget) & {
	Events: { PLAY: string; PAUSE: string; FINISH: string; ERROR: string; READY: string };
};

let widget: Widget | null = null;
let connecting: Promise<Widget> | null = null;
let playback: Playback = 'stopped';
let volume = 100;
const listeners = new Set<(state: Playback) => void>();

function set(next: Playback) {
	if (next === playback) return;
	playback = next;
	// Pure-CSS views (`:root[data-set-playback="playing"] ...`) need no wiring.
	document.documentElement.dataset.setPlayback = next;
	for (const listener of listeners) listener(next);
}

async function connect() {
	const frame = document.createElement('iframe');
	frame.className = 'living-room__widget';
	frame.title = `SoundCloud player: ${SET.title} by ${SET.artist}`;

	frame.allow = 'autoplay; encrypted-media';
	frame.src = SET.embed;
	document.body.append(frame);

	await new Promise((resolve, reject) => {
		const api = document.createElement('script');
		api.src = 'https://w.soundcloud.com/player/api.js';
		api.onload = resolve;
		api.onerror = reject;
		document.head.append(api);
	});

	const SC = (window as Window & { SC?: { Widget: SoundCloud } }).SC;
	if (!SC) throw new Error('no SC.Widget');
	const created = SC.Widget(frame);
	const events = SC.Widget.Events;

	created.bind(events.PLAY, () => {
		if (playback === 'stopped') return created.pause();
		set('playing');
	});
	created.bind(events.PAUSE, () => {
		if (document.hidden) return;

		created.getPosition((position) => {
			if (position > 0) stop();
		});
	});
	created.bind(events.FINISH, stop);
	created.bind(events.ERROR, stop);

	created.bind(events.READY, () => {
		created.setVolume(volume);
		if (playback !== 'stopped') created.play();
	});
	return created;
}

function stop() {
	widget?.pause();
	set('stopped');
}

function play() {
	if (playback !== 'stopped') return;
	set('cueing');
	if (widget) {
		widget.play();
		return;
	}
	// One iframe and one api.js however fast the button is pressed: without
	// this, a second press before the first resolves appends a second player
	// that nobody holds a reference to and nothing can ever pause.
	connecting ??= connect();
	connecting.then(
		(created) => {
			widget = created;
		},
		() => {
			connecting = null;
			// The widget never came up: do not leave a view stuck on `cueing`.
			set('stopped');
		},
	);
}

// Only a mix that actually started is worth resuming: intent that never
// became playback should not begin on its own on the way back.
let resume = false;
document.addEventListener('visibilitychange', () => {
	if (document.hidden) {
		resume = playback === 'playing';
		if (resume) widget?.pause();
	} else if (resume) {
		resume = false;
		widget?.play();
	}
});

/** The one SoundCloud widget on the page, shared by every view of the set. The
 *  iframe and api.js load on the first `play()`, never on page load. */
export const setPlayer = {
	get state() {
		return playback;
	},
	play,
	stop,
	toggle() {
		if (playback === 'stopped') play();
		else stop();
	},
	setVolume(next: number) {
		volume = next;
		widget?.setVolume(next);
	},
	/** Calls `fn` now with the current state, then on every change. */
	subscribe(fn: (state: Playback) => void) {
		listeners.add(fn);
		fn(playback);
		return () => void listeners.delete(fn);
	},
};
