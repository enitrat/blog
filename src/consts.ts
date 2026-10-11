// Place any global data in this file.
// You can import this data from anywhere in your site by using the `import` keyword.

export const SITE_DESCRIPTION = 'I build systems and products, and write about what I learn.';

/** Where to reach Mathieu; the footer and the home bio read this one list. */
export const CONTACT_LINKS = [
	{ label: 'GitHub', href: 'https://github.com/enitrat' },
	{ label: 'X', href: 'https://x.com/eniwhere_' },
	{ label: 'Telegram', href: 'https://t.me/enitrat' },
] as const;

/** The last set recorded: one SoundCloud track, shared by the set tile and the room's record player. */
export const SET = {
	title: 'Bouncyx Seven',
	artist: 'eni',
	url: 'https://soundcloud.com/user2211512/bouncyx-seven',
	embed:
		'https://w.soundcloud.com/player/?url=https%3A//soundcloud.com/user2211512/bouncyx-seven&auto_play=false&hide_related=true&show_comments=false&show_teaser=false',
	/** ISO date the track was published on SoundCloud (its `display_date`). */
	recorded: '2026-09-26',
} as const;
