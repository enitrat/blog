const covers = import.meta.glob<{ default: ImageMetadata }>('../assets/covers/*.{jpg,png,webp}', {
	eager: true,
});

/** Cover art for an ISBN, from `src/assets/covers/<isbn>.<ext>`; undefined when none is checked in. */
export function coverFor(isbn: string): ImageMetadata | undefined {
	return Object.entries(covers).find(([path]) => path.split('/').pop()?.startsWith(`${isbn}.`))?.[1]
		.default;
}
