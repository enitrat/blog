/** Narrow declaration for the untyped, published ESM build used by the prototype. */
declare module 'page-flip/dist/js/page-flip.module.js' {
	export class PageFlip {
		constructor(
			element: HTMLElement,
			settings: {
				width: number;
				height: number;
				size: 'fixed';
				showCover: boolean;
				usePortrait: boolean;
				flippingTime: number;
				maxShadowOpacity: number;
				mobileScrollSupport: boolean;
				showPageCorners: boolean;
			},
		);
		loadFromHTML(pages: NodeListOf<HTMLElement>): void;
		on(event: string, callback: () => void): void;
		flip(page: number): void;
		turnToPage(page: number): void;
		flipNext(): void;
		flipPrev(): void;
		turnToNextPage(): void;
		turnToPrevPage(): void;
		getCurrentPageIndex(): number;
		getPageCount(): number;
		getOrientation(): 'portrait' | 'landscape';
	}
}
