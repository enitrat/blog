import * as THREE from 'three';
import { type Book, books } from '../../../booksData';
import {
	PLEIADE_HEX,
	pageRangeFor,
	pleiadeStyleFor,
	thicknessRatioFor,
} from '../../../utils/pleiade';
import { paintBackCover, paintCover, paintGilt, paintSpine } from './pleiade-paint';
import { packShelves } from './shelf-layout';

type Volume = {
	book: Book;
	group: THREE.Group;
	home: THREE.Vector3;
	button: HTMLButtonElement;
	width: number;
	height: number;
	color: string;
	coverMaterial: THREE.MeshBasicMaterial;
	shelfCover: THREE.CanvasTexture;
	/** 0 shelved, 1 fully drawn out under the pointer. */
	lift: number;
};

type Motion = {
	volume: Volume;
	from: THREE.Vector3;
	to: THREE.Vector3;
	rotationFrom: THREE.Quaternion;
	rotationTo: THREE.Quaternion;
	scaleFrom: number;
	scaleTo: number;
	start: number;
	duration: number;
	done: () => void;
};

/** Everything the reader iframe is allowed to say. */
type ReaderMessage =
	| { type: 'reader-closed' }
	| { type: 'reader-close' }
	| { type: 'reader-page'; page: number; count: number }
	| { type: 'reader-ready'; cover: DOMRectReadOnly };

const isNumber = (value: unknown): value is number => typeof value === 'number';

/** Validate a postMessage payload at the trust boundary. Unknown shapes are dropped. */
function readerMessage(data: unknown): ReaderMessage | undefined {
	if (typeof data !== 'object' || data === null) return;
	const { type, page, count, x, y, width, height } = data as Record<string, unknown>;
	if (type === 'reader-closed' || type === 'reader-close') return { type };
	if (type === 'reader-page' && isNumber(page) && isNumber(count)) return { type, page, count };
	if (type === 'reader-ready' && isNumber(x) && isNumber(y) && isNumber(width) && isNumber(height))
		return { type, cover: new DOMRectReadOnly(x, y, width, height) };
}

const PITCH = 2.64;
/* A shelf board is 0.22 thick centred 0.24 above its shelf line, and a book's
   base sits at 0.36 — leaving this much clear height under the board above. */
const SHELF_INTERIOR = PITCH - 0.23;
/* Air left above the spines. A hovered book rises 0.05 and tips its head
   forward, lifting its back top corner ~0.07; without this it drove straight
   into the board above, and the top of the book was never visible. */
const HOVER_HEADROOM = 0.24;
/* Every volume is the same height. The Bibliothèque de la Pléiade is a single
   standardised format — a flat top line is the collection's signature. Spine
   thickness still varies with page count, which is the real difference. */
const BOOK_HEIGHT = SHELF_INTERIOR - HOVER_HEADROOM;
const HOVER_OUT = 0.55;
const HOVER_TILT = 0.06;
const CORNICE_OVERHANG = 0.55;
const DEPTH = 1.28;
const BREAKPOINT = 720;
/* Sized to how large these actually draw: a spine occupies ~130 CSS px, so at a
   pixel ratio of 2 anything past 256 is oversampling — 55 books' worth of it. */
const SPINE_TEXTURE = { width: 256, height: 1024 };
const SHELF_COVER_TEXTURE = { width: 256, height: 384 };
const OPEN_COVER_TEXTURE = { width: 1024, height: 1536 };

let maxAnisotropy = 4;

function texture(width: number, height: number, draw: (ctx: CanvasRenderingContext2D) => void) {
	const canvas = document.createElement('canvas');
	canvas.width = width;
	canvas.height = height;
	const ctx = canvas.getContext('2d');
	if (!ctx) throw new Error('Canvas 2D is required to draw book textures.');
	draw(ctx);
	const result = new THREE.CanvasTexture(canvas);
	result.colorSpace = THREE.SRGBColorSpace;
	result.anisotropy = maxAnisotropy;
	result.minFilter = THREE.LinearMipmapLinearFilter;
	result.magFilter = THREE.LinearFilter;
	return result;
}

/** Mount the library. The scene renders only while something moves. */
export async function startLibrary() {
	const stage = document.getElementById('library-stage');
	const viewport = document.getElementById('library-viewport');
	const canvas = document.getElementById('library-canvas');
	const targets = document.getElementById('book-targets');
	const caption = document.getElementById('book-caption');
	const loading = document.getElementById('library-loading');
	const dialog = document.getElementById('book-reader');
	const mount = document.getElementById('reader-mount');
	const title = document.getElementById('reader-title');
	const pageStatus = document.getElementById('page-status');
	const close = document.getElementById('return-book');
	const previous = document.getElementById('previous-page');
	const next = document.getElementById('next-page');
	if (
		!(canvas instanceof HTMLCanvasElement) ||
		!(dialog instanceof HTMLDialogElement) ||
		!stage ||
		!viewport ||
		!targets ||
		!caption ||
		!loading ||
		!mount ||
		!title ||
		!pageStatus ||
		!(close instanceof HTMLButtonElement) ||
		!(previous instanceof HTMLButtonElement) ||
		!(next instanceof HTMLButtonElement)
	)
		return;

	let renderer: THREE.WebGLRenderer;
	try {
		renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
	} catch {
		loading.textContent = 'The 3D view is unavailable. The reading archive is linked below.';
		return;
	}
	maxAnisotropy = renderer.capabilities.getMaxAnisotropy();
	// The spines are unlit MeshBasicMaterial; ACES only desaturated the leather
	// and lifted the gilt into a wash. Render them as painted.
	renderer.outputColorSpace = THREE.SRGBColorSpace;
	renderer.toneMapping = THREE.NoToneMapping;
	renderer.setClearAlpha(0);
	const applyPixelRatio = () => renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
	applyPixelRatio();
	await document.fonts.ready;

	const reduced = matchMedia('(prefers-reduced-motion: reduce)');
	const hover = matchMedia('(hover: hover) and (pointer: fine)');
	const scene = new THREE.Scene();
	const camera = new THREE.OrthographicCamera(-10, 10, 8, -8, 0.1, 80);
	const layoutCamera = new THREE.OrthographicCamera(-10, 10, 8, -8, 0.1, 80);
	scene.add(new THREE.HemisphereLight('#fff0ce', '#493b2c', 2.1));
	const sun = new THREE.DirectionalLight('#ffe3af', 2.8);
	scene.add(sun);
	const fill = new THREE.DirectionalLight('#eee8dd', 0.7);
	scene.add(fill);

	const oakSize = 1024;
	const oakTexture = texture(oakSize, oakSize, (ctx) => {
		ctx.fillStyle = '#65482f';
		ctx.fillRect(0, 0, oakSize, oakSize);
		for (let i = 0; i < 1200; i++) {
			const seed = Math.sin(i * 127.1) * 43758.5453;
			const noise = seed - Math.floor(seed);
			ctx.strokeStyle =
				i % 3 ? `rgba(34,18,6,${0.03 + noise * 0.13})` : `rgba(230,177,103,${noise * 0.18})`;
			ctx.lineWidth = 0.4 + noise * 1.8;
			ctx.beginPath();
			for (let x = 0; x <= oakSize; x += 8) {
				const y = noise * oakSize + Math.sin(x / 88 + i * 0.6) * 3.2 + Math.sin(x / 220 + i) * 6;
				if (x === 0) ctx.moveTo(x, y);
				else ctx.lineTo(x, y);
			}
			ctx.stroke();
		}
	});
	oakTexture.wrapS = THREE.RepeatWrapping;
	oakTexture.wrapT = THREE.RepeatWrapping;
	const oak = new THREE.MeshStandardMaterial({ map: oakTexture, roughness: 0.8 });
	const verticalGrain = oakTexture.clone();
	verticalGrain.center.set(0.5, 0.5);
	verticalGrain.rotation = Math.PI / 2;
	const verticalOak = new THREE.MeshStandardMaterial({ map: verticalGrain, roughness: 0.8 });
	const darkOak = new THREE.MeshStandardMaterial({ color: '#4c3421', roughness: 1 });
	const trim = new THREE.MeshStandardMaterial({ color: '#9b6d3d', roughness: 0.65 });
	const backTexture = texture(64, 256, (ctx) => {
		const gradient = ctx.createLinearGradient(0, 0, 0, 256);
		gradient.addColorStop(0, '#231a12');
		gradient.addColorStop(0.65, '#624329');
		gradient.addColorStop(1, '#38261a');
		ctx.fillStyle = gradient;
		ctx.fillRect(0, 0, 64, 256);
	});
	const backMaterial = new THREE.MeshBasicMaterial({ map: backTexture });
	const giltMap = texture(256, 64, paintGilt);
	const giltMat = new THREE.MeshBasicMaterial({ map: giltMap });
	const boardMat = new THREE.MeshBasicMaterial({ color: '#241910' });
	const backByColor = new Map<string, THREE.MeshBasicMaterial>();
	const pageRange = pageRangeFor(books);

	const caseGroup = new THREE.Group();
	scene.add(caseGroup);
	const volumes: Volume[] = [];
	let selected: Volume | undefined;
	let hovered: Volume | undefined;
	let motion: Motion | undefined;
	let frame = 0;
	let drawShelf = true;
	let iframe: HTMLIFrameElement | undefined;
	let flight:
		| { renderer: THREE.WebGLRenderer; scene: THREE.Scene; camera: THREE.OrthographicCamera }
		| undefined;
	let readerTimeout = 0;
	let generation = 0;
	let mobile = false;
	let viewHeight = 12;
	let viewWidth = 12;
	let caseWidth = 6;
	let caseHeight = 12;
	let eyeX = 1.9;
	let eyeY = 1.7;
	let eyeZ = 15;

	const box = (
		width: number,
		height: number,
		depth: number,
		x: number,
		y: number,
		z: number,
		material: THREE.Material,
	) => {
		const mesh = new THREE.Mesh(new THREE.BoxGeometry(width, height, depth), material);
		mesh.position.set(x, y, z);
		caseGroup.add(mesh);
		return mesh;
	};

	const lookAtY = (cam: THREE.OrthographicCamera, targetY: number) => {
		cam.position.set(eyeX, targetY + eyeY, eyeZ);
		cam.lookAt(0, targetY, 0);
	};

	for (const book of books) {
		const style = pleiadeStyleFor(book.author);
		const color = PLEIADE_HEX[style.color];
		const width = 0.5 + thicknessRatioFor(book.edition.pageCount, pageRange) * 0.3;
		const group = new THREE.Group();
		const painted = { author: style.label, title: book.title, color };
		const spineMap = texture(SPINE_TEXTURE.width, SPINE_TEXTURE.height, (ctx) =>
			paintSpine(ctx, painted),
		);
		const shelfCover = texture(SHELF_COVER_TEXTURE.width, SHELF_COVER_TEXTURE.height, (ctx) =>
			paintCover(ctx, { author: book.author, title: book.title, color }),
		);
		let backMat = backByColor.get(color);
		if (!backMat) {
			backMat = new THREE.MeshBasicMaterial({
				map: texture(384, 576, (ctx) => paintBackCover(ctx, color)),
			});
			backByColor.set(color, backMat);
		}
		const spineMat = new THREE.MeshBasicMaterial({ map: spineMap });
		const coverMaterial = new THREE.MeshBasicMaterial({ map: shelfCover });
		const body = new THREE.Mesh(new THREE.BoxGeometry(width, BOOK_HEIGHT, DEPTH), [
			coverMaterial,
			backMat,
			giltMat,
			giltMat,
			spineMat,
			boardMat,
		]);
		group.add(body);
		const button = document.createElement('button');
		button.type = 'button';
		button.className = 'book-target';
		button.setAttribute('aria-label', `Open ${book.title}, ${book.author}`);
		button.setAttribute('aria-haspopup', 'dialog');
		targets.append(button);
		const volume: Volume = {
			book,
			group,
			home: new THREE.Vector3(),
			button,
			width,
			height: BOOK_HEIGHT,
			color,
			coverMaterial,
			shelfCover,
			lift: 0,
		};
		volumes.push(volume);
		scene.add(group);
		button.addEventListener('pointerenter', () => {
			if (hover.matches) pointAt(volume);
		});
		button.addEventListener('pointerleave', () => {
			if (hovered === volume) pointAt(undefined);
		});
		button.addEventListener('focus', () => pointAt(volume));
		button.addEventListener('blur', () => {
			if (hovered === volume) pointAt(undefined);
		});
		button.addEventListener('click', () => openBook(volume));
	}

	const pointAt = (volume: Volume | undefined) => {
		if (selected) return;
		hovered = volume;
		caption.textContent = volume ? `${volume.book.title} · ${volume.book.author}` : '';
		requestRender();
	};

	const frameCamera = () => {
		const canvasH = viewport.clientHeight;
		const travel = Math.max(0, caseHeight - viewHeight);
		const maxScroll = Math.max(1, stage.offsetHeight - canvasH);
		const progress = Math.min(1, Math.max(0, -stage.getBoundingClientRect().top / maxScroll));
		const lookY = travel > 0 ? caseHeight - viewHeight / 2 - progress * travel : caseHeight / 2;
		camera.left = -viewWidth / 2;
		camera.right = viewWidth / 2;
		camera.top = viewHeight / 2;
		camera.bottom = -viewHeight / 2;
		lookAtY(camera, lookY);
		camera.updateProjectionMatrix();
		camera.updateMatrixWorld();
		sun.position.set(-6, lookY + 11, 12);
		fill.position.set(7, lookY + 4, 3);
		drawShelf = true;
	};

	/** Rebuild the oak carcass for a given number of shelves. */
	const buildCase = (rows: number) => {
		for (const child of [...caseGroup.children]) {
			if (child instanceof THREE.Mesh) child.geometry.dispose();
			caseGroup.remove(child);
		}
		box(caseWidth, caseHeight, 0.16, 0, caseHeight / 2, -0.82, darkOak);
		for (let row = 0; row < rows; row++) {
			box(caseWidth - 0.36, PITCH - 0.3, 0.04, 0, row * PITCH + 1.32, -0.72, backMaterial);
		}
		for (let row = 0; row <= rows; row++) {
			box(caseWidth + 0.14, 0.22, 1.98, 0, row * PITCH + 0.24, 0.06, oak);
			box(caseWidth + 0.2, 0.06, 0.08, 0, row * PITCH + 0.33, 1.08, trim);
			box(caseWidth + 0.04, 0.09, 0.12, 0, row * PITCH + 0.14, 1.04, darkOak);
		}
		for (const x of [-caseWidth / 2, caseWidth / 2]) {
			box(0.3, caseHeight, 2.08, x, caseHeight / 2, 0.06, verticalOak);
			box(0.1, caseHeight - 0.12, 0.06, x, caseHeight / 2, 1.12, trim);
		}
		box(caseWidth + CORNICE_OVERHANG, 0.24, 2.2, 0, caseHeight + 0.06, 0.04, oak);
		box(caseWidth + 0.38, 0.12, 2.12, 0, caseHeight - 0.1, 0.04, trim);
		box(caseWidth + 0.4, 0.36, 2.16, 0, 0.02, 0.04, oak);
	};

	/** Centre each shelf's run of books, spreading the slack across its seams. */
	const placeVolumes = (shelves: Volume[][], packedWidth: number, gap: number) => {
		for (const [row, rowVolumes] of shelves.entries()) {
			const packed = rowVolumes.reduce((sum, volume) => sum + volume.width, 0);
			const seams = Math.max(1, rowVolumes.length - 1);
			const spread = gap + Math.min(0.03, Math.max(0, (packedWidth - packed) / seams));
			let x = -(packed + spread * (rowVolumes.length - 1)) / 2;
			const shelfY = (shelves.length - 1 - row) * PITCH;
			for (const volume of rowVolumes) {
				x += volume.width / 2;
				volume.home.set(x, shelfY + 0.36 + volume.height / 2, 0.18);
				volume.lift = 0;
				volume.group.position.copy(volume.home);
				volume.group.quaternion.identity();
				volume.group.scale.setScalar(1);
				x += volume.width / 2 + spread;
			}
		}
	};

	/** Size the orthographic view so the whole carcass, cornice included, fits. */
	const fitView = () => {
		const aspect = viewport.clientWidth / Math.max(1, viewport.clientHeight);
		// The cornice overhangs the carcass, so fit the widest board, not caseWidth,
		// or the crown and the stiles get clipped by the viewport edges.
		const outerWidth = caseWidth + CORNICE_OVERHANG;
		viewWidth = outerWidth + (mobile ? 0.1 : 0.28);
		viewHeight = viewWidth / aspect;
		const minHeight = PITCH * (mobile ? 1.3 : 2.05) + 0.3;
		if (viewHeight < minHeight) {
			viewHeight = minHeight;
			viewWidth = viewHeight * aspect;
		}
		if (viewWidth < outerWidth) {
			viewWidth = outerWidth;
			viewHeight = viewWidth / aspect;
		}

		const canvasH = viewport.clientHeight;
		stage.style.height = `${Math.max(canvasH, Math.round(canvasH * (caseHeight / viewHeight)))}px`;
		renderer.setSize(viewport.clientWidth, viewport.clientHeight, false);
	};

	/**
	 * Park each book's hit target over its spine. The buttons are the keyboard and
	 * pointer surface; `layoutCamera` projects at the scroll position that will
	 * bring each shelf into view, so a target lands on its spine at any scroll.
	 */
	const placeButtons = () => {
		layoutCamera.left = -viewWidth / 2;
		layoutCamera.right = viewWidth / 2;
		layoutCamera.top = viewHeight / 2;
		layoutCamera.bottom = -viewHeight / 2;

		const canvasH = viewport.clientHeight;
		const stageW = stage.clientWidth;
		const stageH = stage.clientHeight;
		const spine = new THREE.Vector3();
		for (const volume of volumes) {
			lookAtY(layoutCamera, volume.home.y);
			layoutCamera.updateProjectionMatrix();
			layoutCamera.updateMatrixWorld();
			spine.set(volume.home.x, volume.home.y, volume.home.z + DEPTH / 2).project(layoutCamera);
			const w = (volume.width / viewWidth) * stageW;
			const h = (volume.height / viewHeight) * canvasH;
			Object.assign(volume.button.style, {
				left: `${((spine.x + 1) / 2) * stageW - w / 2}px`,
				top: `${((caseHeight - volume.home.y) / caseHeight) * stageH - h / 2}px`,
				width: `${Math.max(w, 44)}px`,
				height: `${h}px`,
			});
		}
	};

	const layout = () => {
		drawShelf = true;
		applyPixelRatio();
		mobile = viewport.clientWidth < BREAKPOINT;
		// Aim for a spine around 130px wide: wide enough to read the gilt, narrow
		// enough that the case is wider than it is tall and two shelves stay in view.
		const perRow = mobile ? 3 : Math.max(6, Math.round(viewport.clientWidth / 132));
		const gap = mobile ? 0.02 : 0.022;
		const shelves = packShelves(
			volumes,
			(volume) => volume.width + gap,
			Math.ceil(volumes.length / perRow),
		);
		// The packed result, not the request, decides how tall the carcass is.
		const rows = shelves.length;
		const packedWidth = shelves.reduce((max, rowVolumes) => {
			const packed = rowVolumes.reduce((sum, volume) => sum + volume.width, 0);
			return Math.max(max, packed + gap * Math.max(0, rowVolumes.length - 1));
		}, 0);
		caseWidth = packedWidth + 0.68;
		caseHeight = rows * PITCH + 0.62;
		eyeX = mobile ? 0.95 : 1.45;
		eyeY = mobile ? 1.05 : 1.35;
		eyeZ = mobile ? 11 : 14;

		buildCase(rows);
		placeVolumes(shelves, packedWidth, gap);
		fitView();
		placeButtons();
		frameCamera();
		requestRender();
	};

	const requestRender = () => {
		if (!frame) frame = requestAnimationFrame(render);
	};
	let lastFrame = 0;
	const render = (now: number) => {
		frame = 0;
		// Clamped so a backgrounded tab does not snap every book on return.
		const dt = lastFrame ? Math.min(0.05, (now - lastFrame) / 1000) : 0.016;
		lastFrame = now;
		const settle = 1 - Math.exp(-dt * 13);
		let moving = false;
		for (const volume of volumes) {
			if (volume === selected) continue;
			const target = volume === hovered && !reduced.matches ? 1 : 0;
			const difference = target - volume.lift;
			volume.lift =
				Math.abs(difference) < 0.0015 || reduced.matches
					? target
					: volume.lift + difference * settle;
			if (difference !== 0) {
				// Slides out of the shelf and tips its head toward the reader.
				volume.group.position.z = volume.home.z + volume.lift * HOVER_OUT;
				volume.group.position.y = volume.home.y + volume.lift * 0.05;
				volume.group.rotation.x = volume.lift * HOVER_TILT;
				drawShelf = true;
			}
			if (Math.abs(difference) >= 0.0015) moving = true;
		}
		if (motion) {
			const current = motion;
			const t = Math.min(1, (now - current.start) / current.duration);
			const eased = 1 - (1 - t) ** 3;
			current.volume.group.position.lerpVectors(current.from, current.to, eased);
			current.volume.group.quaternion.slerpQuaternions(
				current.rotationFrom,
				current.rotationTo,
				eased,
			);
			current.volume.group.scale.setScalar(
				THREE.MathUtils.lerp(current.scaleFrom, current.scaleTo, eased),
			);
			if (t === 1) {
				motion = undefined;
				current.done();
			} else moving = true;
		}
		if (drawShelf) {
			renderer.render(scene, camera);
			drawShelf = false;
		}
		if (flight) flight.renderer.render(flight.scene, flight.camera);
		if (moving || motion) requestRender();
	};

	const move = (
		volume: Volume,
		to: THREE.Vector3,
		rotationTo: THREE.Quaternion,
		scaleTo: number,
		duration: number,
		done: () => void,
	) => {
		motion = {
			volume,
			from: volume.group.position.clone(),
			to,
			rotationFrom: volume.group.quaternion.clone(),
			rotationTo,
			scaleFrom: volume.group.scale.x,
			scaleTo,
			start: performance.now(),
			duration: reduced.matches ? 1 : duration,
			done,
		};
		requestRender();
	};

	const openBook = (volume: Volume) => {
		if (selected) return;
		selected = volume;
		hovered = undefined;
		dialog.dataset.phase = 'extracting';
		dialog.showModal();
		let flightRenderer: THREE.WebGLRenderer;
		try {
			flightRenderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
		} catch {
			dialog.close();
			selected = undefined;
			caption.textContent = 'The book could not open. Please try again.';
			return;
		}
		volume.coverMaterial.map = texture(OPEN_COVER_TEXTURE.width, OPEN_COVER_TEXTURE.height, (ctx) =>
			paintCover(ctx, {
				author: volume.book.author,
				title: volume.book.title,
				color: volume.color,
			}),
		);
		volume.coverMaterial.needsUpdate = true;
		drawShelf = true;
		flightRenderer.setPixelRatio(Math.min(devicePixelRatio, 2));
		flightRenderer.setSize(innerWidth, innerHeight);
		flightRenderer.outputColorSpace = THREE.SRGBColorSpace;
		flightRenderer.toneMapping = renderer.toneMapping;
		flightRenderer.domElement.className = 'book-flight';
		dialog.prepend(flightRenderer.domElement);
		const flightCamera = camera.clone();
		const bounds = canvas.getBoundingClientRect();
		const units = viewHeight / bounds.height;
		flightCamera.left -= bounds.left * units;
		flightCamera.right += (innerWidth - bounds.right) * units;
		flightCamera.top += bounds.top * units;
		flightCamera.bottom -= (innerHeight - bounds.bottom) * units;
		flightCamera.updateProjectionMatrix();
		const flightScene = new THREE.Scene();
		flightScene.add(
			new THREE.HemisphereLight('#fff0ce', '#493b2c', 2.1),
			sun.clone(),
			fill.clone(),
			volume.group,
		);
		flight = { renderer: flightRenderer, camera: flightCamera, scene: flightScene };
		requestRender();
		close.focus();
		title.textContent = volume.book.title;
		iframe = document.createElement('iframe');
		iframe.title = `Sample pages for ${volume.book.title}`;
		iframe.src = `/bookshelf/reader/?${new URLSearchParams({ book: volume.book.edition.isbn13, color: volume.color })}`;
		mount.replaceChildren(iframe);
		const token = ++generation;
		readerTimeout = window.setTimeout(() => {
			if (token !== generation) return;
			returnBook();
			caption.textContent = 'The reader could not load. Please try again.';
		}, 12000);
	};

	const releaseFlight = () => {
		if (selected) {
			const map = selected.coverMaterial.map;
			if (map && map !== selected.shelfCover) map.dispose();
			selected.coverMaterial.map = selected.shelfCover;
			selected.coverMaterial.needsUpdate = true;
		}
		flight?.renderer.domElement.remove();
		flight?.renderer.dispose();
		flight?.renderer.forceContextLoss();
		flight = undefined;
	};

	const finishReturn = () => {
		if (!selected || dialog.dataset.phase === 'returning') return;
		clearTimeout(readerTimeout);
		generation++;
		const volume = selected;
		iframe?.remove();
		iframe = undefined;
		dialog.dataset.phase = 'returning';
		volume.group.visible = true;
		const outside = volume.home.clone();
		outside.z += 1.7;
		move(volume, outside, new THREE.Quaternion(), 1, 300, () => {
			move(volume, volume.home.clone(), new THREE.Quaternion(), 1, 200, () => {
				scene.add(volume.group);
				releaseFlight();
				drawShelf = true;
				selected = undefined;
				dialog.close();
				dialog.dataset.phase = 'shelf';
				volume.button.focus({ preventScroll: true });
			});
		});
	};

	const returnBook = () => {
		if (dialog.dataset.phase === 'closing') return;
		if (dialog.dataset.phase === 'reading') {
			dialog.dataset.phase = 'closing';
			iframe?.contentWindow?.postMessage('close', location.origin);
			readerTimeout = window.setTimeout(finishReturn, 900);
		} else finishReturn();
	};

	/** Fly the extracted volume onto the cover the reader iframe just laid out. */
	const flyToReader = (volume: Volume, cover: DOMRectReadOnly) => {
		if (!iframe) return;
		clearTimeout(readerTimeout);
		const frameRect = iframe.getBoundingClientRect();
		const canvasRect = canvas.getBoundingClientRect();
		const centerX = frameRect.x + cover.x + cover.width / 2;
		const centerY = frameRect.y + cover.y + cover.height / 2;
		const depth = new THREE.Vector3(0, volume.home.y, 7).project(camera).z;
		const destination = new THREE.Vector3(
			((centerX - canvasRect.x) / canvasRect.width) * 2 - 1,
			1 - ((centerY - canvasRect.y) / canvasRect.height) * 2,
			depth,
		).unproject(camera);
		const orientation = camera.quaternion
			.clone()
			.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -Math.PI / 2));
		const scale = ((cover.height / canvasRect.height) * viewHeight) / volume.height;
		destination.addScaledVector(
			new THREE.Vector3(0, 0, 1).applyQuaternion(camera.quaternion),
			(-volume.width / 2) * scale,
		);
		const outside = volume.home.clone();
		outside.z += 1.7;
		const token = generation;
		move(volume, outside, new THREE.Quaternion(), 1, 200, () => {
			if (token !== generation) return;
			move(volume, destination, orientation, scale, 400, () => {
				if (token !== generation) return;
				volume.group.visible = false;
				dialog.dataset.phase = 'reading';
				iframe?.contentWindow?.postMessage('open', location.origin);
				requestRender();
			});
		});
	};

	const showPage = (page: number, count: number) => {
		pageStatus.textContent =
			page === 0
				? 'Cover · Sample book'
				: page === count - 1
					? 'Back cover'
					: `Page ${page} · Sample book`;
		previous.disabled = page === 0;
		next.disabled = page >= count - 1;
	};

	addEventListener('message', (event: MessageEvent<unknown>) => {
		if (event.origin !== location.origin || event.source !== iframe?.contentWindow || !selected)
			return;
		const message = readerMessage(event.data);
		if (!message) return;
		if (message.type === 'reader-closed') finishReturn();
		else if (message.type === 'reader-close') returnBook();
		else if (message.type === 'reader-page') showPage(message.page, message.count);
		else if (dialog.dataset.phase === 'extracting') flyToReader(selected, message.cover);
	});
	close.addEventListener('click', returnBook);
	dialog.addEventListener('cancel', (event) => {
		event.preventDefault();
		returnBook();
	});
	previous.addEventListener('click', () =>
		iframe?.contentWindow?.postMessage('previous', location.origin),
	);
	next.addEventListener('click', () => iframe?.contentWindow?.postMessage('next', location.origin));
	dialog.addEventListener('keydown', (event) => {
		if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
			event.preventDefault();
			iframe?.contentWindow?.postMessage(
				event.key === 'ArrowRight' ? 'next' : 'previous',
				location.origin,
			);
		}
	});
	let lastSize = '';
	const relayout = () => {
		const size = `${viewport.clientWidth}x${viewport.clientHeight}`;
		if (lastSize === size) return;
		lastSize = size;
		if (selected) {
			generation++;
			clearTimeout(readerTimeout);
			iframe?.remove();
			iframe = undefined;
			selected.group.visible = true;
			scene.add(selected.group);
			releaseFlight();
			selected = undefined;
			motion = undefined;
			dialog.close();
		}
		layout();
	};
	const resize = new ResizeObserver(relayout);
	resize.observe(viewport);
	let pixelRatioQuery: MediaQueryList | undefined;
	const onPixelRatioChange = () => {
		applyPixelRatio();
		renderer.setSize(viewport.clientWidth, viewport.clientHeight, false);
		drawShelf = true;
		watchPixelRatio();
		requestRender();
	};
	function watchPixelRatio() {
		pixelRatioQuery?.removeEventListener('change', onPixelRatioChange);
		pixelRatioQuery = matchMedia(`(resolution: ${devicePixelRatio}dppx)`);
		pixelRatioQuery.addEventListener('change', onPixelRatioChange, { once: true });
	}
	watchPixelRatio();
	const onScroll = () => {
		frameCamera();
		requestRender();
	};
	addEventListener('scroll', onScroll, { passive: true });
	visualViewport?.addEventListener('resize', relayout);
	reduced.addEventListener('change', requestRender);
	loading.hidden = true;
	relayout();
	addEventListener(
		'pagehide',
		() => {
			cancelAnimationFrame(frame);
			clearTimeout(readerTimeout);
			resize.disconnect();
			pixelRatioQuery?.removeEventListener('change', onPixelRatioChange);
			removeEventListener('scroll', onScroll);
			visualViewport?.removeEventListener('resize', relayout);
			iframe?.remove();
			releaseFlight();
			renderer.dispose();
		},
		{ once: true },
	);
}
