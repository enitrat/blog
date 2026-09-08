/** Display the baked room and frame its navigable bookshelf. */
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import bookmarkUrl from '../../../assets/room/bookmark.glb?url';
import booksUrl from '../../../assets/room/books.glb?url';
import coversUrl from '../../../assets/room/covers.glb?url';
import furnitureUrl from '../../../assets/room/furniture.glb?url';
import movingUrl from '../../../assets/room/moving.glb?url';
import objectsUrl from '../../../assets/room/objects.glb?url';
import shellUrl from '../../../assets/room/shell.glb?url';
// The printed jackets carry the only type in the room meant to be read, so they
// own an atlas instead of sharing one with nine square metres of leather.
import spinesUrl from '../../../assets/room/spines.glb?url';
import type { Bookshelf } from './bookshelf';
import { BOOKSHELF_ANCHOR, CABINET_BOX, ordered, type ShelfFrame } from './shelves';

const VIEWS = {
	room: { position: new THREE.Vector3(6.4, 6.5, 9.9), target: new THREE.Vector3(0, 1, -0.2) },
	closer: {
		position: new THREE.Vector3(0.9, 2.4, 4.8),
		target: new THREE.Vector3(0, 0.94, -1.05),
	},
	record: {
		position: new THREE.Vector3(0.98, 1.48, 0.55),
		target: new THREE.Vector3(0.21, 0.8, -1.16),
	},
};
const ANCHORS = {
	// The cabinet's ring sits on its books, and the bake owns where those are.
	bookshelf: new THREE.Vector3(BOOKSHELF_ANCHOR.x, BOOKSHELF_ANCHOR.y, BOOKSHELF_ANCHOR.z),
	record: new THREE.Vector3(0.21, 0.78, -1.16),
};

type View = keyof typeof VIEWS | 'shelf';

/** Mount a room, or leave its poster in place if an asset or WebGL fails. */
export async function mountRoom(
	host: HTMLElement,
	canvas: HTMLCanvasElement,
	bookshelf: Bookshelf,
): Promise<(() => void) | null> {
	const scene = new THREE.Scene();
	const materials = new Set<THREE.Material>();
	const textures = new Set<THREE.Texture>();
	const geometries = new Set<THREE.BufferGeometry>();
	const loader = new GLTFLoader();
	const releaseAssets = () => {
		for (const geometry of geometries) geometry.dispose();
		for (const material of materials) material.dispose();
		for (const texture of textures) {
			texture.dispose();
			if (typeof ImageBitmap !== 'undefined' && texture.source.data instanceof ImageBitmap) {
				texture.source.data.close();
			}
		}
	};
	// Await all loads even after failure so a late success cannot leak GPU resources.
	const results = await Promise.allSettled(
		[
			shellUrl,
			furnitureUrl,
			objectsUrl,
			movingUrl,
			spinesUrl,
			booksUrl,
			coversUrl,
			bookmarkUrl,
		].map(async (url) => {
			const gltf = await loader.loadAsync(url);
			gltf.scene.traverse((object) => {
				if (!(object instanceof THREE.Mesh)) return;
				geometries.add(object.geometry);
				for (const material of Array.isArray(object.material)
					? object.material
					: [object.material]) {
					materials.add(material);
					if (material instanceof THREE.MeshBasicMaterial && material.map) {
						material.map.anisotropy = 4;
						textures.add(material.map);
					}
				}
			});
			scene.add(gltf.scene);
		}),
	);
	if (results.some((result) => result.status === 'rejected') || !host.isConnected) {
		releaseAssets();
		return null;
	}

	const books = new Map<string, THREE.Group>();
	const covers = new Map<string, THREE.Object3D>();

	/** Recover each printed jacket from the shared atlas using its ISBN slot. */
	function slice(mesh: THREE.Mesh, depth: number) {
		mesh.updateWorldMatrix(true, false);
		const flat = mesh.geometry.clone().toNonIndexed().applyMatrix4(mesh.matrixWorld);
		const position = flat.getAttribute('position');
		const uv = flat.getAttribute('uv');
		const mine = new Map<string, number[]>();
		const rest: number[] = [];
		for (let first = 0; first < position.count; first += 3) {
			let x = 0;
			let y = 0;
			let z = 0;
			for (let corner = 0; corner < 3; corner += 1) {
				x += position.getX(first + corner) / 3;
				y += position.getY(first + corner) / 3;
				z += position.getZ(first + corner) / 3;
			}
			// A book stands in its slot, so the slot it stands in names it. Rows
			// share the run of x, hence the height as well; the shelf and what
			// else the room leaves on it are behind or below every slot.
			const slot = ordered.find(
				(candidate) =>
					Math.abs(x - candidate.x) <= candidate.width / 2 &&
					y >= candidate.y - 0.004 &&
					y <= candidate.y + candidate.height + 0.004 &&
					z <= candidate.z + 0.005 &&
					z >= candidate.z - depth,
			);
			if (!slot) rest.push(first);
			else mine.set(slot.isbn, [...(mine.get(slot.isbn) ?? []), first]);
		}
		const build = (corners: number[]) => {
			const geometry = new THREE.BufferGeometry();
			const points = new Float32Array(corners.length * 9);
			const texture = new Float32Array(corners.length * 6);
			let vertex = 0;
			for (const first of corners)
				for (let corner = 0; corner < 3; corner += 1, vertex += 1) {
					points[vertex * 3] = position.getX(first + corner);
					points[vertex * 3 + 1] = position.getY(first + corner);
					points[vertex * 3 + 2] = position.getZ(first + corner);
					texture[vertex * 2] = uv.getX(first + corner);
					texture[vertex * 2 + 1] = uv.getY(first + corner);
				}
			geometry.setAttribute('position', new THREE.BufferAttribute(points, 3));
			geometry.setAttribute('uv', new THREE.BufferAttribute(texture, 2));
			return geometry;
		};
		const owned = new Map([...mine].map(([isbn, corners]) => [isbn, build(corners)] as const));
		const left = build(rest);
		flat.dispose();
		return { owned, left };
	}

	const atlas = scene.getObjectByName('spines');
	const bookmark = scene.getObjectByName('Bookmark');
	if (bookmark) bookmark.visible = false;
	if (atlas instanceof THREE.Mesh) {
		const printed = slice(atlas, 0.006);

		for (const slot of ordered) {
			const jacket = printed.owned.get(slot.isbn);
			const body = scene.getObjectByName(`Body_${slot.isbn}`);
			const cover = scene.getObjectByName(`Cover_${slot.isbn}`);
			if (!jacket || !body || !cover) {
				releaseAssets();
				return null;
			}
			const book = new THREE.Group();
			book.position.set(slot.x, slot.y, slot.z);
			jacket.translate(-slot.x, -slot.y, -slot.z);
			geometries.add(jacket);
			book.add(new THREE.Mesh(jacket, atlas.material));
			scene.add(book);
			book.attach(body);
			book.attach(cover);
			covers.set(slot.isbn, cover);
			if (bookshelf.hasNotes(slot.isbn)) {
				for (const surface of book.children) {
					if (
						!(surface instanceof THREE.Mesh) ||
						!(surface.material instanceof THREE.MeshBasicMaterial)
					)
						continue;
					surface.material = surface.material.clone();
					surface.material.color.setRGB(1.12, 1.07, 1.02);
					materials.add(surface.material);
				}
			}
			if (bookshelf.isReading(slot.isbn) && bookmark) {
				const ribbon = bookmark.clone();
				ribbon.visible = true;
				ribbon.position.set(slot.width * 0.12, slot.height, 0);
				book.add(ribbon);
			}
			books.set(slot.isbn, book);
		}
		atlas.visible = false;
		printed.left.dispose();
	}

	let renderer: THREE.WebGLRenderer;
	try {
		renderer = new THREE.WebGLRenderer({
			canvas,
			antialias: true,
			alpha: false,
			powerPreference: 'low-power',
		});
	} catch {
		releaseAssets();
		return null;
	}
	renderer.setClearColor(0xedeae3, 1);
	renderer.outputColorSpace = THREE.SRGBColorSpace;
	renderer.toneMapping = THREE.NoToneMapping;
	renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
	const camera = new THREE.PerspectiveCamera(23.83, 16 / 9, 0.05, 50);
	let fromFov = camera.fov;
	let destinationFov = camera.fov;
	const position = VIEWS.room.position.clone();
	const target = VIEWS.room.target.clone();
	const fromPosition = position.clone();
	const fromTarget = target.clone();
	const destination = position.clone();
	const destinationTarget = target.clone();
	const offset = new THREE.Vector2();
	const wantedOffset = new THREE.Vector2();
	const projected = new THREE.Vector3();
	const shelfPosition = new THREE.Vector3();
	const shelfTarget = new THREE.Vector3();
	const motion = matchMedia('(prefers-reduced-motion: reduce)');
	const events = new AbortController();
	const hotspots = [...host.querySelectorAll<HTMLElement>('[data-anchor]')];
	let view: View = 'room';
	let alive = true;
	let visible = true;
	let lost = false;
	let frame = 0;
	let transitionStart: number | null = null;
	let transitionDuration = 750;
	let shelfFrame: ShelfFrame | null = null;
	let lastTime = 0;
	const raycaster = new THREE.Raycaster();
	const pointer = new THREE.Vector2();
	const cabinet = new THREE.Box3(
		new THREE.Vector3(...CABINET_BOX.min),
		new THREE.Vector3(...CABINET_BOX.max),
	);

	// The platter and the arm are their own nodes in moving.glb, turning about
	// their own axis. Blender's Z became the node's Y in the glTF conversion.
	const platter = scene.getObjectByName('Platter');
	const tonearm = scene.getObjectByName('Tonearm');
	const SPEED = (100 * Math.PI) / 90; // 33 1/3 rpm
	const CUED = -0.1846; // the headshell reaches the lead-in groove
	const DROP = 0.1; // and noses down onto it
	let cueing = host.hasAttribute('data-playing');
	let cue = 0; // 0 parked, 1 on the record
	let spin = 0; // rad/s, spinning up and down like a real platter

	/** How far a volume leans out when it is reached for. A finger on the top of
	 *  a book tips it about this far before it comes free of the row. */
	const TILT = THREE.MathUtils.degToRad(6);
	let reached: string | null = null;
	let reading: (typeof ordered)[number] | undefined;
	let opening = 0;
	const readingTarget = new THREE.Vector3();
	const readingCamera = new THREE.Vector3();
	const lookAt = new THREE.Vector3();

	function pull(dt: number) {
		let moving = false;
		const requested = motion.matches ? null : bookshelf.openBook;
		if (requested && reading?.isbn !== requested) {
			reading = ordered.find((slot) => slot.isbn === requested);
			opening = 0;
		}
		const wasOpening = opening;
		opening = motion.matches
			? 0
			: THREE.MathUtils.clamp(opening + dt / (requested ? 0.9 : -0.65), 0, 1);
		host.toggleAttribute('data-book-moving', opening > 0);
		for (const slot of ordered) {
			const book = books.get(slot.isbn);
			if (!book) continue;
			const presenting = slot.isbn === reading?.isbn && opening > 0;
			const wanted = motion.matches || presenting ? 0 : slot.isbn === reached ? TILT : 0;
			const z = slot.z + (motion.matches || presenting ? 0 : slot.isbn === reached ? 0.008 : 0);
			book.rotation.x = THREE.MathUtils.damp(book.rotation.x, wanted, 16, dt);
			book.rotation.y = 0;
			book.position.x = slot.x;
			const cover = covers.get(slot.isbn);
			if (cover) cover.rotation.y = 0;
			if (presenting) {
				const turn = THREE.MathUtils.smoothstep(opening, 0.3, 0.58);
				const unfold = THREE.MathUtils.smoothstep(opening, 0.72, 1);
				book.position.x = THREE.MathUtils.lerp(
					slot.x,
					(shelfFrame?.x ?? slot.x) - 0.075 * (1 - unfold),
					turn,
				);
				book.position.z =
					slot.z +
					0.19 * THREE.MathUtils.smoothstep(opening, 0, 0.3) +
					0.13 * THREE.MathUtils.smoothstep(opening, 0.3, 0.58);
				book.rotation.y = (-Math.PI / 2) * turn;
				if (cover) cover.rotation.y = -Math.PI * unfold;
				moving ||= opening < 1 || !requested;
				continue;
			}
			book.position.z =
				wasOpening > 0 && opening === 0 && slot.isbn === reading?.isbn
					? slot.z
					: THREE.MathUtils.damp(book.position.z, z, 16, dt);
			if (Math.abs(book.rotation.x - wanted) < 0.0004) book.rotation.x = wanted;
			if (Math.abs(book.position.z - z) < 0.0001) book.position.z = z;
			moving ||= book.rotation.x !== wanted || book.position.z !== z;
		}
		if (wasOpening > 0 && opening === 0) {
			reading = undefined;
			settle();
		}
		return moving;
	}

	function turntable(dt: number) {
		const wanted = cueing ? 1 : 0;
		if (motion.matches) {
			// A record turning forever is exactly what reduced motion asks to stop.
			cue = wanted;
			spin = 0;
		} else {
			cue = THREE.MathUtils.damp(cue, wanted, 4, dt);
			if (Math.abs(cue - wanted) < 0.001) cue = wanted;
			spin = THREE.MathUtils.damp(spin, cueing ? SPEED : 0, 2.5, dt);
			if (Math.abs(spin - (cueing ? SPEED : 0)) < 0.01) spin = cueing ? SPEED : 0;
			if (platter) platter.rotation.y -= spin * dt;
		}
		if (tonearm) {
			tonearm.rotation.y = CUED * cue;
			tonearm.rotation.x = DROP * cue;
		}
		return spin !== 0 || cue !== wanted;
	}

	function draw() {
		camera.position.copy(position);
		camera.position.x += offset.x;
		camera.position.y += offset.y;
		lookAt.copy(target);
		if (reading && opening > 0) {
			readingTarget.set(
				shelfFrame?.x ?? reading.x,
				reading.y + reading.height / 2,
				reading.z + 0.32 + reading.width / 2,
			);
			const distance =
				Math.max(0.29, 0.34 / camera.aspect) /
				(2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
			readingCamera.copy(readingTarget);
			readingCamera.z += distance;
			const retreat = THREE.MathUtils.smoothstep(opening, 0, 0.55);
			camera.position.lerp(readingCamera, retreat);
			lookAt.lerp(readingTarget, retreat);
		}
		camera.lookAt(lookAt);
		camera.updateMatrixWorld();
		renderer.render(scene, camera);
		if (reading && opening > 0) {
			const bounds = canvas.getBoundingClientRect();
			const center = books.get(reading.isbn)?.position.x ?? readingTarget.x;
			projected
				.set(center - 0.146, reading.y + reading.height - 0.003, readingTarget.z)
				.project(camera);
			const left = bounds.left + ((projected.x + 1) * bounds.width) / 2;
			const top = bounds.top + ((1 - projected.y) * bounds.height) / 2;
			projected.set(center + 0.146, reading.y + 0.003, readingTarget.z).project(camera);
			bookshelf.positionReader(
				{
					left,
					top,
					width: bounds.left + ((projected.x + 1) * bounds.width) / 2 - left,
					height: bounds.top + ((1 - projected.y) * bounds.height) / 2 - top,
					controlsTop: bounds.bottom + 68,
				},
				THREE.MathUtils.smoothstep(opening, 0.92, 1),
			);
		}
		for (const hotspot of hotspots) {
			const name = hotspot.dataset.anchor;
			if (name !== 'bookshelf' && name !== 'record') continue;
			projected.copy(ANCHORS[name]).project(camera);
			// A close view leaves the other object off frame: take its ring out of
			// the picture and out of the tab order rather than parking it outside.
			// The wide mix credit needs more room than a ring before it fits.
			const slack = hotspot.classList.contains('living-room__mix') ? 0.6 : 0.98;
			// At reading distance the ring has nothing left to offer, so it steps aside.
			hotspot.hidden =
				bookshelf.active ||
				projected.z < -1 ||
				projected.z > 1 ||
				Math.abs(projected.x) > slack ||
				Math.abs(projected.y) > slack;
			hotspot.style.left = `${(projected.x * 0.5 + 0.5) * 100}%`;
			hotspot.style.top = `${(-projected.y * 0.5 + 0.5) * 100}%`;
		}
	}

	function tick(now: number) {
		frame = 0;
		if (!alive || lost || !visible || document.hidden) return;
		// A frame's timestamp is the moment it began, which can precede a
		// `performance.now()` taken while scheduling it. Time never runs backwards:
		// damping would undo itself, and dividing a negative step by a negative
		// rate would open a book nobody reached for.
		const dt = Math.min(Math.max(now - lastTime, 0) / 1000, 0.05);
		lastTime = now;
		if (transitionStart !== null) {
			const t = motion.matches ? 1 : Math.min((now - transitionStart) / transitionDuration, 1);
			const eased = t < 0.5 ? 16 * t ** 5 : 1 - (-2 * t + 2) ** 5 / 2;
			position.lerpVectors(fromPosition, destination, eased);
			target.lerpVectors(fromTarget, destinationTarget, eased);
			camera.fov = THREE.MathUtils.lerp(fromFov, destinationFov, eased);
			camera.updateProjectionMatrix();
			if (t === 1) {
				transitionStart = null;
				settle();
			}
		}
		offset.lerp(wantedOffset, motion.matches ? 1 : 1 - Math.exp(-8 * dt));
		if (offset.distanceToSquared(wantedOffset) < 0.000001) offset.copy(wantedOffset);
		const turning = turntable(dt);
		const pulling = pull(dt);
		draw();
		if (transitionStart !== null || turning || pulling || !offset.equals(wantedOffset))
			requestDraw();
	}

	function requestDraw() {
		if (!frame && alive && visible && !lost && !document.hidden)
			frame = requestAnimationFrame(tick);
	}

	function travel(
		next: View,
		nextPosition: THREE.Vector3,
		nextTarget: THREE.Vector3,
		duration = 750,
		fov = 23.83,
	) {
		if (next === view && destination.equals(nextPosition) && destinationTarget.equals(nextTarget)) {
			// Already going where we are asked to go. Arrived, the targets still
			// need placing in case a resize moved the projection under them; mid
			// flight, settling here would strand them at a view we are leaving.
			if (transitionStart === null) settle();
			return;
		}
		fromPosition.copy(position);
		fromTarget.copy(target);
		fromFov = camera.fov;
		destinationFov = fov;
		destination.copy(nextPosition);
		destinationTarget.copy(nextTarget);
		transitionDuration = duration;
		view = next;
		canvas.style.cursor = '';
		transitionStart = duration > 0 ? performance.now() : null;
		host.toggleAttribute('data-traveling', transitionStart !== null);
		wantedOffset.set(0, 0);
		offset.set(0, 0);
		publish(view);
		arm(next !== 'shelf');
		if (transitionStart === null) {
			position.copy(destination);
			target.copy(destinationTarget);
			camera.fov = fov;
			camera.updateProjectionMatrix();
			settle();
		}
		requestDraw();
	}

	function changeView(next: keyof typeof VIEWS) {
		travel(next, VIEWS[next].position, VIEWS[next].target);
	}

	/** The camera has arrived: place the shelf targets once, against the view
	 *  they will actually be seen in. */
	function settle() {
		host.removeAttribute('data-traveling');
		camera.position.copy(position);
		camera.position.x += offset.x;
		camera.position.y += offset.y;
		camera.lookAt(target);
		camera.updateMatrixWorld();
		bookshelf.place((x, y, z) => {
			projected.set(x, y, z).project(camera);
			return { x: projected.x, y: projected.y, z: projected.z };
		});
	}

	const sameShelf = (a: ShelfFrame, b: ShelfFrame) =>
		a.view === b.view &&
		a.x === b.x &&
		a.y === b.y &&
		a.z === b.z &&
		a.width === b.width &&
		a.height === b.height;

	function frameShelf(shelf: ShelfFrame | null, duration?: number) {
		// `layout()` rebuilds its frames on every resize, so the same shelf arrives
		// as a new object. Framing the shelf we are already framing is a re-
		// projection, not a journey: it must not start the flight again.
		const reframe = shelfFrame !== null && shelf !== null && sameShelf(shelfFrame, shelf);
		const lateral = shelfFrame?.view === 'row' && shelf?.view === 'row';
		shelfFrame = shelf;
		if (!shelf) {
			if (view === 'shelf') changeView('room');
			return;
		}
		camera.aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
		camera.updateProjectionMatrix();
		// A wider lens puts the reading camera beyond the foreground lampshade,
		// rather than inside it. The room and cabinet keep their authored lens.
		const fov = shelf.view === 'row' ? 40 : 23.83;
		const distance =
			Math.max(shelf.height, shelf.width / camera.aspect) /
			(2 * Math.tan(THREE.MathUtils.degToRad(fov / 2)));
		// Stepping along a row is a short pan; arriving at the shelves is a trip.
		travel(
			'shelf',
			shelfPosition.set(shelf.x, shelf.y + (shelf.view === 'row' ? 0.06 : 0), shelf.z + distance),
			shelfTarget.set(shelf.x, shelf.y, shelf.z),
			duration ?? (reframe ? 0 : lateral ? 280 : shelf.view === 'row' ? 500 : 750),
			fov,
		);
	}

	// The mix link shares the record's anchor but leaves the page: only the
	// button in front of the turntable moves the camera.
	host.querySelector<HTMLButtonElement>('button[data-anchor="record"]')?.addEventListener(
		'click',
		() => {
			if (!bookshelf.active) changeView('record');
		},
		{ signal: events.signal },
	);

	// The same visible control backs out of a close view and out of the shelves:
	// the bookshelf owns it while browsing, the camera owns it otherwise.
	host.querySelector<HTMLButtonElement>('[data-shelf-exit]')?.addEventListener(
		'click',
		() => {
			if (!bookshelf.active) changeView('room');
		},
		{ signal: events.signal },
	);

	// The DOM owns playback; the room follows the state it publishes.
	const playing = new MutationObserver(() => {
		cueing = host.hasAttribute('data-playing');
		lastTime = performance.now();
		requestDraw();
	});
	playing.observe(host, { attributeFilter: ['data-playing'] });

	function publish(current: View) {
		host.dataset.view = current;
		canvas.setAttribute(
			'aria-label',
			current === 'room' ? 'Step closer into the room' : 'Step back to the whole room',
		);
	}

	// Clicking the room itself steps in, and steps back out from any close view.
	// The canvas is the control, so the keyboard reaches it without any chrome.
	const step = () => {
		if (!bookshelf.active) changeView(view === 'room' ? 'closer' : 'room');
	};
	// ponytail: one traversal per frame, and only when the pointer has really
	// moved. Pointer events arrive far faster than frames, and the traversal
	// walks every triangle in all five GLBs. A dedicated low-poly pick mesh for
	// the cabinet is the upgrade if the room ever grows.
	let lastPick = { at: -1, x: 0, y: 0, hit: false };
	function pointsAtCabinet(event: MouseEvent) {
		const bounds = canvas.getBoundingClientRect();
		const x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
		const y = 1 - ((event.clientY - bounds.top) / bounds.height) * 2;
		const now = performance.now();
		if (
			now - lastPick.at < 16 &&
			Math.abs(x - lastPick.x) < 0.008 &&
			Math.abs(y - lastPick.y) < 0.008
		)
			return lastPick.hit;
		pointer.set(x, y);
		raycaster.setFromCamera(pointer, camera);
		// Cheap analytic reject first: most of the room is not the cabinet.
		const hit = raycaster.ray.intersectsBox(cabinet)
			? raycaster.intersectObjects(scene.children, true)[0]
			: undefined;
		lastPick = { at: now, x, y, hit: hit !== undefined && cabinet.containsPoint(hit.point) };
		return lastPick.hit;
	}
	// Focusable only while the scene is actually there to steer: an invisible
	// canvas must not keep a tab stop.
	function arm(on: boolean) {
		if (on) {
			canvas.setAttribute('tabindex', '0');
			canvas.setAttribute('role', 'button');
			canvas.removeAttribute('aria-hidden');
		} else {
			canvas.removeAttribute('tabindex');
			canvas.removeAttribute('role');
			canvas.setAttribute('aria-hidden', 'true');
		}
	}
	arm(true);
	canvas.addEventListener(
		'click',
		(event) => {
			// WebKit and Firefox do not focus an element on click just because it
			// has a tabindex. The canvas is the room's control, and Escape below
			// only hears keys dispatched inside the room, so it takes focus here.
			canvas.focus({ preventScroll: true });
			if (bookshelf.active) {
				bookshelf.background();
				return;
			}
			if (pointsAtCabinet(event)) bookshelf.enter();
			else step();
		},
		{ signal: events.signal },
	);
	canvas.addEventListener(
		'keydown',
		(event) => {
			if (event.key === 'Enter' || event.key === ' ') {
				event.preventDefault();
				step();
			}
		},
		{ signal: events.signal },
	);
	// Escape backs out from anywhere inside the room. The canvas takes focus on
	// click, so this is reachable after a pointer as well as after a tab.
	host.addEventListener(
		'keydown',
		(event) => {
			if (
				event.defaultPrevented ||
				bookshelf.active ||
				event.key !== 'Escape' ||
				view === 'room' ||
				!visible
			)
				return;
			changeView('room');
		},
		{ signal: events.signal },
	);
	host.addEventListener(
		'pointermove',
		(event) => {
			if (bookshelf.active || event.pointerType !== 'mouse') return;
			canvas.style.cursor = pointsAtCabinet(event) ? 'pointer' : '';
			if (motion.matches) return;
			const bounds = canvas.getBoundingClientRect();
			wantedOffset.set(
				((event.clientX - bounds.left) / bounds.width - 0.5) * 0.12,
				(0.5 - (event.clientY - bounds.top) / bounds.height) * 0.06,
			);
			requestDraw();
		},
		{ signal: events.signal },
	);
	host.addEventListener(
		'pointerleave',
		() => {
			wantedOffset.set(0, 0);
			requestDraw();
		},
		{ signal: events.signal },
	);
	motion.addEventListener(
		'change',
		() => {
			wantedOffset.set(0, 0);
			requestDraw();
		},
		{ signal: events.signal },
	);

	const resize = new ResizeObserver(() => {
		// The canvas is what is observed and what is drawn into; the wrapper
		// around it also holds the list, so its width is not the picture's.
		const width = canvas.clientWidth;
		const height = canvas.clientHeight;
		if (!width || !height) return;
		renderer.setSize(width, height, false);
		camera.aspect = width / height;
		camera.updateProjectionMatrix();
		if (shelfFrame) {
			// A trip already under way keeps the time it has left, so dragging an
			// edge cannot keep restarting the ease and leave the camera crawling.
			const remaining =
				transitionStart === null
					? 0
					: Math.max(0, transitionDuration - (performance.now() - transitionStart));
			frameShelf(shelfFrame, remaining);
		}
		requestDraw();
	});
	resize.observe(canvas);
	const intersection = new IntersectionObserver(([entry]) => {
		visible = entry?.isIntersecting ?? false;
		if (visible) requestDraw();
		else {
			cancelAnimationFrame(frame);
			frame = 0;
		}
	});
	intersection.observe(canvas);
	document.addEventListener(
		'visibilitychange',
		() => {
			if (document.hidden) {
				cancelAnimationFrame(frame);
				frame = 0;
			} else requestDraw();
		},
		{ signal: events.signal },
	);
	canvas.addEventListener(
		'webglcontextlost',
		(event) => {
			event.preventDefault();
			lost = true;
			bookshelf.fallback();
			arm(false);
			delete host.dataset.live;
			for (const hotspot of hotspots) {
				hotspot.hidden = false;
				hotspot.style.left = '';
				hotspot.style.top = '';
			}
			cancelAnimationFrame(frame);
			frame = 0;
		},
		{ signal: events.signal },
	);
	canvas.addEventListener(
		'webglcontextrestored',
		() => {
			lost = false;
			arm(view !== 'shelf');
			requestDraw();
			host.dataset.live = '';
			bookshelf.connect(frameShelf);
		},
		{ signal: events.signal },
	);

	function dispose() {
		if (!alive) return;
		alive = false;
		cancelAnimationFrame(frame);
		events.abort();
		resize.disconnect();
		intersection.disconnect();
		playing.disconnect();
		releaseAssets();
		renderer.dispose();
		renderer.forceContextLoss();
		arm(false);
		bookshelf.fallback();
		delete host.dataset.live;
	}

	addEventListener(
		'pagehide',
		(event) => {
			if (!event.persisted) dispose();
		},
		{ signal: events.signal },
	);
	renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
	camera.aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
	camera.updateProjectionMatrix();
	publish('room');
	host.dataset.live = '';
	bookshelf.pulls((isbn) => {
		if (!frame) lastTime = performance.now();
		reached = isbn;
		requestDraw();
	});
	bookshelf.connect(frameShelf);
	draw();
	return dispose;
}
