/** Display the baked room and frame its navigable bookshelf. */
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
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
		[shellUrl, furnitureUrl, objectsUrl, movingUrl, spinesUrl].map(async (url) => {
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

	/* A volume tips out of the row on its own, the way a finger on the top of a
	   book tips it: so it has to be its own object, both halves of it. The bake
	   joins the printed jackets into one atlas mesh and the leather bodies into
	   the room's objects, one bake and one draw each. Cut both back out against
	   the manifest Blender placed them from, and hang each pair on a pivot at
	   its bottom front edge, the edge a leaning book turns on. */
	const books = new Map<string, THREE.Group>();

	/** Split one baked mesh into the geometry each book owns and the rest of it.
	 *  `depth` is how far back from the spine face a book reaches: the jacket is
	 *  flat against that face, the body stands behind it. */
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
	const room = scene.getObjectByName('objects');
	if (atlas instanceof THREE.Mesh && room instanceof THREE.Mesh) {
		// Deep enough to take the whole body with the jacket, shallow enough to
		// leave the shelf it stands on and the records behind it where they are.
		const printed = slice(atlas, 0.006);
		const carcass = slice(room, 0.16);
		for (const slot of ordered) {
			const jacket = printed.owned.get(slot.isbn);
			const body = carcass.owned.get(slot.isbn);
			if (!jacket || !body) continue;
			const book = new THREE.Group();
			book.position.set(slot.x, slot.y, slot.z);
			for (const [geometry, material] of [
				[jacket, atlas.material],
				[body, room.material],
			] as const) {
				geometry.translate(-slot.x, -slot.y, -slot.z);
				geometries.add(geometry);
				book.add(new THREE.Mesh(geometry, material));
			}
			scene.add(book);
			books.set(slot.isbn, book);
		}
		// All of them or none: half a shelf of loose books over a shelf that still
		// draws them is worse than a row that cannot lean.
		if (books.size === ordered.length) {
			atlas.visible = false;
			// What the room keeps is drawn as its own mesh, not handed back to the
			// baked node: these vertices have that node's transform in them
			// already, and it would be applied to them a second time.
			room.visible = false;
			geometries.add(carcass.left);
			scene.add(new THREE.Mesh(carcass.left, room.material));
		} else {
			for (const book of books.values()) {
				scene.remove(book);
				for (const part of book.children)
					if (part instanceof THREE.Mesh) {
						geometries.delete(part.geometry);
						part.geometry.dispose();
					}
			}
			books.clear();
			carcass.left.dispose();
		}
		printed.left.dispose();
		for (const [isbn, geometry] of printed.owned) if (!books.has(isbn)) geometry.dispose();
		for (const [isbn, geometry] of carcass.owned) if (!books.has(isbn)) geometry.dispose();
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
	const TILT = THREE.MathUtils.degToRad(12);
	let reached: string | null = null;

	function pull(dt: number) {
		let moving = false;
		for (const [isbn, book] of books) {
			const wanted = isbn === reached ? TILT : 0;
			if (book.rotation.x === wanted) continue;
			const eased = motion.matches
				? wanted
				: book.rotation.x + (wanted - book.rotation.x) * (1 - Math.exp(-12 * dt));
			book.rotation.x = Math.abs(wanted - eased) < 0.0004 ? wanted : eased;
			moving = true;
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
		camera.lookAt(target);
		camera.updateMatrixWorld();
		renderer.render(scene, camera);
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
		const dt = Math.min((now - lastTime) / 1000, 0.05);
		lastTime = now;
		if (transitionStart !== null) {
			const t = motion.matches ? 1 : Math.min((now - transitionStart) / transitionDuration, 1);
			const eased = 1 - (1 - t) ** 3;
			position.lerpVectors(fromPosition, destination, eased);
			target.lerpVectors(fromTarget, destinationTarget, eased);
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
		shelfFrame = shelf;
		if (!shelf) {
			if (view === 'shelf') changeView('room');
			return;
		}
		camera.aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
		camera.updateProjectionMatrix();
		const distance =
			Math.max(shelf.height, shelf.width / camera.aspect) /
			(2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
		// Stepping along a row is a short pan; arriving at the shelves is a trip.
		const lateral = view === 'shelf' && shelf.view === 'row';
		travel(
			'shelf',
			shelfPosition.set(shelf.x, shelf.y, shelf.z + distance),
			shelfTarget.set(shelf.x, shelf.y, shelf.z),
			duration ?? (reframe ? 0 : lateral ? 320 : 750),
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
		if (isbn === reached) return;
		reached = isbn;
		requestDraw();
	});
	bookshelf.connect(frameShelf);
	draw();
	return dispose;
}
