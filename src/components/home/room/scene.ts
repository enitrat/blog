/** Display the baked room and frame its navigable bookshelf. */
import * as THREE from 'three';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import bookmarkUrl from '../../../assets/room/bookmark.glb?url';
import booksUrl from '../../../assets/room/books.glb?url';
import coversUrl from '../../../assets/room/covers.glb?url';
import furnitureUrl from '../../../assets/room/furniture.glb?url';
import movingUrl from '../../../assets/room/moving.glb?url';
import objectsUrl from '../../../assets/room/objects.glb?url';
import sheetsUrl from '../../../assets/room/sheets.glb?url';
import sheets from '../../../assets/room/sheets.json';
import shellUrl from '../../../assets/room/shell.glb?url';
// The printed jackets carry the only type in the room meant to be read, so they
// own an atlas instead of sharing one with nine square metres of leather.
import spinesUrl from '../../../assets/room/spines.glb?url';
import type { Bookshelf, Hint, Pointed } from './bookshelf';
import { BOOKSHELF_ANCHOR, CABINET_BOX, ordered, rows, type ShelfFrame } from './shelves';

const VIEWS = {
	room: { position: new THREE.Vector3(5.5, 5.7, 8.5), target: new THREE.Vector3(0, 1, -0.2) },
	closer: {
		position: new THREE.Vector3(0.9, 2.4, 4.8),
		target: new THREE.Vector3(0, 0.94, -1.05),
	},
	record: {
		position: new THREE.Vector3(0.98, 1.48, 0.55),
		target: new THREE.Vector3(), // above the platter, once it has loaded
	},
	// Seated: over the chair, looking down the leather at the two piles.
	desk: {
		position: new THREE.Vector3(1.1, 1.3, 0.72),
		target: new THREE.Vector3(1.72, 0.77, 0.72),
		fov: 40,
	},
};
const ANCHORS = {
	// The cabinet's ring sits on its books, and the bake owns where those are.
	bookshelf: new THREE.Vector3(BOOKSHELF_ANCHOR.x, BOOKSHELF_ANCHOR.y, BOOKSHELF_ANCHOR.z),
	record: new THREE.Vector3(), // above the platter, once it has loaded
	// On the top sheet of the nearer pile, where the bake put it.
	desk: new THREE.Vector3(sheets[0]?.x ?? 1.65, sheets[0]?.y ?? 0.77, sheets[0]?.z ?? 0.55),
};

type View = keyof typeof VIEWS | 'shelf';
/** The three things in the room that do something. */
const THINGS = ['bookshelf', 'record', 'desk'] as const;
type Thing = (typeof THINGS)[number];

/** Once per visit, not once per page: a flag in session storage, which may be
 *  missing or refuse, in which case the moment simply plays again. */
function once(key: string) {
	try {
		if (sessionStorage.getItem(key)) return false;
		sessionStorage.setItem(key, '1');
	} catch {}
	return true;
}

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
	// The static room groups ship Meshopt-compressed geometry.
	const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
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
			sheetsUrl,
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
	/** The annotated jackets' own materials, which breathe while browsing. */
	const breathing: THREE.MeshBasicMaterial[] = [];

	const bookmark = scene.getObjectByName('Bookmark');
	if (bookmark) bookmark.visible = false;
	for (const slot of ordered) {
		const jacket = scene.getObjectByName(`Book_${slot.isbn}`);
		const body = scene.getObjectByName(`Body_${slot.isbn}`);
		// Only an annotated volume is baked a front cover, because only it opens.
		// A note written since the last covers bake still opens, just without
		// its cover swinging; the build's asset check fails before that ships.
		const cover = scene.getObjectByName(`Cover_${slot.isbn}`);
		if (!jacket || !body) {
			releaseAssets();
			return null;
		}
		const book = new THREE.Group();
		book.position.set(slot.x, slot.y, slot.z);
		scene.add(book);
		book.attach(jacket);
		book.attach(body);
		if (cover) {
			book.attach(cover);
			covers.set(slot.isbn, cover);
		}
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
				breathing.push(surface.material);
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

	// The manuscripts: a baked node under each real link. A link whose sheet is
	// missing names a piece the bake never saw, and comes off the desk.
	const deskTargets = host.querySelector<HTMLElement>('[data-desk-targets]');
	const manuscripts = [
		...(deskTargets?.querySelectorAll<HTMLAnchorElement>('a[data-sheet]') ?? []),
	].flatMap((link) => {
		const sheet = sheets.find((entry) => entry.slug === link.dataset.sheet);
		const node = sheet && scene.getObjectByName(`Sheet_${sheet.slug}`);
		if (!sheet || !node) {
			link.remove();
			return [];
		}
		return [{ link, node, sheet, rest: node.position.y }];
	});

	const platter = scene.getObjectByName('Platter');
	const tonearm = scene.getObjectByName('Tonearm');
	if (!platter || !tonearm) {
		releaseAssets();
		return null;
	}
	// The platter's origin is the spindle axis, so the record's ring and close-up
	// follow the turntable wherever the bake puts it.
	const axis = platter.getWorldPosition(new THREE.Vector3());
	ANCHORS.record.set(axis.x, axis.y + 0.07, axis.z);
	VIEWS.record.target.set(axis.x, axis.y + 0.09, axis.z);

	/* The room's one highlight. Every surface inside a box warms, fading out a
	   little past its faces. It works on world position in the shader, so one
	   merged mesh with one baked atlas can still light a single object on it,
	   and nothing has to be rebaked to change what lights. Slots: the three
	   things, then whatever row or volume the shelves say is pointed at. */
	const glow = {
		glowMin: { value: Array.from({ length: 4 }, () => new THREE.Vector3()) },
		glowMax: { value: Array.from({ length: 4 }, () => new THREE.Vector3()) },
		glowSoft: { value: [0.08, 0.06, 0.08, 0.012] },
		glowAmount: { value: [0, 0, 0, 0] },
	};
	const boxes: Record<Thing, THREE.Box3> = {
		// Off the floor, which would otherwise light in a strip under the plinth.
		bookshelf: new THREE.Box3(
			new THREE.Vector3(CABINET_BOX.min[0], 0.05, CABINET_BOX.min[2]),
			new THREE.Vector3(...CABINET_BOX.max),
		),
		// The deck and its arm, down to the plinth they stand on.
		record: new THREE.Box3()
			.setFromObject(platter)
			.union(new THREE.Box3().setFromObject(tonearm))
			.expandByVector(new THREE.Vector3(0.05, 0.03, 0.05)),
		// The pedestal desk and its lamp, as bake-room.mjs stands it; Blender's
		// (x, y, z) is the glTF's (x, z, -y).
		desk: new THREE.Box3(new THREE.Vector3(1.4, 0.05, -0.11), new THREE.Vector3(2.04, 1.1, 1.21)),
	};
	boxes.record.min.y -= 0.05;
	THINGS.forEach((thing, index) => {
		glow.glowMin.value[index].copy(boxes[thing].min);
		glow.glowMax.value[index].copy(boxes[thing].max);
	});
	// Aiming need not be exact: the pick volumes are a little larger.
	const reach = Object.fromEntries(
		THINGS.map((thing) => [thing, boxes[thing].clone().expandByScalar(0.03)]),
	) as Record<Thing, THREE.Box3>;
	for (const material of materials) {
		material.onBeforeCompile = (shader) => {
			Object.assign(shader.uniforms, glow);
			shader.vertexShader = shader.vertexShader
				.replace('#include <common>', '#include <common>\nvarying vec3 vGlowWorld;')
				.replace(
					'#include <project_vertex>',
					'#include <project_vertex>\nvGlowWorld = (modelMatrix * vec4(transformed, 1.0)).xyz;',
				);
			shader.fragmentShader = shader.fragmentShader
				.replace(
					'#include <common>',
					`#include <common>
varying vec3 vGlowWorld;
uniform vec3 glowMin[4];
uniform vec3 glowMax[4];
uniform float glowSoft[4];
uniform float glowAmount[4];`,
				)
				.replace(
					'#include <map_fragment>',
					`#include <map_fragment>
float glow = 0.0;
for (int i = 0; i < 4; i++) {
	vec3 outside = max(max(glowMin[i] - vGlowWorld, vGlowWorld - glowMax[i]), 0.0);
	glow += glowAmount[i] * (1.0 - smoothstep(0.0, glowSoft[i], length(outside)));
}
// Lamplight, not a flash: the baked colour lifts, reds and ambers most, and
// a little warm light reaches even the surfaces the bake left in shadow.
diffuseColor.rgb = diffuseColor.rgb * (1.0 + glow * vec3(0.8, 0.56, 0.3)) + glow * vec3(0.03, 0.016, 0.004);`,
				);
		};
		material.customProgramCacheKey = () => 'room-glow';
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
	renderer.setClearColor(0x171513, 1);
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
	/** A pan of camera and aim together: the arrival hint that there is more. */
	const nudge = new THREE.Vector3();
	const wantedNudge = new THREE.Vector3();
	let nudgeHint: Hint = null;
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
	/** What the pointer is on, and what keyboard focus is on: either lights it. */
	let hovered: Thing | null = null;
	let focused: Thing | null = null;
	let pointedWanted = 0;
	let breathed = false;
	/** 1 when the tonearm has just been reached for, running down to 0. */
	let twitch = 0;

	// The platter and the arm are their own nodes in moving.glb, turning about
	// their own axis. Blender's Z became the node's Y in the glTF conversion.
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
	/** The manuscript a reader's hand is over, lifting from the pile. */
	let lifted: string | null = null;
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

	/** A reached-for sheet rises a little off the pile and its head comes up,
	 *  the way a page is picked up by its far edge before it is read. */
	function lift(dt: number) {
		let moving = false;
		for (const { node, sheet, rest } of manuscripts) {
			const up = lifted === sheet.slug && !motion.matches;
			const y = rest + (up ? 0.02 : 0);
			const tilt = up ? 0.07 : 0;
			node.position.y = THREE.MathUtils.damp(node.position.y, y, 14, dt);
			node.rotation.z = THREE.MathUtils.damp(node.rotation.z, tilt, 14, dt);
			if (Math.abs(node.position.y - y) < 0.0001) node.position.y = y;
			if (Math.abs(node.rotation.z - tilt) < 0.0004) node.rotation.z = tilt;
			moving ||= node.position.y !== y || node.rotation.z !== tilt;
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
		// Reached for while parked, the arm lifts a hair toward the record and
		// settles back: the tell that it will play.
		twitch = motion.matches ? 0 : Math.max(0, twitch - dt / 0.7);
		if (tonearm) {
			tonearm.rotation.y = CUED * cue - 0.05 * Math.sin(Math.PI * twitch);
			tonearm.rotation.x = DROP * cue;
		}
		return spin !== 0 || cue !== wanted || twitch > 0;
	}

	/** Seen from across the room, each thing keeps a low lamplight that rises
	 *  and falls, out of step with the others, so what can be clicked is plain
	 *  without searching for it. Reduced motion holds it still, mid-range. */
	function idle(now: number, index: number) {
		if (bookshelf.active || (view !== 'room' && view !== 'closer')) return 0;
		if (motion.matches) return 0.35;
		return 0.35 + 0.22 * Math.sin((now / 1000) * ((2 * Math.PI) / 2.2) - index * 1.9);
	}

	function light(now: number, dt: number) {
		let moving = false;
		const amounts = glow.glowAmount.value;
		const wanted = [
			...THINGS.map((thing, index) =>
				hovered === thing || focused === thing ? 1 : idle(now, index),
			),
			pointedWanted,
		];
		wanted.forEach((to, index) => {
			const rate = index === 3 ? 18 : 10;
			amounts[index] = motion.matches ? to : THREE.MathUtils.damp(amounts[index], to, rate, dt);
			if (Math.abs(amounts[index] - to) < 0.002) amounts[index] = to;
			moving ||= amounts[index] !== to;
		});
		// ponytail: breathing redraws the whole room every frame while a row is
		// framed. A pulse baked into the jacket's own shader would spare the rest.
		const breathe =
			view === 'shelf' && !motion.matches && !bookshelf.openBook && breathing.length > 0;
		if (breathe || breathed) {
			const k = breathe ? 1 + 0.05 * Math.sin((now / 1000) * ((2 * Math.PI) / 3.2)) : 1;
			for (const material of breathing) material.color.setRGB(1.12 * k, 1.07 * k, 1.02 * k);
			breathed = breathe;
		}
		// ponytail: the idle glow keeps the room view redrawing every frame
		// while it is on screen, same as the breathing jackets.
		const idling = !motion.matches && idle(now, 0) > 0;
		return moving || breathe || idling;
	}

	function draw() {
		camera.position.copy(position);
		camera.position.x += offset.x;
		camera.position.y += offset.y;
		camera.position.add(nudge);
		lookAt.copy(target).add(nudge);
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
			if (name !== 'bookshelf' && name !== 'record' && name !== 'desk') continue;
			projected.copy(ANCHORS[name]).project(camera);
			// A close view leaves the other object off frame: take its ring out of
			// the picture and out of the tab order rather than parking it outside.
			// The wide mix credit needs more room than a ring before it fits.
			const slack = hotspot.classList.contains('living-room__mix') ? 0.6 : 0.98;
			// At reading distance the ring has nothing left to offer, so it steps aside.
			// Seated at the desk, the sheets themselves are the targets.
			hotspot.hidden =
				bookshelf.active ||
				(name === 'desk' && view === 'desk') ||
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
		nudge.lerp(wantedNudge, motion.matches ? 1 : 1 - Math.exp(-11 * dt));
		if (nudge.distanceToSquared(wantedNudge) < 0.0000001) nudge.copy(wantedNudge);
		const turning = turntable(dt);
		const pulling = pull(dt);
		const lifting = lift(dt);
		const lighting = light(now, dt);
		draw();
		if (
			transitionStart !== null ||
			turning ||
			pulling ||
			lifting ||
			lighting ||
			!offset.equals(wantedOffset) ||
			!nudge.equals(wantedNudge)
		)
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
		hover(null);
		view = next;
		transitionStart = duration > 0 ? performance.now() : null;
		host.toggleAttribute('data-traveling', transitionStart !== null);
		wantedOffset.set(0, 0);
		offset.set(0, 0);
		wantedNudge.set(0, 0, 0);
		nudge.set(0, 0, 0);
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
		const to = VIEWS[next];
		travel(next, to.position, to.target, 750, 'fov' in to ? to.fov : 23.83);
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
		placeManuscripts();
		// The first row a visitor reaches drifts a little toward what lies past
		// its edge and springs back: the only sign, with no arrows, that it goes on.
		if (view === 'shelf' && shelfFrame?.view === 'row' && nudgeHint && !motion.matches) {
			const hint = nudgeHint;
			nudgeHint = null;
			if (once('room-nudged')) {
				wantedNudge.set(hint === 'right' ? 0.035 : 0, hint === 'down' ? -0.03 : 0, 0);
				setTimeout(() => {
					wantedNudge.set(0, 0, 0);
					requestDraw();
				}, 320);
				requestDraw();
			}
		}
	}

	/** Point at a thing: it lights, names itself after a moment, and gives
	 *  its own small tell. */
	function hover(next: Thing | null) {
		if (next === hovered) return;
		hovered = next;
		canvas.style.cursor = next ? 'pointer' : '';
		if (next === 'record' && !cueing) twitch = 1;
		// From across the room, the top sheet stirs on the pile.
		if (view !== 'desk') lifted = next === 'desk' ? (manuscripts[0]?.sheet.slug ?? null) : null;
		const spot = next ? spotFor(next) : undefined;
		bookshelf.label(spot?.textContent?.trim() ?? '', spot);
		if (!frame) lastTime = performance.now();
		requestDraw();
	}

	const spotFor = (thing: Thing) =>
		hotspots.find(
			(spot) => spot.dataset.anchor === thing && !spot.classList.contains('living-room__mix'),
		);

	/** Lay each link over the exposed head of its sheet, as the desk view shows
	 *  it. Anywhere else the pile is scenery, and the links leave the tab order. */
	function placeManuscripts() {
		if (!deskTargets) return;
		deskTargets.hidden = view !== 'desk';
		if (view !== 'desk') return;
		for (const { link, sheet } of manuscripts) {
			let left = Infinity;
			let top = Infinity;
			let right = -Infinity;
			let bottom = -Infinity;
			let visible = true;
			for (const [x, y, z] of sheet.band) {
				projected.set(x, y, z).project(camera);
				visible &&= projected.z > -1 && projected.z < 1;
				left = Math.min(left, projected.x);
				right = Math.max(right, projected.x);
				top = Math.min(top, -projected.y);
				bottom = Math.max(bottom, -projected.y);
			}
			link.hidden = !visible;
			link.style.left = `${(left * 0.5 + 0.5) * 100}%`;
			link.style.top = `${(top * 0.5 + 0.5) * 100}%`;
			link.style.width = `${(right - left) * 50}%`;
			link.style.height = `${(bottom - top) * 50}%`;
		}
	}

	const sameShelf = (a: ShelfFrame, b: ShelfFrame) =>
		a.view === b.view &&
		a.x === b.x &&
		a.y === b.y &&
		a.z === b.z &&
		a.width === b.width &&
		a.height === b.height;

	/** The shelves' listener: where to look, and which way there is more. */
	function follow(shelf: ShelfFrame | null, hint: Hint) {
		nudgeHint = hint;
		frameShelf(shelf);
	}

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
		// The cabinet's straight-on sightline runs through the sofa. Approach from
		// the open side while keeping the same camera-to-shelf framing distance.
		const cameraX = shelf.view === 'cabinet' ? 1.5 : shelf.x;
		const depth = Math.sqrt(distance ** 2 - (cameraX - shelf.x) ** 2);
		// Stepping along a row is a short pan; arriving at the shelves is a trip.
		travel(
			'shelf',
			shelfPosition.set(cameraX, shelf.y + (shelf.view === 'row' ? 0.06 : 0), shelf.z + depth),
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

	// The desk link leads to the writing index on a page without a room; with
	// one, it pulls out the chair instead, and the sheets carry the links.
	host.querySelector<HTMLAnchorElement>('a[data-anchor="desk"]')?.addEventListener(
		'click',
		(event) => {
			if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button !== 0)
				return;
			event.preventDefault();
			if (!bookshelf.active) changeView('desk');
		},
		{ signal: events.signal },
	);
	for (const { link, sheet } of manuscripts) {
		const reach = () => {
			if (!frame) lastTime = performance.now();
			lifted = sheet.slug;
			bookshelf.label(`${link.textContent?.trim() ?? sheet.title} · Read`);
			requestDraw();
		};
		const release = () => {
			if (lifted === sheet.slug) lifted = null;
			bookshelf.label('');
			requestDraw();
		};
		link.addEventListener('pointerenter', reach, { signal: events.signal });
		link.addEventListener('focus', reach, { signal: events.signal });
		link.addEventListener('pointerleave', release, { signal: events.signal });
		link.addEventListener('blur', release, { signal: events.signal });
	}

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
	// walks every triangle in every loaded GLB. Low-poly pick meshes are the
	// upgrade if the room ever grows.
	let lastPick: { at: number; x: number; y: number; hit: Thing | null } = {
		at: -1,
		x: 0,
		y: 0,
		hit: null,
	};
	/** The thing under the pointer that this view lets you act on. */
	function pick(event: MouseEvent): Thing | null {
		if (transitionStart !== null || bookshelf.active) return null;
		const allowed: readonly Thing[] =
			view === 'room' || view === 'closer' ? THINGS : view === 'record' ? ['record'] : [];
		if (!allowed.length) return null;
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
		// Cheap analytic reject first: most of the room is none of them.
		const near = allowed.filter((thing) => raycaster.ray.intersectsBox(reach[thing]));
		const point = near.length
			? raycaster.intersectObjects(scene.children, true)[0]?.point
			: undefined;
		const hit = (point && near.find((thing) => reach[thing].containsPoint(point))) || null;
		lastPick = { at: now, x, y, hit };
		return hit;
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
			const thing = pick(event);
			if (thing === 'bookshelf') bookshelf.enter();
			// The hidden button carries playback as well as the camera.
			else if (thing === 'record') spotFor('record')?.click();
			else if (thing === 'desk') changeView('desk');
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
			hover(pick(event));
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
			hover(null);
			wantedOffset.set(0, 0);
			requestDraw();
		},
		{ signal: events.signal },
	);
	// Keyboard focus on a thing's control lights the thing and names it at once.
	host.addEventListener(
		'focusin',
		(event) => {
			const spot = (event.target as Element).closest?.<HTMLElement>('[data-anchor]');
			const thing = THINGS.find((name) => spot && spotFor(name) === spot) ?? null;
			if (thing === focused) return;
			focused = thing;
			if (spot && thing) bookshelf.label(spot.textContent?.trim() ?? '', spot, true);
			if (!frame) lastTime = performance.now();
			requestDraw();
		},
		{ signal: events.signal },
	);
	host.addEventListener(
		'focusout',
		() => {
			if (!focused) return;
			focused = null;
			bookshelf.label('');
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
			if (deskTargets) deskTargets.hidden = true;
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
			bookshelf.connect(follow);
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
	bookshelf.points((pointed: Pointed) => {
		// Close up, a spine or a row needs far less light to stand out.
		pointedWanted = pointed ? 0.45 : 0;
		if (pointed) {
			const slots =
				'row' in pointed
					? (rows.find(({ row }) => row === pointed.row)?.books ?? [])
					: ordered.filter((slot) => slot.isbn === pointed.isbn);
			if (slots.length) {
				const first = slots[0];
				const last = slots[slots.length - 1];
				glow.glowMin.value[3].set(first.x - first.width / 2, first.y, first.z - 0.2);
				glow.glowMax.value[3].set(
					last.x + last.width / 2,
					first.y + Math.max(...slots.map((slot) => slot.height)),
					first.z + 0.2,
				);
			}
		}
		if (!frame) lastTime = performance.now();
		requestDraw();
	});
	bookshelf.connect(follow);
	draw();
	performance.mark('room-live');
	// A tap on the poster that arrived before the room did is carried out now.
	const intent = host.dataset.intent;
	delete host.dataset.intent;
	if (intent === 'record' || intent === 'desk') changeView(intent);
	return dispose;
}
