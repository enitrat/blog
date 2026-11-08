import { Tile } from '@/ds';

/** PLACEHOLDER: the room agent replaces this with the port of RoomTile.astro and LivingRoom.astro. */
export function RoomTile({ order }: { order?: number }) {
	return <Tile order={order} label="Room" className="room-tile" />;
}
