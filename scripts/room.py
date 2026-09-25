"""Original living-room assets. Run through `bun run room:bake` with Blender 4.5.

Dimensions are metres, Z is up. Cycles resolves the materials and illumination;
the exported glTF materials are unlit. No reference-site assets are included.
"""
import bmesh
import bpy
import json
import numpy as np
import math
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from mathutils import Matrix, Vector

started = time.perf_counter()
args = sys.argv[sys.argv.index('--') + 1:]
work = Path(args[0])
out = Path(args[1])
samples = int(args[2])
size = int(args[3])
preview_only = '--preview' in args
minimal_art = '--minimal-art' in args
# room-targets.mjs owns the groups: their names, in bake order, and which bake alone.
group_names = json.loads((work / 'groups.json').read_text())
isolated_groups = set(group_names['isolated'])
group_names = group_names['all']
only_index = args.index('--only')
targets = set(args[only_index + 1].split(','))
if bpy.app.version != (4, 5, 3):
    raise RuntimeError(f'Blender 4.5.3 required; found {bpy.app.version_string}')
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = samples
scene.cycles.use_denoising = True
scene.cycles.max_bounces = 8
scene.cycles.diffuse_bounces = 5
scene.cycles.glossy_bounces = 3
scene.cycles.use_adaptive_sampling = True
scene.cycles.adaptive_threshold = 0.025
try:
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'METAL'
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == 'METAL'
    scene.cycles.device = 'GPU' if any(device.use for device in prefs.devices) else 'CPU'
except (TypeError, RuntimeError):
    scene.cycles.device = 'CPU'
# A CPU bake is far slower at the same samples; say which one ran.
print('DEVICE', scene.cycles.device, flush=True)
scene.world.use_nodes = True
scene.world.node_tree.nodes['Background'].inputs[0].default_value = (0.018, 0.028, 0.045, 1)
scene.world.node_tree.nodes['Background'].inputs[1].default_value = 0.10
scene.view_settings.view_transform = 'AgX'
scene.view_settings.look = 'AgX - Medium High Contrast'
scene.view_settings.exposure = 0.30
scene.render.image_settings.file_format = 'PNG'
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.resolution_percentage = 100

groups = {name: {} for name in group_names}
group = 'shell'
# Parts of `moving` keep their own transform so the runtime can turn them, and
# their origin sits on the axis they turn about.
part = None
pivots = {}

def linear(hex_color):
    c = [int(hex_color[i:i+2], 16) / 255 for i in (1, 3, 5)]
    return tuple(v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in c)

def material(name, color, rough=.6, texture=None, metal=0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get('Principled BSDF')
    base = linear(color)
    bsdf.inputs['Base Color'].default_value = (*base, 1)
    bsdf.inputs['Roughness'].default_value = rough
    bsdf.inputs['Metallic'].default_value = metal
    if texture:
        coord = nodes.new('ShaderNodeTexCoord')
        mapping = nodes.new('ShaderNodeVectorMath')
        mapping.operation = 'MULTIPLY'
        mapping.inputs[1].default_value = (
            (3, 55, 4) if texture == 'wood' else
            (5, 5, 5) if texture == 'plaster' else
            (180, 180, 180)
        )
        links.new(coord.outputs['Generated'], mapping.inputs[0])
        noise = nodes.new('ShaderNodeTexNoise')
        noise.inputs['Scale'].default_value = 2.5
        noise.inputs['Detail'].default_value = 3
        noise.inputs['Roughness'].default_value = .7
        links.new(mapping.outputs[0], noise.inputs['Vector'])
        ramp = nodes.new('ShaderNodeValToRGB')
        ramp.color_ramp.elements[0].position = .2
        dark, light = (.78, 1.08) if texture == 'plaster' else (.65, 1.2)
        ramp.color_ramp.elements[0].color = (*(v * dark for v in base), 1)
        ramp.color_ramp.elements[1].position = .8
        ramp.color_ramp.elements[1].color = (*(min(1, v * light) for v in base), 1)
        links.new(noise.outputs['Fac'], ramp.inputs[0])
        links.new(ramp.outputs[0], bsdf.inputs['Base Color'])
        bump = nodes.new('ShaderNodeBump')
        bump.inputs['Strength'].default_value = .10 if texture == 'plaster' else .18
        bump.inputs['Distance'].default_value = .0015 if texture == 'wood' else .003 if texture == 'plaster' else .0007
        links.new(noise.outputs['Fac'], bump.inputs['Height'])
        links.new(bump.outputs[0], bsdf.inputs['Normal'])
        if texture == 'velvet':
            bsdf.inputs['Sheen Weight'].default_value = .18
    return mat

plaster = material('Deep green wall', '#13271f', .92, 'plaster')
walnut = material('Oiled walnut', '#321d14', .42, 'wood')
oak = material('Smoked oak', '#44291a', .48, 'wood')
darkwood = material('Walnut shadow', '#1b100b', .65, 'wood')
black = material('Charcoal enamel', '#242725', .35)
metal = material('Brushed aluminium', '#aaa9a2', .3, metal=.2)
brass = material('Aged brass', '#8b6936', .28, metal=.68)
cream = material('Ivory paper', '#ddd3b5', .85)
linen = material('Oatmeal linen', '#b99b6c', .9, 'fabric')
ceramic = material('Porcelain', '#bdbca9', .24)
green = material('Rubber plant leaves', '#405b31', .45)
soil = material('Potting soil', '#302821', 1)
terra = material('Terracotta', '#a66a48', .9, 'fabric')
rug_red = material('Kilim madder', '#6f2526', .98, 'fabric')
rug_blue = material('Kilim deep green', '#203b33', .98, 'fabric')
rug_cream = material('Kilim tobacco', '#ad875e', .98, 'fabric')

def finish(obj, name, mat, bevel=0):
    obj.name = name
    obj.data.materials.append(mat)
    if bevel:
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
        mod = obj.modifiers.new('Soft manufactured edges', 'BEVEL')
        mod.width = bevel
        mod.segments = 1 if bevel <= .003 else (3 if bevel < .03 else 5)
        mod.harden_normals = True
        mod = obj.modifiers.new('Weighted corner normals', 'WEIGHTED_NORMAL')
        mod.keep_sharp = True
    groups[group].setdefault(part or group, []).append(obj)
    return obj

# Primitives are built with bmesh rather than bpy.ops.mesh.primitive_*: every
# operator call refreshes the whole view layer, which made building the room
# take half a minute. Each one matches its operator's vertices, faces and UVs.
def mesh_object(name, bm, at, rotation=(0, 0, 0)):
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = at
    obj.rotation_euler = rotation
    scene.collection.objects.link(obj)
    return obj

def new_bmesh():
    bm = bmesh.new()
    bm.loops.layers.uv.new('UVMap')
    return bm

def box(name, at, dims, mat, bevel=.008, rotation=0):
    bm = new_bmesh()
    bmesh.ops.create_cube(bm, size=1, matrix=Matrix.Diagonal((*dims, 1)), calc_uvs=True)
    return finish(mesh_object(name, bm, at, (0, 0, rotation)), name, mat, bevel)

def cylinder(name, at, radius, depth, mat, vertices=48, radius2=None):
    bm = new_bmesh()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=vertices,
                          radius1=radius if radius2 is None else radius2, radius2=radius,
                          depth=depth, calc_uvs=True)
    obj = finish(mesh_object(name, bm, at), name, mat, .002)
    for p in obj.data.polygons:
        p.use_smooth = len(p.vertices) == 4
    return obj

def sphere(name, at, scale, mat):
    bm = new_bmesh()
    bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=1,
                              matrix=Matrix.Diagonal((*scale, 1)), calc_uvs=True)
    obj = mesh_object(name, bm, at)
    for p in obj.data.polygons:
        p.use_smooth = True
    return finish(obj, name, mat)

def torus(name, at, major_radius, minor_radius, major_segments, minor_segments, mat, rotation=(0, 0, 0)):
    """The rings and winding of bpy.ops.mesh.primitive_torus_add, without its UVs."""
    bm = new_bmesh()
    verts = []
    for i in range(major_segments):
        a = 2 * math.pi * i / major_segments
        for j in range(minor_segments):
            b = 2 * math.pi * j / minor_segments
            d = major_radius + minor_radius * math.cos(b)
            verts.append(bm.verts.new((d * math.cos(a), d * math.sin(a), minor_radius * math.sin(b))))
    for i in range(major_segments):
        ring, next_ring = i * minor_segments, (i + 1) % major_segments * minor_segments
        for j in range(minor_segments):
            k = (j + 1) % minor_segments
            bm.faces.new((verts[ring + j], verts[next_ring + j], verts[next_ring + k], verts[ring + k]))
    return finish(mesh_object(name, bm, at, rotation), name, mat)

def tube(name, points, radius, mat):
    curve = bpy.data.curves.new(name, 'CURVE')
    curve.dimensions = '3D'
    curve.resolution_u = 3 if radius < .003 else 8
    curve.bevel_depth = radius
    curve.bevel_resolution = 0 if radius < .003 else 2
    curve.use_fill_caps = True
    spline = curve.splines.new('BEZIER')
    spline.bezier_points.add(len(points)-1)
    for p, co in zip(spline.bezier_points, points):
        p.co = co
        p.handle_left_type = p.handle_right_type = 'AUTO'
    obj = bpy.data.objects.new(name, curve)
    scene.collection.objects.link(obj)
    return finish(obj, name, mat)

def plane_image(name, at, width, height, path, flat=None):
    """An upright print facing -y, or, given a yaw, one lying flat with its top edge turned that way."""
    mat = material(name, '#ffffff', .8)
    if path.exists():
        tex = mat.node_tree.nodes.new('ShaderNodeTexImage')
        tex.image = bpy.data.images.load(str(path))
        mat.node_tree.links.new(tex.outputs['Color'], mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'])
    elif not minimal_art or group in targets:
        raise FileNotFoundError(path)
    bm = new_bmesh()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=.5,
                          matrix=Matrix.Diagonal((width, height, 1, 1)), calc_uvs=True)
    obj = mesh_object(name, bm, at, (math.pi/2, 0, 0) if flat is None else (0, 0, flat))
    return finish(obj, name, mat)

# A dark, panelled cutaway room. Short parquet blocks give the floor scale in
# both the overview and the close desk view without requiring a texture asset.
box('Floating walnut foundation', (0, 0, -.10), (4.3, 3.1, .18), darkwood, .035)
floor_mats = [material('Parquet '+str(i), c, .48, 'wood') for i, c in enumerate(
    ['#342016', '#3c2519', '#482e1e', '#2d1b14', '#4b3020'])]
for row in range(15):
    y = -1.44 + row * .205
    for col in range(9):
        x = -1.90 + col * .46
        offset = .08 if row % 2 else 0
        box('Staggered parquet block', (x+offset, y, .003), (.448, .198, .025),
            floor_mats[(row*5+col*3)%len(floor_mats)], .0015)
box('Back wall', (0, 1.52, 1.28), (4.3, .09, 2.56), plaster, .012)
box('Left wall', (-2.105, .45, 1.28), (.09, 2.14, 2.56), plaster, .012)
box('Back skirting', (0, 1.452, .09), (4.18, .045, .17), darkwood, .005)
box('Left skirting', (-2.041, .45, .09), (.045, 2.03, .17), darkwood, .005)
box('Back picture rail', (1.88, 1.445, 1.72), (.45, .055, .065), walnut, .008)
box('Back crown', (0, 1.442, 2.48), (4.22, .075, .11), darkwood, .014)
box('Left picture rail', (-2.038, .45, 1.72), (.055, 2.02, .065), walnut, .008)
box('Left crown', (-2.035, .45, 2.48), (.075, 2.06, .11), darkwood, .014)

# Shallow wall mouldings catch the warm light but stay behind every interaction.
for x in [1.68, 2.08]:
    box('Panel stile', (x, 1.437, .91), (.035, .035, 1.47), walnut, .004)
for z in [.20, 1.63]:
    box('Panel rail', (1.88, 1.437, z), (.45, .035, .035), walnut, .004)

# An arched night window on the left wall. It is a dark recess rather than a
# boolean opening, which renders the same from the room camera at far lower cost.
night = material('Rainy night glass', '#071413', .2)
box('Window dark recess', (-2.045, .36, 1.38), (.014, .94, 1.66), night, .015)
for y in [-.12, .36, .84]:
    box('Window vertical frame', (-2.025, y, 1.38), (.035, .032, 1.66), darkwood, .005)
for z in [.58, 1.14, 1.70, 2.21]:
    box('Window horizontal frame', (-2.025, .36, z), (.035, .94, .032), darkwood, .005)
box('Window sill', (-1.99, .36, .53), (.16, 1.08, .08), walnut, .012)
window_arch = [(-2.023, .36 + .47*math.cos(a), 1.72 + .47*math.sin(a))
               for a in [i*math.pi/16 for i in range(17)]]
tube('Window arched head', window_arch, .025, darkwood)

group = 'furniture'
# Bookcase: thinner shelves and a recessed back give each bay depth. Its
# dimensions come from the manifest the browser also reads, so the frame, anchor
# and pick volume there cannot drift from this box. The manifest uses the
# runtime's axes (Y up, depth along -Z), so its front plane comes back as +y.
cabinet = json.loads((work / 'cabinet.json').read_text())
cx, cw, cd = cabinet['x'], cabinet['width'], cabinet['depth']
cy = -cabinet['face'] + cd / 2
crown, plinth = .036, .14
box('Bookcase inset back', (cx, cy+.18, .9), (cw-.035, .018, 1.57), darkwood, .003)
for x in [cx-cw/2+.013, cx+cw/2-.013]:
    box('Bookcase side', (x, cy, .89), (.027, cd, 1.59), walnut)
box('Bookcase crown', (cx, cy, cabinet['ceiling'] - crown/2), (1.10, .42, crown), walnut)
box('Bookcase plinth', (cx, cy, cabinet['floor'] + plinth/2), (.99, .34, plinth), darkwood)
shelves = cabinet['shelves']
for z in shelves:
    box('Shelf with rounded lip', (cx, cy, z-.013), (1.03, .39, .027), walnut, .005)
    box('Bookcase shelf nosing', (cx, 1.073, z-.005), (1.07, .035, .045), walnut, .006)
for x in [cx-cw/2-.005, cx+cw/2+.005]:
    box('Bookcase face stile', (x, 1.073, .92), (.055, .035, 1.66), walnut, .006)

# Join the active cabinet and the listening console into one wall of millwork.
# These parts are only the surround, so redesigning them cannot move a book target.
box('Library left pilaster', (-1.79, 1.28, 1.22), (.14, .28, 2.36), darkwood, .008)
box('Library centre pilaster', (-.49, 1.28, 1.22), (.12, .28, 2.36), darkwood, .008)
box('Library right pilaster', (1.57, 1.28, 1.22), (.14, .28, 2.36), darkwood, .008)
box('Library entablature', (-.11, 1.29, 2.36), (3.52, .31, .16), walnut, .012)
box('Library crown cap', (-.11, 1.28, 2.47), (3.65, .36, .08), darkwood, .012)
box('Library dentil rail', (-.11, 1.105, 2.31), (3.54, .035, .055), darkwood, .006)
for x in [-1.79, -.49, 1.57]:
    for offset in [-.035, .035]:
        box('Pilaster flute', (x+offset, 1.125, 1.31), (.012, .018, 1.82), walnut, .004)
    box('Pilaster capital', (x, 1.115, 2.25), (.18, .05, .10), walnut, .008)
    box('Pilaster foot block', (x, 1.115, .20), (.18, .05, .20), walnut, .006)

# Upper shelves over the record console turn the separate pieces into a built-in.
for z in [1.13, 2.30]:
    box('Listening shelf', (.54, 1.31, z), (1.94, .34, .045), walnut, .006)
    box('Listening shelf nosing', (.54, 1.125, z-.002), (2.00, .035, .065), walnut, .006)
for z in [1.53, 1.93]:
    box('Listening shelf', (-.19, 1.31, z), (.46, .34, .045), walnut, .006)
    box('Listening shelf', (1.27, 1.31, z), (.46, .34, .045), walnut, .006)
    box('Listening shelf nosing', (-.19, 1.125, z), (.49, .035, .055), walnut, .005)
    box('Listening shelf nosing', (1.27, 1.125, z), (.49, .035, .055), walnut, .005)
box('Listening cabinet back', (.54, 1.46, 1.72), (1.94, .025, 1.18), darkwood, .003)
for x in [-.43, .54, 1.51]:
    box('Listening shelf divider', (x, 1.31, 1.72), (.045, .34, 1.18), walnut, .005)

# A 1970s walnut credenza, with sliding fronts, finger pulls and tapered legs.
hx, hy = .68, 1.15
for x in [-.03, 1.39]:
    for y in [.99, 1.33]:
        cylinder('Credenza tapered foot', (x, y, .12), .022, .22, walnut, 16, .015)
box('Credenza carcass', (hx, hy, .42), (1.66, .47, .45), darkwood)
box('Credenza top', (hx, hy, .665), (1.72, .51, .04), walnut, .012)
for x in [.26, 1.09]:
    box('Sliding walnut door', (x, .904, .435), (.812, .025, .387), walnut, .005)
    for y in [.294, .576]:
        box('Credenza door rail', (x, .885, y), (.70, .018, .025), darkwood, .003)
    for edge in [x-.35, x+.35]:
        box('Credenza door stile', (edge, .885, .435), (.025, .018, .30), darkwood, .003)
    cylinder('Credenza brass pull', (x+.30, .875, .455), .012, .018, brass, 24).rotation_euler.x = math.pi/2
box('Credenza lower rail', (hx, .91, .221), (1.64, .035, .028), walnut)

# A writer's desk from the century before last: a fumed-oak pedestal desk with
# a leather writing surface, turned corner columns, fielded panels and brass
# swan-neck pulls. It stands along the open side of the room. The drawers face
# the chair; the camera reads the panelled back, the end and the leather top.
fumed = material('Fumed oak', '#352016', .40, 'wood')
fumed_shadow = material('Fumed oak shadow', '#25160f', .55, 'wood')
hide = material('Bottle-green writing leather', '#2b3f33', .55, 'fabric')
chair_leather = material('Burgundy chair leather', '#4a1717', .52, 'fabric')
# The bake script owns where the desk stands, so the manuscripts it lays out land on it.
desk = json.loads((work / 'desk.json').read_text())
dx, dy, dz = desk['centre']  # centre, underside of the top
for py in [dy-.45, dy+.45]:
    box('Pedestal plinth', (dx, py, .035), (.585, .425, .07), fumed_shadow, .004)
    box('Pedestal carcass', (dx, py, .3975), (.56, .40, .655), fumed, .004)
    # Three graduated drawers, the deepest at the bottom, each with a bail pull.
    z = .075 + .012
    for h in (.222, .20, .17):
        face = dx - .28 - .005
        box('Drawer front', (face, py, z + h/2), (.01, .36, h), fumed, .003)
        tube('Swan-neck brass pull', [(face-.008, py-.035, z+h/2-.012), (face-.018, py, z+h/2+.004),
                                      (face-.008, py+.035, z+h/2-.012)], .003, brass)
        z += h + .012
    # Fielded panels on the back and both ends; the kneehole shows the inner end.
    box('Fielded back panel', (dx+.28+.004, py, .40), (.008, .30, .56), fumed, .005)
    for sign in [-1, 1]:
        box('Fielded end panel', (dx, py+sign*(.20+.004), .40), (.44, .008, .56), fumed, .005)
    # Turned quarter columns at every corner, with a ring at each end.
    for x in [dx-.27, dx+.27]:
        for y in [py-.19, py+.19]:
            cylinder('Turned corner column', (x, y, .3975), .02, .655, fumed, 24)
            cylinder('Column capital', (x, y, .705), .024, .02, fumed_shadow, 24)
            cylinder('Column base ring', (x, y, .09), .024, .02, fumed_shadow, 24)
# A modesty panel closes the kneehole on the side the room sees.
box('Modesty panel', (dx+.27, dy, .42), (.025, .50, .60), fumed, .004)
box('Fielded modesty panel', (dx+.27+.0165, dy, .42), (.008, .40, .46), fumed, .005)
# A frieze spans the kneehole under the top and carries a shallow centre drawer.
box('Kneehole frieze', (dx, dy, .69), (.56, .50, .07), fumed, .004)
box('Centre drawer front', (dx-.28-.005, dy, .69), (.01, .44, .05), fumed, .003)
tube('Swan-neck brass pull', [(dx-.293, dy-.035, .678), (dx-.303, dy, .694), (dx-.293, dy+.035, .678)], .003, brass)
# The top, with an ogee lip beneath, and a leather surface in three panels.
box('Desk top', (dx, dy, dz+.0175), (.62, 1.30, .035), fumed, .006)
box('Ogee lip', (dx, dy, dz+.003), (.64, 1.32, .012), fumed_shadow, .004)
box('Leather writing surface', (dx, dy, dz+.036), (.46, 1.14, .002), hide, .0008)
for y in [dy-.22, dy+.22]:
    box('Oak divider strip', (dx, y, dz+.0365), (.46, .02, .003), fumed, .001)

# The writer's chair, a dark oak side chair with a curved slatted back and a
# nailed leather seat, turned a little from the desk as if just left.
def turn(objects, pivot, angle):
    """Yaw already-placed parts about a vertical axis through `pivot`."""
    pivot = Vector(pivot)
    for obj in objects:
        offset = obj.location - pivot
        obj.location = pivot + Vector((offset.x*math.cos(angle) - offset.y*math.sin(angle),
                                       offset.x*math.sin(angle) + offset.y*math.cos(angle), offset.z))
        obj.rotation_euler.z += angle

qx, qy = 1.10, -.55
chair_start = len(groups['furniture']['furniture'])
for y in [qy-.20, qy+.20]:
    cylinder('Chair front leg', (qx+.19, y, .22), .02, .44, fumed, 16, .014)
    tube('Chair rear stile', [(qx-.19, y, .01), (qx-.19, y, .46), (qx-.25, y, .94)], .017, fumed)
    tube('Chair side stretcher', [(qx+.19, y, .17), (qx-.19, y, .17)], .008, fumed)
tube('Chair cross stretcher', [(qx, qy-.20, .17), (qx, qy+.20, .17)], .008, fumed)
box('Chair seat frame', (qx, qy, .43), (.42, .44, .06), fumed, .006)
box('Nailed leather seat', (qx, qy, .485), (.41, .43, .05), chair_leather, .025)
for i in range(13):
    sphere('Brass nail head', (qx+.207, qy-.18+i*.03, .49), (.0035, .0035, .0035), brass)
    sphere('Brass nail head', (qx-.18+i*.03, qy-.217, .49), (.0035, .0035, .0035), brass)
    sphere('Brass nail head', (qx-.18+i*.03, qy+.217, .49), (.0035, .0035, .0035), brass)
tube('Chair lower back rail', [(qx-.20, qy-.19, .57), (qx-.20, qy+.19, .57)], .012, fumed)
tube('Chair crest rail', [(qx-.25, qy-.20, .94), (qx-.275, qy, .955), (qx-.25, qy+.20, .94)], .02, fumed)
box('Chair upholstered back', (qx-.24, qy, .76), (.075, .34, .34), chair_leather, .055)
for y in [qy-.09, qy+.09]:
    for z in [.70, .82]:
        sphere('Chair back button', (qx-.198, y, z), (.008, .008, .008), fumed_shadow)
turn(groups['furniture']['furniture'][chair_start:], (qx, qy, 0), -.2)

# A low round library table leaves more of the rug visible and suits the
# heavier period furniture.
table_x, table_y = -.28, -.59
marble = material('Brown marble', '#59483a', .30, 'plaster')
cylinder('Round coffee table top', (table_x, table_y, .415), .43, .065, marble, 64)
torus('Coffee table moulded rim', (table_x, table_y, .442), .405, .012, 64, 10, darkwood)
cylinder('Coffee table upper collar', (table_x, table_y, .365), .12, .055, darkwood, 32, .09)
cylinder('Coffee table turned pedestal', (table_x, table_y, .225), .065, .27, walnut, 32, .10)
cylinder('Coffee table lower collar', (table_x, table_y, .095), .16, .045, darkwood, 32, .12)
for angle in [0, math.pi/2, math.pi, math.pi*1.5]:
    foot = (table_x+math.cos(angle)*.30, table_y+math.sin(angle)*.30, .045)
    tube('Coffee table splayed foot', [(table_x, table_y, .10), foot], .024, walnut)

# A compact oxblood listening sofa runs along the open left edge and faces the
# writing desk. Its low back keeps the shelf camera's sightline clear.
velvet = material('Oxblood cotton velvet', '#42040b', .72, 'velvet')
velvet_dark = material('Oxblood velvet shadow', '#240207', .80, 'velvet')
sofa_x, sofa_y = -1.42, -.61
box('Sofa shadow plinth', (sofa_x, sofa_y, .15), (.54, 1.40, .065), darkwood, .025)
box('Sofa back', (sofa_x-.29, sofa_y, .68), (.20, 1.58, .86), velvet, .09)
box('Sofa seat deck', (sofa_x, sofa_y, .42), (.69, 1.50, .20), velvet_dark, .07)
box('Sofa front apron', (sofa_x+.345, sofa_y, .38), (.055, 1.38, .18), velvet, .025)
for y in [sofa_y-.49, sofa_y, sofa_y+.49]:
    box('Sofa loose seat cushion', (sofa_x+.05, y, .53), (.57, .46, .16), velvet, .055)
    box('Sofa padded back', (sofa_x-.12, y, .79), (.18, .44, .49), velvet, .07)
    for z in [.68, .84, .99]:
        sphere('Sofa tuft button', (sofa_x-.018, y, z), (.012, .012, .012), velvet_dark)
for y in [sofa_y-.80, sofa_y+.80]:
    box('Sofa rolled arm', (sofa_x+.01, y, .62), (.72, .20, .46), velvet, .09)
    scroll = cylinder('Sofa arm scroll', (sofa_x+.31, y, .68), .10, .20, velvet, 32)
    scroll.rotation_euler.x = math.pi/2
    for cap_y in [y-.102, y+.102]:
        sphere('Sofa arm upholstered cap', (sofa_x+.31, cap_y, .68), (.101, .018, .101), velvet)
for y in [sofa_y-.62+i*.125 for i in range(11)]:
    sphere('Sofa brass nail', (sofa_x+.376, y, .38), (.004, .004, .004), brass)
for x in [sofa_x-.23, sofa_x+.23]:
    for y in [sofa_y-.65, sofa_y+.65]:
        cylinder('Sofa walnut foot', (x, y, .10), .027, .20, walnut, 16, .02)

# Two restrained cushions tie the sofa to the green walls and old rug.
green_cushion = box('Sofa green cushion', (sofa_x-.08, sofa_y-.37, .73), (.13, .34, .34), hide, .055, -.08)
green_cushion.rotation_euler.y = -.22
tapestry_cushion = box('Sofa tapestry cushion', (sofa_x-.07, sofa_y+.34, .72), (.13, .32, .32), rug_cream, .05, .10)
tapestry_cushion.rotation_euler.y = -.18

group = 'objects'
# A dark Persian-style rug. Flat geometry becomes one baked textile atlas, so
# the browser pays no runtime cost for the medallion and repeated border.
box('Rug ground', (.10, -.35, .025), (2.6, 1.72, .012), rug_red, .01)
box('Rug outer border', (.10, -.35, .032), (2.43, 1.55, .004), rug_blue, .003)
box('Rug guard stripe', (.10, -.35, .035), (2.31, 1.43, .003), rug_cream, .002)
box('Rug inner border', (.10, -.35, .038), (2.19, 1.31, .002), rug_red, .002)
box('Rug dark field', (.10, -.35, .040), (1.91, 1.03, .002), rug_blue, .002)
for layer, (radius, mat) in enumerate([(.47, rug_red), (.36, rug_cream), (.27, rug_red), (.15, rug_cream)]):
    motif = cylinder('Rug central medallion', (.10, -.35, .043+layer*.002), radius, .002, mat, 12)
    motif.scale.y = .62
for angle in [0, math.pi/2, math.pi, math.pi*1.5]:
    x, y = .10+math.cos(angle)*.47, -.35+math.sin(angle)*.29
    petal = cylinder('Rug medallion palmette', (x, y, .052), .10, .002, rug_red, 8)
    petal.scale.y = .48
    petal.rotation_euler.z = angle
for x in [-.70, .90]:
    for y in [-.72, .02]:
        motif = cylinder('Rug corner motif', (x, y, .043), .13, .002, rug_red, 8)
        motif.scale.y = .70
        cylinder('Rug corner motif centre', (x, y, .045), .045, .002, rug_cream, 10)
for x in [-.65, -.28, .48, .85]:
    for y in [-.68, -.35, -.02]:
        cylinder('Rug field flower', (x, y, .044), .035, .002, rug_cream, 8)
        for angle in [0, math.pi/2, math.pi, math.pi*1.5]:
            px, py = x+math.cos(angle)*.06, y+math.sin(angle)*.06
            petal = cylinder('Rug field petal', (px, py, .043), .035, .002, rug_red, 8)
            petal.scale.y = .45
            petal.rotation_euler.z = angle
for y in [-1.01, .31]:
    for i in range(18):
        cylinder('Rug border rosette', (-.92+i*.12, y, .044), .026, .002,
                 rug_cream if i%2 else rug_red, 8)
for x in [-1.09, 1.29]:
    for i in range(9):
        cylinder('Rug side rosette', (x, -.83+i*.12, .044), .026, .002,
                 rug_cream if i%2 else rug_red, 8)
for x in [-1.23, 1.43]:
    for i in range(55):
        tube('Rug fringe', [(x, -1.14+i*.029, .026), (x+(.04 if x>0 else -.04), -1.14+i*.029, .02)], .002, rug_cream)

books = json.loads((work/'books.json').read_text())
slots = json.loads((work/'book-slots.json').read_text())
for bay in range(3):
    z = shelves[bay+1]
    for idx, (book, slot) in enumerate(zip(books, slots)):
        if slot['row'] != bay:
            continue
        width, h, z = slot['width'], slot['height'], slot['y']
        x = slot['x']-width/2
        group = 'books'
        part = 'Body_'+book['isbn']
        pivots[part] = (slot['x'], -slot['z'], z)
        cover = material('Leather '+str(idx), book['color'], .7, 'fabric')
        # Books move after baking. A little local fill keeps the covered sides
        # readable when exposed without lifting the cabinet's contact shadows.
        binding = cover.node_tree.nodes.get('Principled BSDF')
        binding.inputs['Emission Color'].default_value = (*linear(book['color']), 1)
        binding.inputs['Emission Strength'].default_value = .18
        box('Pleiade back cover', (x+.00075, 1.145, z+h/2), (.0015, .15, h), cover, .0005)
        box('Pleiade leather spine', (x+width/2, 1.072, z+h/2), (width, .004, h), cover, .001)
        box('Bible paper edges', (x+width/2, 1.145, z+h/2), (width-.003, .142, h-.004), cream, .0005)
        # Only an annotated volume gets a cover. Baking them all would split this
        # atlas fifty ways for faces nobody reaches, leaving the openable one
        # lettered at about eight pixels per centimetre.
        if book.get('noted'):
            group = 'covers'
            part = 'Cover_'+book['isbn']
            pivots[part] = (x+width-.00075, 1.070, z)
            box('Hinged leather cover', (x+width-.00075, 1.145, z+h/2), (.0015, .15, h), cover, .0005)
            box('Ivory front endpaper', (x+width-.00155, 1.145, z+h/2), (.0001, .145, h-.006), cream, .00004)
            artwork = plane_image('Front cover lettering', (x+width+.00006, 1.145, z+h/2), .15, h, work/f'cover-{idx}.png')
            artwork.rotation_euler.z = math.pi/2
        group = 'spines'
        part = 'Book_'+book['isbn']
        plane_image(part, (slot['x'], -slot['z'], z+h/2), width-.001, h-.003, work/f'book-{idx}.png')
        part = None
        group = 'objects'
    used_right = max((s['x']+s['width']/2 for s in slots if s['row'] == bay), default=cx-.49)
    if bay != 1 and used_right < cx+.295:
        for i in range(3):
            box('Books laid flat', (cx+.39, 1.19, z+.015+i*.031), (.16, .23, .028), cream, .002, .03*i)
# A reusable woven bookmark, modelled at the first volume's head. Its short
# folded tab lies over the headband instead of forming an upright handle.
group = 'bookmark'
part = 'Bookmark'
slot = slots[0]
pivots[part] = (slot['x'], -slot['z'], slot['y'] + slot['height'])
cloth = material('Madder silk bookmark', '#5b1320', .72)
bsdf = cloth.node_tree.nodes.get('Principled BSDF')
bsdf.inputs['Sheen Weight'].default_value = .35
nodes, links = cloth.node_tree.nodes, cloth.node_tree.links
coord = nodes.new('ShaderNodeTexCoord')
wave = nodes.new('ShaderNodeTexWave')
wave.wave_type = 'BANDS'
wave.bands_direction = 'X'
wave.inputs['Scale'].default_value = 95
links.new(coord.outputs['Generated'], wave.inputs['Vector'])
bump = nodes.new('ShaderNodeBump')
bump.inputs['Strength'].default_value = .22
bump.inputs['Distance'].default_value = .00008
links.new(wave.outputs['Color'], bump.inputs['Height'])
links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
# Cubic sections turn the cloth over once, with a shallow trough across its width.
knots = [( .020, -.003), (.018, .006), (.012, .008), (.006, .006),
         (-.001, .003), (-.004, -.005), (-.006, -.014)]
vertices, faces = [], []
for i in range(49):
    t = i / 48 * (len(knots)-1)
    k = min(int(t), len(knots)-2)
    u = t-k
    a, b, c, d = [knots[min(max(j, 0), len(knots)-1)] for j in (k-1,k,k+1,k+2)]
    y, z = [ .5*((2*b[n])+(-a[n]+c[n])*u+(2*a[n]-5*b[n]+4*c[n]-d[n])*u*u+(-a[n]+3*b[n]-3*c[n]+d[n])*u*u*u) for n in (0,1)]
    for j in range(9):
        across = j/8*2-1
        x = across*.0028*(1-.12*(i/48)**2) + .0015*(i/48)**2
        notch = .0012*(1-abs(across)) * max(0, (i-44)/4)
        twist = across*.0018*math.sin(math.pi*i/48)
        vertices.append((slot['x']+x, -slot['z']+y+twist, slot['y']+slot['height']+z+.0012*across*across+notch))
        if i < 48 and j < 8:
            v = i*9+j
            faces.append((v,v+1,v+10,v+9))
mesh = bpy.data.meshes.new('Woven bookmark folds')
mesh.from_pydata(vertices, [], faces)
obj = bpy.data.objects.new(part, mesh)
scene.collection.objects.link(obj)
finish(obj, part, cloth)
# The reusable template must not leave a shadow in the static room atlases.
obj.hide_render = True
for polygon in mesh.polygons:
    polygon.use_smooth = True
solid = obj.modifiers.new('Woven thickness', 'SOLIDIFY')
solid.thickness = .00018
bevel = obj.modifiers.new('Soft cloth edge', 'BEVEL')
bevel.width = .00009
bevel.segments = 2
part = None
group = 'objects'
# Heavy burgundy curtains frame the night window and carry the oxblood colour
# into the architecture. Overlapping rounded strips are broad enough to keep
# their folds after the object atlas is baked down for the browser.
curtain = material('Burgundy velvet drapery', '#35090e', .86, 'velvet')
tube('Curtain brass rod', [(-1.96, -.34, 2.29), (-1.96, 1.06, 2.29)], .012, brass)
for y in [-.34, 1.06]:
    sphere('Curtain rod finial', (-1.96, y, 2.29), (.025, .025, .025), brass)
for centre in [-.16, .88]:
    for fold in range(5):
        offset = (fold - 2) * .047
        height = 1.70 - abs(fold - 2) * .025
        box('Burgundy curtain fold', (-1.985 + (fold % 2) * .018, centre + offset, .53 + height / 2),
            (.07, .082, height), curtain, .028)
    box('Curtain brass tieback', (-1.94, centre, 1.27), (.08, .30, .025), brass, .008)

# Vinyl spines sit below the books.
record_colors = [cream, rug_red, rug_blue, black, linen, terra]
for i in range(39):
    box('Record sleeve in shelf', (-1.6+i*.012, 1.21, .352), (.009, .308, .32), record_colors[i%6], .001)

# A few quiet rows dress the upper built-in. These are scenery; the authored
# library remains the only set with readable jackets and interaction identity.
for shelf_index, z in enumerate([1.552, 1.952]):
    for side, start in enumerate([-.39, 1.07]):
        cursor = start
        for i in range(9):
            width = .025 + ((i*7 + shelf_index*3 + side) % 4) * .006
            height = .22 + ((i*5 + side) % 3) * .025
            book = box('Upper shelf book', (cursor+width/2, 1.17, z+height/2),
                       (width, .20, height), record_colors[(i+shelf_index+side)%len(record_colors)], .002)
            if i in (2, 5):
                book.rotation_euler.y = (-.10 if side else .10) * (1 if shelf_index else 1.4)
            cursor += width + .005

# Small collected objects break the regular book rows without competing with
# the interactive library or its readable jackets.
globe = material('Antique globe parchment', '#786744', .72, 'fabric')
gx, gy, gz = -.08, 1.18, 1.705
cylinder('Globe turned base', (gx, gy, 1.565), .055, .025, darkwood, 32, .042)
cylinder('Globe brass stem', (gx, gy, 1.625), .008, .11, brass, 20)
sphere('Antique library globe', (gx, gy, gz), (.082, .082, .082), globe)
torus('Globe meridian ring', (gx, gy, gz), .091, .003, 48, 8, brass, (math.pi/2, 0, 0))

bx, by = 1.42, 1.17
box('Classical bust plinth', (bx, by, 1.565), (.13, .11, .025), darkwood, .004)
cylinder('Classical bust torso', (bx, by, 1.625), .035, .10, cream, 24, .070)
cylinder('Classical bust neck', (bx, by, 1.675), .025, .07, cream, 20)
sphere('Classical bust head', (bx, by, 1.735), (.044, .040, .060), cream)
sphere('Classical bust hair', (bx+.005, by+.018, 1.755), (.047, .030, .045), darkwood)
sphere('Classical bust nose', (bx, by-.039, 1.741), (.010, .014, .012), cream)

for i, (width, yaw) in enumerate([(.22, -.04), (.20, .03), (.18, -.02)]):
    box('Upper shelf book stack', (.78, 1.19, 1.955+i*.027),
        (width, .17, .024), record_colors[(i+2)%len(record_colors)], .002, yaw)

cylinder('Library brass vase', (.84, 1.20, 2.035), .065, .15, brass, 32, .045)
for i, end in enumerate([(.72, 1.17, 2.25), (.78, 1.14, 2.29), (.87, 1.16, 2.27), (.94, 1.18, 2.23)]):
    tube('Library foliage stem', [(.84, 1.20, 2.10), end], .003, green)
    sphere('Library foliage leaf', end, (.045, .018, .075), green).rotation_euler.y = (-.35+i*.22)

# A trailing plant softens the rigid upper shelving without entering the three
# interactive book rows below it.
plant_x, plant_y = -.30, 1.13
cylinder('Trailing plant pot', (plant_x, plant_y, 2.005), .07, .14, terra, 32, .055)
for i, end in enumerate([(-.43, 1.08, 1.63), (-.30, 1.06, 1.57), (-.16, 1.08, 1.68)]):
    points = [(plant_x, plant_y, 2.07), ((plant_x+end[0])/2, 1.08, 1.88), end]
    tube('Trailing plant stem', points, .003, green)
    for step in [.30, .55, .78]:
        a = Vector(points[0]).lerp(Vector(end), step)
        leaf = sphere('Trailing plant leaf', a, (.045, .018, .070), green)
        leaf.rotation_euler.y = (-.45 + i*.25) * (1 if step != .55 else -1)

# Turntable.
tx, ty, tz = .26, 1.16, .712
box('Turntable walnut plinth', (tx, ty, tz), (.50, .38, .052), walnut, .014)
box('Turntable charcoal deck', (tx, ty, tz+.030), (.473, .356, .013), black, .005)
cylinder('Spindle', (tx-.052, ty, tz+.067), .0025, .018, metal, 12)

# The platter and the arm are the two things that move, so they leave the room's
# single mesh and turn about their own axis instead.
group, part = 'moving', 'Platter'
pivots[part] = (tx-.052, ty, tz)
cylinder('Platter edge', (tx-.052, ty, tz+.045), .151, .018, metal, 96)
cylinder('Vinyl LP', (tx-.052, ty, tz+.056), .147, .004, black, 96)
label = material('Record label brick red', '#a54e36', .8)
cylinder('Record paper label', (tx-.052, ty, tz+.059), .043, .001, label)
for radius in [.065, .078, .091, .108, .12, .134, .143]:
    torus('Pressed vinyl groove', (tx-.052, ty, tz+.058), radius, .00045, 96, 4, metal)

part = 'Tonearm'
pivots[part] = (tx+.183, ty+.125, tz)
cylinder('Tonearm bearing', (tx+.183, ty+.125, tz+.073), .019, .05, metal, 24)
tube('S shaped tonearm', [(tx+.183, ty+.125, tz+.10), (tx+.17, ty+.04, tz+.105), (tx+.13, ty-.085, tz+.105), (tx+.085, ty-.12, tz+.10)], .0045, metal)
box('Headshell', (tx+.083, ty-.13, tz+.096), (.018, .034, .014), black, .002, -.25)
box('Cartridge', (tx+.074, ty-.143, tz+.088), (.012, .019, .011), rug_red, .001, -.25)
tube('Stylus', [(tx+.073, ty-.148, tz+.087), (tx+.071, ty-.151, tz+.061)], .0007, metal)
group, part = 'objects', None
box('Start switch', (tx-.203, ty-.14, tz+.044), (.036, .026, .008), brass, .002)
cylinder('Speed selector', (tx-.202, ty+.12, tz+.046), .013, .012, brass, 20)
for x in [tx-.155, tx-.125]:
    box('Speed button', (x, ty+.137, tz+.044), (.018, .025, .008), metal, .002)
# Smoked open lid is modelled with a frame; no opaque card hiding the record.
for x in [tx-.246, tx+.246]:
    tube('Dust cover edge', [(x, ty+.18, tz+.036), (x, ty+.24, tz+.30)], .004, black)
tube('Dust cover upper edge', [(tx-.246, ty+.24, tz+.30), (tx+.246, ty+.24, tz+.30)], .004, black)

# Receiver with scale marks, switches and knurled knobs.
rx = .90
box('Receiver cabinet', (rx, 1.16, .754), (.60, .35, .13), walnut, .008)
box('Receiver brushed face', (rx, .978, .754), (.578, .008, .112), metal, .003)
box('Tuner dark glass', (rx-.072, .972, .774), (.35, .004, .05), black, .002)
dial = material('Amber tuner illumination', '#d2a85c', .6)
bsdf = dial.node_tree.nodes['Principled BSDF']
bsdf.inputs['Emission Color'].default_value = (1, .48, .1, 1)
bsdf.inputs['Emission Strength'].default_value = .4
for i in range(26):
    box('Radio frequency marking', (rx-.231+i*.012, .969, .779), (.0012, .001, .016 if i%5 == 0 else .008), dial, 0)
box('Tuner needle', (rx-.08, .967, .773), (.002, .002, .037), rug_red, 0)
for x, radius in [(rx+.226, .031), (rx-.19, .011), (rx-.12, .011), (rx-.05, .011), (rx+.025, .011)]:
    obj = cylinder('Receiver control', (x, .957, .739), radius, .026, metal, 32)
    obj.rotation_euler.x = math.pi/2
    box('Knob indicator', (x, .942, .743+radius*.4), (.0015, .001, radius*.4), black, 0)

# A matching pair of speakers, with recessed cones and rubber surrounds.
for x, y, z in [(-.31, 1.17, .71), (1.67, 1.15, .025)]:
    if z > .1:
        box('Speaker stand base', (x,y,.045), (.28,.30,.035), black)
        cylinder('Speaker stand column', (x,y,.375), .026,.65,black,24)
        box('Speaker stand top', (x,y,.696), (.25,.27,.025), black)
    box('Speaker walnut enclosure', (x, y, z+.25), (.285, .30, .50), walnut, .012)
    box('Speaker baffle', (x, y-.153, z+.25), (.254, .012, .464), black, .006)
    for h, radius in [(.17, .091), (.377, .042)]:
        obj = cylinder('Speaker cone', (x, y-.164, z+h), radius, .018, soil, 48, radius*.72)
        obj.rotation_euler.x = math.pi/2
        sphere('Speaker dust cap', (x, y-.182, z+h), (radius*.40, .018, radius*.40), black)
        torus('Speaker rubber surround', (x,y-.174,z+h), radius*.92, .006, 48, 8, black, (math.pi/2,0,0))

# Pleated linen shade with visible ribs, a brass stem and a broad warm practical.
lx, ly = -1.69, .18
cylinder('Lamp foot', (lx, ly, .046), .19, .045, black)
cylinder('Lamp brass stem', (lx, ly, .79), .012, 1.49, brass, 24)
verts, faces = [], []
for z, r in [(1.40, .265), (1.79, .145)]:
    for i in range(128):
        angle = 2*math.pi*i/128
        radius = r + (.006 if i%2 == 0 else -.006)
        verts.append((lx+math.cos(angle)*radius, ly+math.sin(angle)*radius, z))
for i in range(128):
    faces.append((i,(i+1)%128,(i+1)%128+128,i+128))
mesh = bpy.data.meshes.new('Pleated shade')
mesh.from_pydata(verts, [], faces)
shade = bpy.data.objects.new('Pleated linen shade', mesh)
scene.collection.objects.link(shade)
finish(shade, 'Pleated linen shade', linen)
for z, radius in [(1.4,.265),(1.79,.145)]:
    torus('Shade sewn binding', (lx,ly,z), radius, .004, 96, 6, cream)
bulb = material('Lamp glowing bulb', '#ffe3a6')
bulb.node_tree.nodes['Principled BSDF'].inputs['Emission Color'].default_value = (1,.66,.28,1)
bulb.node_tree.nodes['Principled BSDF'].inputs['Emission Strength'].default_value = 3
sphere('Lamp bulb', (lx,ly,1.57), (.043,.043,.058), bulb)

# Curved leaves, with stems and a raised central vein, rather than crossed cards.
px, py = 1.78, .39
cylinder('Terracotta pot', (px,py,.18), .155, .32, terra, 48, .115)
cylinder('Pot lip', (px,py,.328), .166, .035, terra)
cylinder('Pot soil', (px,py,.337), .145, .008, soil)
for i in range(10):
    angle = i*2.4
    h = .50+i*.056
    reach = .24 if i<7 else .17
    end = Vector((px+math.cos(angle)*reach, py+math.sin(angle)*reach, h))
    start = Vector((px,py,.34))
    tube('Plant stem', [start, (px+.02,py,h-.18), end], .0035, green)
    direction = Vector((math.cos(angle), math.sin(angle), .28)).normalized()
    side = Vector((-math.sin(angle), math.cos(angle), 0))
    vertices = []
    for j in range(9):
        t = j/8
        mid = end+direction*t*.24+Vector((0,0,.045*math.sin(t*math.pi)-.07*t*t))
        width = math.sin(t*math.pi)**.7*.066
        vertices.extend([mid-side*width, mid+Vector((0,0,.009*math.sin(t*math.pi))), mid+side*width])
    faces = []
    for j in range(8):
        for k in range(2):
            a = j*3+k
            faces.append((a,a+3,a+4,a+1))
    mesh = bpy.data.meshes.new('Rubber leaf')
    mesh.from_pydata(vertices, [], faces)
    obj = bpy.data.objects.new('Curved rubber leaf', mesh)
    scene.collection.objects.link(obj)
    for p in mesh.polygons:
        p.use_smooth = True
    finish(obj, obj.name, green)

# A graphic print, made for this room, and a sleeve left on the table.
box('Print walnut frame', (.53,1.453,1.76), (.72,.035,.84), walnut)
plane_image('Geometric music print', (.53,1.432,1.76), .655,.775,work/'print.png')
cover = plane_image('Record sleeve on table', (-.43,-.59,.455), .315,.315,work/'sleeve.png')
cover.rotation_euler = (0,0,-.19)
# Open mug, coffee and a curved handle.
cylinder('Mug', (.02,-.55,.505), .038,.108,ceramic,48,.031)
cylinder('Coffee', (.02,-.55,.560), .032,.001,soil)
tube('Mug handle', [(.052,-.55,.538),(.088,-.55,.533),(.083,-.55,.492),(.05,-.55,.478)], .007,ceramic)
# What is on the desk: a brass banker's lamp with an emerald glass shade, a
# manuscript with a loose sheet, a fountain pen and an inkwell.
glass = material('Emerald lamp glass', '#1d6a3c', .18)
bsdf = glass.node_tree.nodes.get('Principled BSDF')
bsdf.inputs['Emission Color'].default_value = (*linear('#2f9a55'), 1)
bsdf.inputs['Emission Strength'].default_value = .45
top = desk['top']
cylinder('Banker lamp base', (1.86, -.25, top+.007), .058, .014, brass, 48)
sphere('Banker lamp base dome', (1.86, -.25, top+.014), (.04, .04, .02), brass)
cylinder('Banker lamp stem', (1.86, -.25, top+.014+.125), .008, .25, brass, 24)
sphere('Banker lamp finial', (1.86, -.25, top+.27), (.012, .012, .012), brass)
tube('Banker lamp arm', [(1.86, -.25, top+.27), (1.83, -.25, top+.28)], .006, brass)
# The shade is a half cylinder along the desk, open at the bottom, given glass thickness.
sx, sy, sz, radius, half = 1.83, -.25, top+.28, .065, .135
vertices, faces = [], []
for j in range(13):
    a = math.pi*j/12
    vertices.extend([(sx+radius*math.cos(a), sy-half, sz+radius*math.sin(a)),
                     (sx+radius*math.cos(a), sy+half, sz+radius*math.sin(a))])
    if j:
        faces.append((2*j-2, 2*j-1, 2*j+1, 2*j))
mesh = bpy.data.meshes.new('Banker lamp shade')
mesh.from_pydata(vertices, [], faces)
shade = bpy.data.objects.new('Banker lamp shade', mesh)
scene.collection.objects.link(shade)
for p in mesh.polygons:
    p.use_smooth = True
shade.modifiers.new('Glass thickness', 'SOLIDIFY').thickness = .004
finish(shade, shade.name, glass)
for y in [sy-half, sy+half]:
    tube('Shade brass rim', [(sx+radius, y, sz), (sx, y, sz+radius), (sx-radius, y, sz)], .004, brass)
# A fountain pen laid between the piles: a tapered black resin barrel, the cap
# posted on its end, a brass band and clip, and the nib out, ready.
pen = Vector((1.56, -.715, top+.0065))
along = Vector((math.cos(-.12), math.sin(-.12), 0))
def pen_point(t): return pen + along*t
tube('Fountain pen barrel', [pen_point(.0), pen_point(.045), pen_point(.075)], .0058, black)
cap = cylinder('Fountain pen cap', (0, 0, 0), .0068, .058, black, 32, .0055)
cap.location = pen_point(.104)
cap.rotation_euler = along.to_track_quat('Z', 'Y').to_euler()
band = cylinder('Pen cap band', (0, 0, 0), .0072, .004, brass, 32)
band.location = pen_point(.077)
band.rotation_euler = cap.rotation_euler
sphere('Pen cap finial', pen_point(.134), (.004, .004, .004), brass)
tube('Pen clip', [pen_point(.085) + Vector((0, 0, .0075)), pen_point(.128) + Vector((0, 0, .0085)), pen_point(.130) + Vector((0, 0, .006))], .0013, brass)
nib = cylinder('Pen nib', (0, 0, 0), .0038, .016, brass, 16, .0008)
nib.location = pen_point(-.008)
nib.rotation_euler = (-along).to_track_quat('Z', 'Y').to_euler()
cylinder('Inkwell', (1.88, -.12, top+.017), .022, .034, black, 32)
cylinder('Inkwell lid', (1.88, -.12, top+.040), .012, .012, brass, 24)

# The writing, one manuscript per published piece, fanned in two piles so the
# head of every sheet shows. Each is its own node: the runtime lifts the one a
# reader reaches for. Their layout arrives in the runtime's axes.
for index, sheet in enumerate(json.loads((work/'sheets.json').read_text())):
    group = 'sheets'
    part = 'Sheet_' + sheet['slug']
    x, y, z, yaw = sheet['x'], -sheet['z'], sheet['y'], sheet['yaw']
    pivots[part] = (x, y, z)
    box('Manuscript leaves', (x, y, z+.00125), (sheet['height'], sheet['width'], .0025), cream, .0003, yaw)
    # The page is drawn head-up; lying flat it is turned so the head faces +x.
    plane_image(part, (x, y, z+.0028), sheet['width'], sheet['height'], work/f'sheet-{index}.png', yaw - math.pi/2)
    part = None
    group = 'objects'

def light(name, at, target, energy, color, size):
    data = bpy.data.lights.new(name, 'AREA')
    data.energy = energy
    data.color = color
    data.shape = 'DISK'
    data.size = size
    obj = bpy.data.objects.new(name,data)
    scene.collection.objects.link(obj)
    obj.location = at
    obj.rotation_euler = (Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()

light('Soft window moonlight', (-3,-2.3,4.5), (-.4,.4,.7), 220, (.56,.67,.92), 4)
light('Warm room key', (3,-1.5,3.8), (.1,.4,.8), 120, (1,.78,.58), 3)
light('Library wash', (0,1.0,2.35), (0,1.42,1.10), 90, (1,.65,.35), 1.4)
light('Lamp down', (lx,ly,1.42), (lx,ly,0), 75, (1,.53,.22), .32)
light('Lamp up', (lx,ly,1.77), (lx,ly,2.6), 18, (1,.62,.32), .22)
light('Banker lamp', (1.83,-.25,dz+.30), (1.83,-.25,dz), 30, (1,.73,.42), .07)

# A long-lens overview, with the complete cutaway visible against warm white.
camera_data = bpy.data.cameras.new('Room camera')
camera = bpy.data.objects.new('Room camera',camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
camera.location = (6.4,-9.9,6.5)
target = Vector((0,.20,1.0))
camera.rotation_euler = (target-camera.location).to_track_quat('-Z','Y').to_euler()
camera_data.type = 'PERSP'
camera_data.lens = 48
scene.render.film_transparent = True
out.mkdir(parents=True, exist_ok=True)

# Convert and apply in object space before UV unwrapping; retain all viewing sides.
for parts in groups.values():
    for name, objects in parts.items():
        bpy.ops.object.select_all(action='DESELECT')
        for obj in objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = objects[0]
        bpy.ops.object.convert(target='MESH')
        if len(objects) > 1:
            bpy.ops.object.join()
        joined = bpy.context.object
        joined.name = name
        objects[:] = [joined]
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        # A movable part turns about its origin, so put the origin on that axis
        # while the geometry stays where it was modelled and baked.
        if name in pivots:
            scene.cursor.location = pivots[name]
            bpy.ops.object.origin_set(type='ORIGIN_CURSOR')

source_name = 'living-room.blend' if not minimal_art else f"living-room-{'-'.join(sorted(targets))}.blend"
bpy.ops.wm.save_as_mainfile(filepath=str(work/source_name))
print(f'TIMING scene {time.perf_counter()-started:.2f}s', flush=True)
if preview_only:
    scene.render.filepath = str(work/'preview.png')
    render_started = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    print(f'TIMING preview {time.perf_counter()-render_started:.2f}s', flush=True)
    sys.exit(0)

@contextmanager
def isolated():
    """Hide every mesh from the render, and restore them even if a bake fails."""
    hidden = [(obj, obj.hide_render) for obj in scene.objects if obj.type == 'MESH']
    for obj, _ in hidden:
        obj.hide_render = True
    try:
        yield
    finally:
        for obj, was_hidden in hidden:
            obj.hide_render = was_hidden

def pack_grid(meshes, columns, source_layer):
    """Lay each mesh's bake UVs, taken from `source_layer`, into its own padded grid cell."""
    rows = math.ceil(len(meshes)/columns)
    padding = 10/(size*2)
    for index, obj in enumerate(meshes):
        for source, baked_uv in zip(obj.data.uv_layers[source_layer].data, obj.data.uv_layers[-1].data):
            baked_uv.uv = (
                (index % columns)/columns + padding + source.uv.x*(1/columns-2*padding),
                (index // columns)/rows + padding + source.uv.y*(1/rows-2*padding),
            )

for name, parts in groups.items():
    if name not in targets:
        continue
    bake_started = time.perf_counter()
    meshes = [objects[0] for objects in parts.values()]
    bpy.ops.object.select_all(action='DESELECT')
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    # The source UVs carry print artwork; smart-project writes a second UV layer.
    # Several parts unwrap together, so one group is still one atlas.
    for obj in meshes:
        source_uv = obj.data.uv_layers.active
        for mat in obj.data.materials:
            for node in list(mat.node_tree.nodes):
                if node.type == 'TEX_IMAGE' and not node.inputs['Vector'].is_linked:
                    uv = mat.node_tree.nodes.new('ShaderNodeUVMap')
                    uv.uv_map = source_uv.name
                    mat.node_tree.links.new(uv.outputs[0],node.inputs['Vector'])
        obj.data.uv_layers.new(name='BakeUV')
        obj.data.uv_layers.active_index = len(obj.data.uv_layers)-1
        obj.data.uv_layers.active.active_render = True
    # Books bake one at a time into their own atlas cell, so each is unwrapped
    # alone to fill it.
    for batch in [[obj] for obj in meshes] if name == 'books' else [meshes]:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in batch:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = batch[0]
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=.006)
        bpy.ops.object.mode_set(mode='OBJECT')
    for obj in meshes:
        obj.select_set(True)
    if name == 'sheets':
        # Smart Project packs each object separately, so their UVs otherwise
        # overlap and later sheet bakes overwrite the earlier ones.
        pack_grid(meshes, math.ceil(math.sqrt(len(meshes))), -1)
    if name == 'spines':
        # A regular grid spends its pixels on lettering, not on the empty bands
        # auto-packing leaves around tall, narrow islands. Cells are spine-shaped,
        # about one to twelve: a squarer cell starves the only type in the room a
        # visitor is meant to read.
        pack_grid(meshes, math.ceil(math.sqrt(len(meshes)*12)), 0)
    # A cover is read at arm's length, so it is sized from how many are actually
    # baked rather than from a fixed sheet: one note wants a whole one to itself.
    # ponytail: doubles per four covers, capped; revisit if the shelf ever carries
    # enough notes to need per-cover atlases instead.
    covers_atlas = min(size * 2, 2048 * 2 ** max(0, math.ceil(math.log2(max(1, len(parts)) / 4) / 2)))
    resolution = (
        1024 if name == 'bookmark' else
        covers_atlas if name == 'covers' else
        size * 2 if name in ('objects', 'spines', 'sheets') else size // 2 if name == 'moving' else size
    )
    image = bpy.data.images.new(name+' baked',width=resolution,height=resolution,float_buffer=True)
    for obj in meshes:
        for mat in obj.data.materials:
            node = mat.node_tree.nodes.new('ShaderNodeTexImage')
            node.image = image
            mat.node_tree.nodes.active = node
    scene.render.bake.use_pass_direct = True
    scene.render.bake.use_pass_indirect = True
    scene.render.bake.use_pass_color = True
    scene.render.bake.use_pass_glossy = False
    scene.render.bake.use_pass_transmission = False
    scene.render.bake.margin = 16
    print('BAKING', name, resolution, samples, flush=True)
    # Moving surfaces cannot inherit shadows from their neighbours. Bake each
    # volume in the same room light, with only its own covers shading its pages.
    # Moving surfaces bake with the rest of the room hidden, so none keeps the
    # shadow of where it stood.
    with isolated() if name in isolated_groups else nullcontext():
        if name in ('covers', 'sheets'):
            bpy.ops.object.duplicate(linked=False)
            if len(meshes) > 1:
                bpy.ops.object.join()
            combined = bpy.context.object
            combined.hide_render = False
            # These flat surfaces need no self-shadowing. Exclude them from
            # secondary rays; a joined copy bakes once while the originals retain
            # their runtime transforms.
            combined.visible_shadow = False
            combined.visible_diffuse = False
            combined.visible_glossy = False
            bpy.ops.object.bake(type='COMBINED')
            bpy.data.objects.remove(combined, do_unlink=True)
        elif name == 'spines':
            # Each jacket stays its own `Book_<isbn>` node for the browser. A joined
            # copy bakes them all in one pass, with the originals out of its light.
            bpy.ops.object.duplicate(linked=False)
            bpy.ops.object.join()
            combined = bpy.context.object
            for obj in meshes:
                obj.hide_render = True
            bpy.ops.object.bake(type='COMBINED')
            bpy.data.objects.remove(combined, do_unlink=True)
            for obj in meshes:
                obj.hide_render = False
        elif name == 'books':
            # Each volume bakes alone into a small image, then lands in its own
            # cell of the atlas. Cycles denoises the whole target image on every
            # bake, so baking straight into the atlas would denoise it once per book.
            columns = math.ceil(math.sqrt(len(meshes)))
            rows = math.ceil(len(meshes)/columns)
            cell_w, cell_h = resolution // columns, resolution // rows
            # Inset each unwrap a few pixels inside its cell, so filtering in the
            # browser never samples the neighbouring book.
            inset_x, inset_y = 4/cell_w, 4/cell_h
            cell = bpy.data.images.new('book cell', width=cell_w, height=cell_h, float_buffer=True)
            pixels = np.empty(cell_w*cell_h*4, dtype=np.float32)
            atlas = np.zeros((resolution, resolution, 4), dtype=np.float32)
            scene.render.bake.use_clear = True
            for index, obj in enumerate(meshes):
                col, row = index % columns, index // columns
                baked_uv = obj.data.uv_layers[-1].data
                for loop in baked_uv:
                    loop.uv = (inset_x + loop.uv.x*(1-2*inset_x), inset_y + loop.uv.y*(1-2*inset_y))
                for mat in obj.data.materials:
                    mat.node_tree.nodes.active.image = cell
                bpy.ops.object.select_all(action='DESELECT')
                obj.hide_render = False
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.bake(type='COMBINED')
                obj.hide_render = True
                cell.pixels.foreach_get(pixels)
                atlas[row*cell_h:(row+1)*cell_h, col*cell_w:(col+1)*cell_w] = pixels.reshape(cell_h, cell_w, 4)
                for loop in baked_uv:
                    loop.uv = ((col + loop.uv.x)*cell_w/resolution, (row + loop.uv.y)*cell_h/resolution)
                for mat in obj.data.materials:
                    mat.node_tree.nodes.active.image = image
            image.pixels.foreach_set(atlas.ravel())
            image.update()
            bpy.data.images.remove(cell)
        else:
            for obj in meshes:
                obj.hide_render = False
            bpy.ops.object.bake(type='COMBINED')
    # Color-manage once here. Runtime emission materials are display-referred.
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.quality = 95
    image.save_render(str(work/(name+'.jpg')),scene=scene)
    baked = bpy.data.images.load(str(work/(name+'.jpg')))
    mat = bpy.data.materials.new(name+' baked unlit')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    tex = nodes.new('ShaderNodeTexImage')
    tex.image = baked
    output = nodes.new('ShaderNodeOutputMaterial')
    # A color connected directly to Surface is Blender's glTF unlit convention.
    mat.node_tree.links.new(tex.outputs['Color'],output.inputs[0])
    # Do not change the other groups' source materials until every bake is done.
    for obj in meshes:
        obj['baked_material'] = mat.name
    print(f'TIMING bake:{name} {time.perf_counter()-bake_started:.2f}s', flush=True)

checkpoint_started = time.perf_counter()
checkpoint_name = source_name.removesuffix('.blend') + '-baked.blend'
bpy.ops.wm.save_as_mainfile(filepath=str(work/checkpoint_name))
print(f'TIMING checkpoint {time.perf_counter()-checkpoint_started:.2f}s', flush=True)

for name, parts in groups.items():
    if name not in targets:
        continue
    export_started = time.perf_counter()
    meshes = [objects[0] for objects in parts.values()]
    bpy.ops.object.select_all(action='DESELECT')
    for obj in meshes:
        obj.data.materials.clear()
        obj.data.materials.append(bpy.data.materials[obj['baked_material']])
        for p in obj.data.polygons:
            p.material_index = 0
        while len(obj.data.uv_layers)>1:
            obj.data.uv_layers.remove(obj.data.uv_layers[0])
        obj.data.uv_layers.active_index = 0
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.export_scene.gltf(filepath=str(out/(name+'.glb')),export_format='GLB',use_selection=True,
                              export_texcoords=True,export_normals=False,export_materials='EXPORT',
                              export_image_format='JPEG',export_jpeg_quality=95)
    print(f'TIMING export:{name} {time.perf_counter()-export_started:.2f}s', flush=True)
print('Room assets exported',flush=True)
