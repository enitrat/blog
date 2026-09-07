"""Original living-room assets. Run through `bun run room:bake` with Blender 4.5.

Dimensions are metres, Z is up. Cycles resolves the materials and illumination;
the exported glTF materials are unlit. No reference-site assets are included.
"""
import bpy
import json
import math
import os
import random
import sys
from pathlib import Path
from mathutils import Vector

args = sys.argv[sys.argv.index('--') + 1:]
work = Path(args[0])
out = Path(args[1])
samples = int(args[2])
size = int(args[3])
preview_only = '--preview' in args
random.seed(28)
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
    scene.cycles.device = 'GPU'
except (TypeError, RuntimeError):
    scene.cycles.device = 'CPU'
scene.world.use_nodes = True
scene.world.node_tree.nodes['Background'].inputs[0].default_value = (0.72, 0.79, 0.9, 1)
scene.world.node_tree.nodes['Background'].inputs[1].default_value = 0.3
scene.view_settings.view_transform = 'AgX'
scene.view_settings.look = 'AgX - Medium High Contrast'
scene.view_settings.exposure = 0.35
scene.render.image_settings.file_format = 'PNG'
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.resolution_percentage = 100

groups = {'shell': {}, 'furniture': {}, 'objects': {}, 'moving': {}, 'spines': {}}
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
        mapping.inputs[1].default_value = (3, 55, 4) if texture == 'wood' else (180, 180, 180)
        links.new(coord.outputs['Generated'], mapping.inputs[0])
        noise = nodes.new('ShaderNodeTexNoise')
        noise.inputs['Scale'].default_value = 2.5
        noise.inputs['Detail'].default_value = 3
        noise.inputs['Roughness'].default_value = .7
        links.new(mapping.outputs[0], noise.inputs['Vector'])
        ramp = nodes.new('ShaderNodeValToRGB')
        ramp.color_ramp.elements[0].position = .2
        ramp.color_ramp.elements[0].color = (*(v * .65 for v in base), 1)
        ramp.color_ramp.elements[1].position = .8
        ramp.color_ramp.elements[1].color = (*(min(1, v * 1.2) for v in base), 1)
        links.new(noise.outputs['Fac'], ramp.inputs[0])
        links.new(ramp.outputs[0], bsdf.inputs['Base Color'])
        bump = nodes.new('ShaderNodeBump')
        bump.inputs['Strength'].default_value = .18
        bump.inputs['Distance'].default_value = .0015 if texture == 'wood' else .0007
        links.new(noise.outputs['Fac'], bump.inputs['Height'])
        links.new(bump.outputs[0], bsdf.inputs['Normal'])
    return mat

plaster = material('Warm lime plaster', '#d3c9b5', .95, 'fabric')
walnut = material('Oiled walnut', '#795037', .38, 'wood')
oak = material('Oak end grain', '#a17b50', .48, 'wood')
darkwood = material('Walnut shadow', '#493223', .65, 'wood')
black = material('Charcoal enamel', '#242725', .35)
metal = material('Brushed aluminium', '#aaa9a2', .3, metal=.2)
brass = material('Aged brass', '#ad8750', .32, metal=.15)
cream = material('Ivory paper', '#ddd3b5', .85)
linen = material('Oatmeal linen', '#c5ad7e', .9, 'fabric')
leather = material('Cognac leather', '#925634', .5, 'fabric')
ceramic = material('Porcelain', '#bdbca9', .24)
green = material('Rubber plant leaves', '#405b31', .45)
soil = material('Potting soil', '#302821', 1)
terra = material('Terracotta', '#a66a48', .9, 'fabric')
rug_red = material('Kilim madder', '#9b483a', .98, 'fabric')
rug_blue = material('Kilim indigo', '#3a5359', .98, 'fabric')
rug_cream = material('Kilim flax', '#c6ac81', .98, 'fabric')

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

def box(name, at, dims, mat, bevel=.008, rotation=0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=at)
    obj = bpy.context.object
    obj.dimensions = dims
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.rotation_euler.z = rotation
    return finish(obj, name, mat, bevel)

def cylinder(name, at, radius, depth, mat, vertices=48, radius2=None):
    bpy.ops.mesh.primitive_cone_add(vertices=vertices, radius1=radius if radius2 is None else radius2,
                                  radius2=radius, depth=depth, location=at)
    obj = finish(bpy.context.object, name, mat, .002)
    for p in obj.data.polygons:
        p.use_smooth = len(p.vertices) == 4
    return obj

def sphere(name, at, scale, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, location=at)
    obj = bpy.context.object
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    for p in obj.data.polygons:
        p.use_smooth = True
    return finish(obj, name, mat)

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

def plane_image(name, at, width, height, path):
    mat = material(name, '#ffffff')
    tex = mat.node_tree.nodes.new('ShaderNodeTexImage')
    tex.image = bpy.data.images.load(str(path))
    mat.node_tree.links.new(tex.outputs['Color'], mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'])
    bpy.ops.mesh.primitive_plane_add(size=1, location=at, rotation=(math.pi/2, 0, 0))
    obj = bpy.context.object
    obj.scale = (width, height, 1)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return finish(obj, name, mat)

# A cutaway corner, with open space around the furniture and a real floor edge.
box('Floating oak foundation', (0, 0, -.10), (4.3, 3.1, .18), darkwood, .035)
floor_mats = [material('Oak board '+str(i), c, .58, 'wood') for i, c in enumerate(
    ['#ad8a5e', '#a98559', '#b39165', '#a27b50', '#af8b5f'])]
for row in range(16):
    y = -1.45 + row * .193
    for col in range(4):
        x = -1.6125 + col * 1.075
        box('Individual floor board', (x, y, .003), (1.071, .189, .025), floor_mats[(row*3+col)%5], .002)
box('Back wall', (0, 1.52, 1.28), (4.3, .09, 2.56), plaster, .012)
# Left wall is low at the front, preserving the lamp silhouette from the room camera.
box('Left wall', (-2.105, .45, 1.28), (.09, 2.14, 2.56), plaster, .012)
box('Back skirting', (0, 1.452, .07), (4.18, .035, .11), oak, .005)
box('Left skirting', (-2.041, .45, .07), (.035, 2.03, .11), oak, .005)

group = 'furniture'
# Bookcase: thinner shelves and a recessed back give each bay depth.
cx, cy, cw, ch, cd = -1.12, 1.24, 1.08, 1.68, .40
box('Bookcase inset back', (cx, cy+.18, .9), (cw-.035, .018, 1.57), darkwood, .003)
for x in [cx-cw/2+.013, cx+cw/2-.013]:
    box('Bookcase side', (x, cy, .89), (.027, cd, 1.59), walnut)
box('Bookcase crown', (cx, cy, 1.695), (1.10, .42, .036), walnut)
box('Bookcase plinth', (cx, cy, .09), (.99, .34, .14), darkwood)
shelves = [.18, .55, .91, 1.27]
for z in shelves:
    box('Shelf with rounded lip', (cx, cy, z-.013), (1.03, .39, .027), walnut, .005)

# A 1970s walnut credenza, with sliding fronts, finger pulls and tapered legs.
hx, hy = .68, 1.15
for x in [-.03, 1.39]:
    for y in [.99, 1.33]:
        cylinder('Credenza tapered foot', (x, y, .12), .022, .22, walnut, 16, .015)
box('Credenza carcass', (hx, hy, .42), (1.66, .47, .45), darkwood)
box('Credenza top', (hx, hy, .665), (1.72, .51, .04), walnut, .012)
for x in [.26, 1.09]:
    box('Sliding walnut door', (x, .904, .435), (.812, .025, .387), walnut, .005)
    box('Recessed finger pull', (x+.30, .887, .455), (.012, .008, .085), black, .005)
box('Credenza lower rail', (hx, .91, .221), (1.64, .035, .028), walnut)

# A small leather reading chair, curved cushions and an exposed walnut frame.
ax, ay = 1.29, -.67
for x in [ax-.30, ax+.30]:
    for y in [ay-.28, ay+.29]:
        cylinder('Chair leg', (x, y, .22), .022, .42, walnut, 16, .015)
    tube('Chair continuous arm', [(x, ay-.33, .57), (x, ay-.15, .62), (x, ay+.34, .69)], .033, walnut)
box('Chair seat frame', (ax, ay, .37), (.65, .66, .09), walnut, .035)
seat = box('Soft leather seat', (ax, ay-.01, .46), (.60, .60, .15), leather, .065)
back = box('Soft leather back', (ax, ay+.285, .78), (.59, .17, .58), leather, .07)
back.rotation_euler.x = math.radians(12)
for x in [ax-.15, ax+.15]:
    sphere('Upholstery button', (x, ay+.187, .81), (.016, .009, .016), darkwood)
for offset in [-.273, .273]:
    tube('Seat welt', [(ax+offset, ay-.26, .493), (ax+offset, ay, .505), (ax+offset, ay+.24, .493)], .0025, darkwood)

# Coffee table, with a rounded rectangular top and splayed legs.
for x in [-.57, .01]:
    for y in [-.80, -.40]:
        tube('Coffee table leg', [(x, y, .035), (x*.88-.03, y*.94, .38)], .018, walnut)
box('Coffee table top', (-.28, -.59, .405), (.90, .62, .055), walnut, .10)

group = 'objects'
# Woven kilim. The motifs are geometry in the source and become texture in the bake.
box('Kilim ground', (.10, -.35, .025), (2.6, 1.72, .012), rug_red, .01)
box('Kilim border', (.10, -.35, .032), (2.41, 1.53, .004), rug_cream, .003)
box('Kilim inner field', (.10, -.35, .035), (2.29, 1.41, .003), rug_blue, .002)
for x in [-.69, -.17, .35, .87]:
    obj = box('Kilim diamond', (x, -.35, .038), (.32, .32, .001), rug_red, 0, math.pi/4)
    box('Kilim diamond centre', (x, -.35, .040), (.145, .145, .001), rug_cream, 0, math.pi/4)
for y in [-.94, .24]:
    for i in range(22):
        box('Kilim border stitch', (-.99+i*.104, y, .039), (.036, .06, .001), rug_cream, 0, math.pi/4)
for x in [-1.23, 1.43]:
    for i in range(55):
        tube('Kilim fringe', [(x, -1.14+i*.029, .026), (x+(.04 if x>0 else -.04), -1.14+i*.029, .02)], .002, rug_cream)

books = json.loads((work/'books.json').read_text())
slots = json.loads((work/'book-slots.json').read_text())
for bay in range(3):
    z = shelves[bay+1]
    for idx, (book, slot) in enumerate(zip(books, slots)):
        if slot['row'] != bay:
            continue
        width, h, z = slot['width'], slot['height'], slot['y']
        x = slot['x']-width/2
        cover = material('Leather '+str(idx), book['color'], .7)
        box('Pleiade leather binding', (x+width/2, 1.145, z+h/2), (width, .15, h), cover, .003)
        box('Bible paper edges', (x+width/2, 1.15, z+h/2+.001), (width-.004, .135, h-.006), cream, .001)
        group = 'spines'
        part = 'Book_'+book['isbn']
        jacket = plane_image(part, (slot['x'], -slot['z'], z+h/2), width-.001, h-.003, work/f'book-{idx}.png')
        jacket['isbn'] = book['isbn']
        part = None
        group = 'objects'
    used_right = max((s['x']+s['width']/2 for s in slots if s['row'] == bay), default=cx-.49)
    if bay != 1 and used_right < cx+.295:
        for i in range(3):
            box('Books laid flat', (cx+.39, 1.19, z+.015+i*.031), (.16, .23, .028), cream, .002, .03*i)
# Vinyl spines sit below the books.
record_colors = [cream, rug_red, rug_blue, black, linen, terra]
for i in range(39):
    box('Record sleeve in shelf', (-1.6+i*.012, 1.21, .352), (.009, .308, .32), record_colors[i%6], .001)

# Turntable. The platter and tonearm are exported as their own movable nodes.
tx, ty, tz = .26, 1.16, .712
box('Turntable walnut plinth', (tx, ty, tz), (.50, .38, .052), walnut, .014)
box('Turntable aluminium deck', (tx, ty, tz+.030), (.473, .356, .013), metal, .005)
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
    bpy.ops.mesh.primitive_torus_add(major_segments=96, minor_segments=4, location=(tx-.052, ty, tz+.058), major_radius=radius, minor_radius=.00045)
    finish(bpy.context.object, 'Pressed vinyl groove', metal)

part = 'Tonearm'
pivots[part] = (tx+.183, ty+.125, tz)
cylinder('Tonearm bearing', (tx+.183, ty+.125, tz+.073), .019, .05, metal, 24)
tube('S shaped tonearm', [(tx+.183, ty+.125, tz+.10), (tx+.17, ty+.04, tz+.105), (tx+.13, ty-.085, tz+.105), (tx+.085, ty-.12, tz+.10)], .0045, metal)
box('Headshell', (tx+.083, ty-.13, tz+.096), (.018, .034, .014), black, .002, -.25)
group, part = 'objects', None
box('Start switch', (tx-.203, ty-.14, tz+.044), (.036, .026, .008), black, .002)
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
        bpy.ops.mesh.primitive_torus_add(major_segments=48, minor_segments=8, location=(x,y-.174,z+h), rotation=(math.pi/2,0,0), major_radius=radius*.92, minor_radius=.006)
        finish(bpy.context.object, 'Speaker rubber surround', black)

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
    bpy.ops.mesh.primitive_torus_add(major_segments=96, minor_segments=6, location=(lx,ly,z), major_radius=radius, minor_radius=.004)
    finish(bpy.context.object, 'Shade sewn binding', cream)
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
cover = plane_image('Record sleeve on table', (-.43,-.59,.438), .315,.315,work/'sleeve.png')
cover.rotation_euler = (0,0,-.19)
# Open mug, coffee and a curved handle.
cylinder('Mug', (.02,-.55,.493), .038,.108,ceramic,48,.031)
cylinder('Coffee', (.02,-.55,.548), .032,.001,soil)
tube('Mug handle', [(.052,-.55,.526),(.088,-.55,.521),(.083,-.55,.48),(.05,-.55,.466)], .007,ceramic)

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

light('Large soft window', (-3,-2.3,4.5), (0,.5,.5), 450, (1,.89,.72), 4)
light('Cool room fill', (3,-1.5,3.8), (0,.5,.8), 170, (.72,.82,1), 3)
light('Lamp down', (lx,ly,1.42), (lx,ly,0), 22, (1,.64,.30), .35)
light('Lamp up', (lx,ly,1.77), (lx,ly,2.6), 15, (1,.72,.42), .22)

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

bpy.ops.wm.save_as_mainfile(filepath=str(work/'living-room.blend'))
if preview_only:
    scene.render.filepath = str(work/'preview.png')
    bpy.ops.render.render(write_still=True)
    sys.exit(0)

for name, parts in groups.items():
    if '--spines-only' in args and name != 'spines':
        continue
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
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=.006)
    bpy.ops.object.mode_set(mode='OBJECT')
    if name == 'spines':
        # A regular print atlas spends its pixels on lettering rather than the
        # empty bands produced by packing tall, narrow islands automatically.
        columns = math.ceil(math.sqrt(len(meshes)*3))
        rows = math.ceil(len(meshes)/columns)
        padding = 20/(size*2)
        for index, obj in enumerate(meshes):
            for source, baked_uv in zip(obj.data.uv_layers[0].data, obj.data.uv_layers[-1].data):
                baked_uv.uv = (
                    (index % columns)/columns + padding + source.uv.x*(1/columns-2*padding),
                    (index // columns)/rows + padding + source.uv.y*(1/rows-2*padding),
                )
        # The ISBN manifest owns picking. Join the printed planes after packing
        # so Cycles bakes once and the browser draws this atlas once.
        bpy.ops.object.join()
        joined = bpy.context.object
        joined.name = 'spines'
        meshes = [joined]
        parts.clear()
        parts['spines'] = meshes
    resolution = (
        size * 2 if name in ('objects', 'spines') else size // 2 if name == 'moving' else size
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
    print('BAKING',name,size,samples,flush=True)
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

for name, parts in groups.items():
    if '--spines-only' in args and name != 'spines':
        continue
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
(out/'book-slots.json').write_text(json.dumps(slots, indent=2)+'\n')
print('Room assets exported',flush=True)
