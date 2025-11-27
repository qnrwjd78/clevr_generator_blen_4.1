# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import sys, random, os, math
import bpy, bpy_extras
from mathutils import Vector


"""
Some utility functions for interacting with Blender
"""


def extract_args(input_argv=None):
  """
  Pull out command-line arguments after "--". Blender ignores command-line flags
  after --, so this lets us forward command line arguments from the blender
  invocation to our own script.
  """
  if input_argv is None:
    input_argv = sys.argv
  output_argv = []
  if '--' in input_argv:
    idx = input_argv.index('--')
    output_argv = input_argv[(idx + 1):]
  return output_argv


def parse_args(parser, argv=None):
  return parser.parse_args(extract_args(argv))


# I wonder if there's a better way to do this?
def delete_object(obj):
  """Delete a specified blender object and any children it parents."""
  if obj is None:
    return
  for child in list(getattr(obj, "children", [])):
    delete_object(child)
  data_block = getattr(obj, "data", None)
  bpy.data.objects.remove(obj, do_unlink=True)
  if data_block and hasattr(data_block, "users") and data_block.users == 0:
    if isinstance(data_block, bpy.types.Mesh):
      bpy.data.meshes.remove(data_block, do_unlink=True)


def get_camera_coords(cam, pos):
  """
  For a specified point, get both the 3D coordinates and 2D pixel-space
  coordinates of the point from the perspective of the camera.

  Inputs:
  - cam: Camera object
  - pos: Vector giving 3D world-space position

  Returns a tuple of:
  - (px, py, pz): px and py give 2D image-space coordinates; pz gives depth
    in the range [-1, 1]
  """
  scene = bpy.context.scene
  x, y, z = bpy_extras.object_utils.world_to_camera_view(scene, cam, pos)
  scale = scene.render.resolution_percentage / 100.0
  w = int(scale * scene.render.resolution_x)
  h = int(scale * scene.render.resolution_y)
  px = int(round(x * w))
  py = int(round(h - y * h))
  return (px, py, z)


def get_mesh_descendants(obj):
  """
  Return a list of all mesh-type descendants for a given object, including
  the object itself if it is a mesh.
  """
  meshes = []
  if obj is None:
    return meshes

  def _collect(current):
    if current is None:
      return
    if getattr(current, "type", None) == 'MESH' and getattr(current, "data", None) is not None:
      meshes.append(current)
    for child in getattr(current, "children", []):
      _collect(child)

  _collect(obj)
  return meshes

def combine_collection(object_dir, name):
  # object count
  count = 0

  for obj in bpy.data.objects:
    if obj.name.startswith(name):
      count += 1

  # load collection
  filename = os.path.join(object_dir, '%s.blend' % name)

  collection_names = []
  object_names = []
  try:
    with bpy.data.libraries.load(filename) as (data_from, _):
      collection_names = list(data_from.collections)
      object_names = [obj_name for obj_name in data_from.objects if obj_name]
  except Exception as exc:
    print("!! Unable to read datablocks in blend file:", filename, exc)
    return None

  prev_objects = set(bpy.data.objects)

  if collection_names:
    col_name = collection_names[0]
    directory = os.path.join(filename, 'Collection')
    bpy.ops.wm.append(
      filepath=os.path.join(directory, col_name),
      directory=directory,
      filename=col_name
    )
  elif object_names:
    directory = os.path.join(filename, 'Object')
    for obj_name in object_names:
      bpy.ops.wm.append(
        filepath=os.path.join(directory, obj_name),
        directory=directory,
        filename=obj_name
      )
  else:
    print("!! No collections or objects to append in blend file:", filename)
    return None

  new_objects = [obj for obj in bpy.data.objects if obj not in prev_objects]
  mesh_objs = [obj for obj in new_objects if obj.type == "MESH"]
  if not mesh_objs:
    print("!! No mesh objects found after appending:", filename)
    return None

  new_name = '%s_%d' % (name, count)
  count += 1
  parent = bpy.data.objects.new(new_name, None)
  parent.empty_display_type = 'PLAIN_AXES'
  bpy.context.scene.collection.objects.link(parent)

  for obj in new_objects:
    obj.parent = parent
    obj.matrix_parent_inverse = parent.matrix_world.inverted()
  
  return parent.name


def _compute_group_min_z(obj):
  meshes = get_mesh_descendants(obj)
  if not meshes:
    return 0.0
  min_z = None
  for mesh in meshes:
    for vert in mesh.data.vertices:
      z_val = (mesh.matrix_world @ vert.co).z
      if min_z is None or z_val < min_z:
        min_z = z_val
  return min_z if min_z is not None else 0.0

def compute_xy_radius(obj, depsgraph=None):
  """
  Measure the maximum XY extent of obj (including mesh children) and return
  the distance from the center of that span to the furthest vertex so placement
  can use a geometry-aware footprint.
  """
  meshes = get_mesh_descendants(obj)
  if not meshes:
    # Fall back to the largest horizontal scale component if no meshes exist.
    return max(abs(obj.scale[0]), abs(obj.scale[1]))

  if depsgraph is None:
    depsgraph = bpy.context.evaluated_depsgraph_get()

  min_x = math.inf
  max_x = -math.inf
  min_y = math.inf
  max_y = -math.inf

  for mesh_obj in meshes:
    eval_obj = mesh_obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    world_mat = mesh_obj.matrix_world
    for vert in mesh.vertices:
      world_co = world_mat @ vert.co
      min_x = min(min_x, world_co.x)
      max_x = max(max_x, world_co.x)
      min_y = min(min_y, world_co.y)
      max_y = max(max_y, world_co.y)
    eval_obj.to_mesh_clear()

  if not (math.isfinite(min_x) and math.isfinite(max_x) and
          math.isfinite(min_y) and math.isfinite(max_y)):
    return max(abs(obj.scale[0]), abs(obj.scale[1]))

  center_x = 0.5 * (min_x + max_x)
  center_y = 0.5 * (min_y + max_y)
  max_radius = 0.0
  for mesh_obj in meshes:
    eval_obj = mesh_obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    world_mat = mesh_obj.matrix_world
    for vert in mesh.vertices:
      world_co = world_mat @ vert.co
      dx = world_co.x - center_x
      dy = world_co.y - center_y
      dist = math.hypot(dx, dy)
      if dist > max_radius:
        max_radius = dist
    eval_obj.to_mesh_clear()

  return max_radius if max_radius > 0 else max(abs(obj.scale[0]), abs(obj.scale[1]))



def add_object(object_dir, name, scale, loc, theta=0,
               center_offset=None, footprint_radius=None,
               base_center_offset=None, center_position=None):
  """
  Load an object from a file. We assume that in the directory object_dir, there
  is a file named "$name.blend" which contains a single object named "$name"
  that has unit size and is centered at the origin.

  - scale: scalar giving the size that the object should be in the scene
  - loc: tuple (x, y) giving the coordinates on the ground plane where the
    object should be placed.
  """
  # First figure out how many of this object are already in the scene so we can
  # give the new object a unique name
  # count = 0
  # for obj in bpy.data.objects:
  #   if obj.name.startswith(name):
  #     count += 1

  # filename = os.path.join(object_dir, '%s.blend' % name, 'Object', name)
  # bpy.ops.wm.append(filename=filename)
  loaded_name = combine_collection(object_dir, name)
  if loaded_name is None:
    raise RuntimeError("Unable to append object '%s' from %s" % (name, object_dir))

  # Set the new object as active, then rotate, scale, and translate it
  x, y = loc
  obj = bpy.data.objects[loaded_name]
  bpy.ops.object.select_all(action='DESELECT')
  obj.select_set(True)
  bpy.context.view_layer.objects.active = obj
  obj.rotation_euler[2] = theta
  obj.scale = (scale, scale, scale)
  obj.location = (x, y, 0)
  if hasattr(bpy.context.view_layer, "update"):
    bpy.context.view_layer.update()

  min_z = _compute_group_min_z(obj)
  obj.location.z -= min_z

  if center_offset is not None:
    rot_mat = obj.matrix_world.to_3x3().normalized()
    offset_local = Vector((center_offset[0], center_offset[1], 0)) 
    offset_world = rot_mat @ offset_local
    obj.location -= offset_world

  obj["clevr_scale"] = compute_xy_radius(obj)
  
  if footprint_radius is not None:
    obj["footprint_radius"] = footprint_radius
  if base_center_offset is not None:
    obj["center_offset"] = base_center_offset
  if center_position is not None:
    obj["center_position"] = center_position
  print(f"Final object scale: {obj['clevr_scale']}")
  return obj["clevr_scale"]


def load_materials(material_dir):
  """
  Load materials from a directory. We assume that the directory contains .blend
  files with one material each. The file X.blend has a single NodeTree item named
  X; this NodeTree item must have a "Color" input that accepts an RGBA value.
  """
  for fn in os.listdir(material_dir):
    if not fn.endswith('.blend'): continue
    name = os.path.splitext(fn)[0]
    filepath = os.path.join(material_dir, fn, 'NodeTree', name)
    bpy.ops.wm.append(filename=filepath)


def add_material(name, relchange=False, **properties):
  """
  Create a new material and assign it to the active object. "name" should be the
  name of a material that has been previously loaded using load_materials.
  """
  # Figure out how many materials are already in the scene
  mat_count = len(bpy.data.materials)

  # Create a new material datablock explicitly so we do not depend on context
  mat = bpy.data.materials.new(name='Material_%d' % mat_count)
  mat.use_nodes = True

  # Attach the new material to every mesh descendant of the active object.
  obj = bpy.context.active_object
  targets = get_mesh_descendants(obj)
  if not targets:
    return
  if not relchange:
    for target in targets:
      target.data.materials.clear()
  for target in targets:
    target.data.materials.append(mat)

  # Find the output node of the new material
  output_node = None
  for n in mat.node_tree.nodes:
    if n.name == 'Material Output':
      output_node = n
      break

  # Add a new GroupNode to the node tree of the active material,
  # and copy the node tree from the preloaded node group to the
  # new group node. This copying seems to happen by-value, so
  # we can create multiple materials of the same type without them
  # clobbering each other
  group_node = mat.node_tree.nodes.new('ShaderNodeGroup')
  group_node.node_tree = bpy.data.node_groups[name]

  # Find and set the "Color" input of the new group node
  for inp in group_node.inputs:
    if inp.name in properties:
      inp.default_value = properties[inp.name]

  # Wire the output of the new group node to the input of
  # the MaterialOutput node
  mat.node_tree.links.new(
      group_node.outputs['Shader'],
      output_node.inputs['Surface'],
  )


def add_radius_indicator(location, radius, color=(1.0, 0.0, 0.0, 1.0), height_offset=0.02, parent=None):
  """
  Create a thin torus hovering slightly above the ground at the given location.
  """
  if location is None or radius is None or radius <= 0:
    return None

  bpy.ops.object.select_all(action='DESELECT')
  lx, ly, lz = location
  minor_radius = 0.01
  bpy.ops.mesh.primitive_torus_add(
      major_radius=radius,
      minor_radius=minor_radius,
      location=(lx, ly, lz + height_offset)
  )
  ring = bpy.context.object
  ring.name = "radius_indicator"
  if parent is not None:
    ring.parent = parent
    ring.matrix_parent_inverse = parent.matrix_world.inverted()

  mat = bpy.data.materials.new(name=f"{ring.name}_mat")
  mat.use_nodes = True
  nodes = mat.node_tree.nodes
  nodes.clear()
  emission = nodes.new(type='ShaderNodeEmission')
  emission.inputs['Color'].default_value = color
  emission.inputs['Strength'].default_value = 5.0
  output = nodes.new(type='ShaderNodeOutputMaterial')
  mat.node_tree.links.new(emission.outputs['Emission'], output.inputs['Surface'])
  ring.data.materials.clear()
  ring.data.materials.append(mat)

  return ring
