# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import sys, random, os, math, json, tempfile
from collections import Counter
import bpy, bpy_extras
from mathutils import Vector
from clevr_math_utils import Box, clamp, rotate_offset, find_valid_xy

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
    with bpy.data.libraries.load(filename) as (data_from, data_to):
      data_to.objects = data_from.objects
    
    for obj in data_to.objects:
      if obj:
        bpy.context.collection.objects.link(obj)
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

def configure_cycles_device(use_gpu):
  """
  Configure Cycles to use GPU devices when available. Blender 4.x exposes GPU
  selection via bpy.context.preferences instead of user_preferences.
  """
  scene = bpy.context.scene
  print("configure_cycles_device use_gpu", use_gpu)
  try:
    cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
  except KeyError:
    # Cycles is not available or not enabled; nothing to configure.
    return


  if use_gpu:
    # Refresh devices first
    if hasattr(cycles_prefs, 'get_devices'):
      cycles_prefs.get_devices()
    elif hasattr(cycles_prefs, 'refresh_devices'):
      cycles_prefs.refresh_devices()

    # Try CUDA first, then OPTIX
    for backend in ('CUDA', 'OPTIX', 'HIP', 'METAL', 'ONEAPI'):
      try:
        cycles_prefs.compute_device_type = backend
        print(f"Tried setting compute_device_type to {backend}", flush=True)
        
        # Refresh devices again to see what's available for this backend
        if hasattr(cycles_prefs, 'get_devices'):
            cycles_prefs.get_devices()
        elif hasattr(cycles_prefs, 'refresh_devices'):
            cycles_prefs.refresh_devices()
            
        # Check if we have any GPU devices available for this backend
        found_gpu = False
        for device in cycles_prefs.devices:
            if device.type != 'CPU':
                found_gpu = True
                break
        
        if found_gpu:
            print(f"Found GPU devices for backend {backend}", flush=True)
            break
        else:
            print(f"No GPU devices found for backend {backend}", flush=True)
            
      except TypeError:
        continue
        
    # Now enable only the GPU devices
    print("Enabling GPU devices:", flush=True)
    for device in cycles_prefs.devices:
      if device.type != 'CPU':
        device.use = True
        print(f"  - Enabled: {device.name} ({device.type})", flush=True)
      else:
        device.use = False
        print(f"  - Disabled: {device.name} ({device.type})", flush=True)
        
    scene.cycles.device = 'GPU'
    print("Configured Cycles to use GPU with backend", cycles_prefs.compute_device_type, flush=True)
  else:
    scene.cycles.device = 'CPU'


def resolve_object_scale(shape_key, shape_scales, size_dict, fallback_sizes):
  """
  Determine the numeric scale to use for a given shape.
  """
  if shape_scales and shape_key in shape_scales:
    raw_value = shape_scales[shape_key]
    if isinstance(raw_value, str):
      if raw_value not in size_dict:
        raise KeyError("Shape size '%s' not defined in sizes dictionary" % raw_value)
      return size_dict[raw_value]
    try:
      return float(raw_value)
    except (TypeError, ValueError):
      pass
  if fallback_sizes:
    choice = random.choice(fallback_sizes)
    if isinstance(choice, (list, tuple)) and len(choice) >= 2:
      return choice[1]
    return float(choice)
  return 1.0


def get_object_positions(blender_objects):
  """
  Return a list of (x, y, scale) tuples for the currently tracked Blender objects.
  """
  positions = []
  for obj in blender_objects:
    radius = obj.get("footprint_radius", obj.get("clevr_scale", obj.location[2]))
    center = obj.get("center_position")
    if center is None:
      center = (obj.location[0], obj.location[1])
    positions.append((center[0], center[1], radius))
  return positions


def load_shape_radii(path):
  if not os.path.isfile(path):
    raise FileNotFoundError("Precomputed radii file not found: %s" % path)
  with open(path, 'r') as f:
    data = json.load(f)
  if not isinstance(data, dict):
    raise ValueError("Invalid shape radii data: expected dict")
  return data


def get_shape_radius_info(shape_key, radius_cache):
  if radius_cache is None:
    raise RuntimeError("Shape radius cache is not loaded")
  if shape_key not in radius_cache:
    raise RuntimeError("Shape '%s' missing from precomputed radii file" % shape_key)
  entry = radius_cache[shape_key]
  if not isinstance(entry, dict) or 'radius' not in entry:
    raise RuntimeError("Shape '%s' entry malformed in radii file" % shape_key)
  radius = entry['radius']
  offset = entry.get('center_offset', [0.0, 0.0])
  if not isinstance(offset, (list, tuple)) or len(offset) != 2:
    offset = [0.0, 0.0]
  return float(radius), (float(offset[0]), float(offset[1]))


def reset_scene(base_objects, initial_locations):
  # Use a list copy to avoid modification during iteration
  for obj in list(bpy.data.objects):
    # Check if object still exists and is valid
    try:
        if obj.name not in base_objects:
            delete_object(obj)
    except ReferenceError:
        continue
        
  for name, loc in initial_locations.items():
    if name in bpy.data.objects:
      bpy.data.objects[name].location = loc


def camera_view_bounds_2d(scene, cam_ob, me_ob):
    """
    Returns camera space bounding box of mesh object.
    """

    if getattr(me_ob, "type", None) != 'MESH':
        child_meshes = get_mesh_descendants(me_ob)
        boxes = [camera_view_bounds_2d(scene, cam_ob, child) for child in child_meshes]
        boxes = [box for box in boxes if box.width != 0 or box.height != 0]
        if not boxes:
            r = scene.render
            fac = r.resolution_percentage * 0.01
            dim_x = r.resolution_x * fac
            dim_y = r.resolution_y * fac
            return Box(0.0, 0.0, 0.0, 0.0, dim_x, dim_y)
        min_x = min(box.min_x for box in boxes)
        max_x = max(box.max_x for box in boxes)
        min_y = min(box.min_y for box in boxes)
        max_y = max(box.max_y for box in boxes)
        return Box(min_x, min_y, max_x, max_y, boxes[0].dim_x, boxes[0].dim_y)

    mat = cam_ob.matrix_world.normalized().inverted()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = me_ob.evaluated_get(depsgraph)
    me = eval_obj.to_mesh()
    me.transform(me_ob.matrix_world)
    me.transform(mat)

    camera = cam_ob.data
    frame = [-v for v in camera.view_frame(scene=scene)[:3]]
    camera_persp = camera.type != 'ORTHO'

    lx = []
    ly = []

    for v in me.vertices:
        co_local = v.co
        z = -co_local.z

        if camera_persp:
            if z == 0.0:
                lx.append(0.5)
                ly.append(0.5)
            else:
                frame = [(v / (v.z / z)) for v in frame]

        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y

        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)

        lx.append(x)
        ly.append(y)

    min_x = clamp(min(lx), 0.0, 1.0)
    max_x = clamp(max(lx), 0.0, 1.0)
    min_y = clamp(min(ly), 0.0, 1.0)
    max_y = clamp(max(ly), 0.0, 1.0)

    eval_obj.to_mesh_clear()

    r = scene.render
    fac = r.resolution_percentage * 0.01
    dim_x = r.resolution_x * fac
    dim_y = r.resolution_y * fac

    return Box(min_x, min_y, max_x, max_y, dim_x, dim_y)


def render_shadeless(blender_objects, path='flat.png'):
  """
  Render a version of the scene with shading disabled and unique materials
  assigned to all objects, and return a set of all colors that should be in the
  rendered image.
  """
  scene = bpy.context.scene
  render_args = scene.render

  # Cache the render args we are about to clobber
  old_filepath = render_args.filepath
  old_engine = render_args.engine
  old_samples = scene.cycles.samples
  old_device = getattr(scene.cycles, 'device', None)
  old_adaptive = getattr(scene.cycles, 'use_adaptive_sampling', None)

  # Override some render settings to have flat shading
  render_args.filepath = path
  render_args.engine = 'CYCLES'
  scene.cycles.samples = 1
  if old_adaptive is not None:
    scene.cycles.use_adaptive_sampling = False

  # Hide the lights and ground plane so they do not influence the render
  hidden_objects = {}
  for name in ['Lamp_Key', 'Lamp_Fill', 'Lamp_Back', 'Ground']:
    obj = bpy.data.objects.get(name)
    if obj is None:
      continue
    hidden_objects[name] = {
        'viewport': obj.hide_get(),
        'render': getattr(obj, 'hide_render', False),
    }
    obj.hide_set(True)
    if hasattr(obj, 'hide_render'):
      obj.hide_render = True

  # Add random emission materials to all objects
  object_colors = set()
  old_materials = []
  temp_materials = []
  mesh_groups = []
  for i, obj in enumerate(blender_objects):
    meshes = get_mesh_descendants(obj)
    mesh_groups.append(meshes)
    obj_materials = []
    for mesh in meshes:
      obj_materials.append(list(mesh.data.materials))
    old_materials.append(obj_materials)
    if not meshes:
      continue
    mat = bpy.data.materials.new(name='Material_Flat_%d' % i)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    emission_node = nodes.new(type='ShaderNodeEmission')
    while True:
      r, g, b = [random.random() for _ in range(3)]
      if (r, g, b) not in object_colors:
        break
    color = (r, g, b, 1.0)
    object_colors.add((r, g, b))
    emission_node.inputs['Color'].default_value = color
    emission_node.inputs['Strength'].default_value = 1.0
    mat.node_tree.links.new(emission_node.outputs['Emission'],
                            output_node.inputs['Surface'])
    for mesh in meshes:
      mesh.data.materials.clear()
      mesh.data.materials.append(mat)
    temp_materials.append(mat)

  try:
    # Render the scene
    bpy.ops.render.render(write_still=True)
  finally:
    # Restore the materials to objects
    for meshes, mats_per_mesh in zip(mesh_groups, old_materials):
      for mesh, mats in zip(meshes, mats_per_mesh):
        mesh.data.materials.clear()
        for mat in mats:
          if mat is not None:
            mesh.data.materials.append(mat)

    # Clean up the temporary flat materials
    for mat in temp_materials:
      bpy.data.materials.remove(mat, do_unlink=True)

    # Restore visibility for lights and ground
    for name, state in hidden_objects.items():
      obj = bpy.data.objects.get(name)
      if obj is None:
        continue
      obj.hide_set(state['viewport'])
      if hasattr(obj, 'hide_render'):
        obj.hide_render = state['render']

    # Set the render settings back to what they were
    render_args.filepath = old_filepath
    render_args.engine = old_engine
    scene.cycles.samples = old_samples
    if old_adaptive is not None:
      scene.cycles.use_adaptive_sampling = old_adaptive
    if old_device is not None:
      scene.cycles.device = old_device

  return object_colors


def check_visibility(blender_objects, min_pixels_per_object):
  """
  Check whether all objects in the scene have some minimum number of visible
  pixels.
  """
  # Ignore helper meshes such as footprint indicators when checking visibility.
  filtered_objects = [
      obj for obj in blender_objects
      if not getattr(obj, "name", "").startswith("radius_indicator")
  ]
  if not filtered_objects:
    return True

  f, path = tempfile.mkstemp(suffix='.png')
  object_colors = render_shadeless(filtered_objects, path=path)
  img = bpy.data.images.load(path)
  try:
    p = list(img.pixels)
  finally:
    bpy.data.images.remove(img, do_unlink=True)
  color_count = Counter((p[i], p[i+1], p[i+2], p[i+3])
                        for i in range(0, len(p), 4))
  os.remove(path)
  if len(color_count) != len(filtered_objects) + 1:
    return False
  for _, count in color_count.most_common():
    if count < min_pixels_per_object:
      return False
  return True


def add_random_objects(scene_struct, num_objects, args, camera, radius_cache):
  """
  Add random objects to the current blender scene
  """

  # Load the property file
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['colors'].items():
      rgba = [float(c) / 255.0  for c in rgb] + [1.0]
      color_name_to_rgba[name] = rgba
    material_mapping = [(v, k) for k, v in properties['materials'].items()]
    object_mapping = [(v, k) for k, v in properties['shapes'].items()]
    sizes = properties.get('sizes', {})
    size_mapping = list(sizes.items())
    shape_scales = properties.get('shape_scales', {})

  shape_color_combos = None
  if args.shape_color_combos_json is not None:
    with open(args.shape_color_combos_json, 'r') as f:
      shape_color_combos = list(json.load(f).items())

  positions = []
  objects = []
  blender_objects = []
  for i in range(num_objects):
    print(f"Adding object {i+1} of {num_objects}")
    
    # Choose random color and shape
    if shape_color_combos is None:
      obj_name, obj_name_out = random.choice(object_mapping)
      color_name, rgba = random.choice(list(color_name_to_rgba.items()))
    else:
      obj_name_out, color_choices = random.choice(shape_color_combos)
      color_name = random.choice(color_choices)
      obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
      rgba = color_name_to_rgba[color_name]

    r = resolve_object_scale(obj_name_out, shape_scales, sizes, size_mapping)
    print(f"Choosen object: {obj_name_out}, scale: {r}")

    footprint_radius, base_offset = get_shape_radius_info(obj_name_out, radius_cache)
    
    try:
      x, y = find_valid_xy(
          footprint_radius,
          positions,
          scene_struct,
          args,
          max_attempts=args.max_retries
      )
    except RuntimeError:
      for obj in blender_objects:
        delete_object(obj)
      return add_random_objects(scene_struct, num_objects, args, camera, radius_cache)

    # Choose random orientation for the object.
    theta = 360.0 * random.random()

    # Actually add the object to the scene (offset is applied inside utils.add_object)
    rotated_offset = rotate_offset(base_offset, theta)
    add_object(
        args.shape_dir,
        obj_name,
        r,
        (x, y),
        theta=theta,
        center_offset=base_offset,
        footprint_radius=footprint_radius,
        base_center_offset=base_offset,
        center_position=(x, y)
    )
    
    obj = bpy.context.object

    blender_objects.append(obj)
    positions.append((x, y, footprint_radius))
    add_radius_indicator((x, y, obj.location.z), footprint_radius, parent=obj)

    # Record data about the object in the scene data structure
    pixel_coords = get_camera_coords(camera, obj.location)
    bbox = camera_view_bounds_2d(bpy.context.scene, camera, obj)

    objects.append({
      'shape': obj_name_out,
      'scale': r,
      'radius': footprint_radius,
      'center_offset': base_offset,
      '3d_coords': tuple(obj.location),
      'rotation': theta,
      'pixel_coords': pixel_coords,
      'bbox': bbox.to_tuple(),
      'color': color_name,
    })

  # Check that all objects are at least partially visible in the rendered image
  if args.check_visibility == 1:
    all_visible = check_visibility(blender_objects, args.min_pixels_per_object)
  else:
    all_visible = True # Optimization: Skip visibility check for speed
    
  if not all_visible:
    # If any of the objects are fully occluded then start over; delete all
    # objects from the scene and place them all again.
    print('Some objects are occluded; replacing objects')
    for obj in blender_objects:
      delete_object(obj)
    return add_random_objects(scene_struct, num_objects, args, camera, radius_cache)

  return objects, blender_objects
