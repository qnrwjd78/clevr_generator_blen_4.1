# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#
# Modification copyright 2020 Azade Farshad
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import math, sys, random, argparse, json, os, tempfile
from datetime import datetime as dt
from collections import Counter

"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

INSIDE_BLENDER = True
try:
  import bpy, bpy_extras
  from mathutils import Vector
except ImportError as e:
  INSIDE_BLENDER = False
if INSIDE_BLENDER:
  try:
    import utils
    print("USING UTILS:", utils.__file__)
  except ImportError as e:
    print("\nERROR")
    print("Running render_images.py from Blender and cannot import utils.py.") 
    print("You may need to add a .pth file to the site-packages of Blender's")
    print("bundled python with a command like this:\n")
    print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
    print("\nWhere $BLENDER is the directory where Blender is installed, and")
    print("$VERSION is your Blender version (such as 2.78).")
    sys.exit(1)

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--base_scene_blendfile', default='data/base_scene.blend',
    help="Base blender file on which all scenes are based; includes " +
          "ground plane, lights, and camera.")
parser.add_argument('--properties_json', default='data/properties.json',
    help="JSON file defining objects, materials, sizes, and colors. " +
         "The \"colors\" field maps from CLEVR color names to RGB values; " +
         "The \"sizes\" field maps from CLEVR size names to scalars used to " +
         "rescale object models; the \"materials\" and \"shapes\" fields map " +
         "from CLEVR material and shape names to .blend files in the " +
         "--object_material_dir and --shape_dir directories respectively.")
parser.add_argument('--shape_dir', default='data/shapes',
    help="Directory where .blend files for object models are stored")
parser.add_argument('--material_dir', default='data/materials',
    help="Directory where .blend files for materials are stored")
parser.add_argument('--shape_color_combos_json', default=None,
    help="Optional path to a JSON file mapping shape names to a list of " +
         "allowed color names for that shape. This allows rendering images " +
         "for CLEVR-CoGenT.")
parser.add_argument('--shape_radii_json', default='data/shape_radii.json',
    help="Path to JSON file containing precomputed footprint radii and offsets.")

# Settings for objects
parser.add_argument('--min_objects', default=3, type=int,
    help="The minimum number of objects to place in each scene")
parser.add_argument('--max_objects', default=7, type=int,
    help="The maximum number of objects to place in each scene")
parser.add_argument('--min_dist', default=0.25, type=float,
    help="The minimum allowed distance between object centers")
parser.add_argument('--margin', default=0.4, type=float,
    help="Along all cardinal directions (left, right, front, back), all " +
         "objects will be at least this distance apart. This makes resolving " +
         "spatial relationships slightly less ambiguous.")
parser.add_argument('--min_pixels_per_object', default=200, type=int,
    help="All objects will have at least this many visible pixels in the " +
         "final rendered images; this ensures that no objects are fully " +
         "occluded by other objects.")
parser.add_argument('--max_retries', default=30000, type=int,
    help="The number of times to try placing an object before giving up and " +
         "re-placing all objects in the scene.")

# Output settings
parser.add_argument('--start_idx', default=0, type=int,
    help="The index at which to start for numbering rendered images. Setting " +
         "this to non-zero values allows you to distribute rendering across " +
         "multiple machines and recombine the results later.")
parser.add_argument('--num_images', default=5, type=int,
    help="The number of images to render")
parser.add_argument('--filename_prefix', default='CLEVR',
    help="This prefix will be prepended to the rendered images and JSON scenes")
parser.add_argument('--split', default='new',
    help="Name of the split for which we are rendering. This will be added to " +
         "the names of rendered images, and will also be stored in the JSON " +
         "scene structure for each image.")
parser.add_argument('--output_image_dir', default='../output/images/',
    help="The directory where output images will be stored. It will be " +
         "created if it does not exist.")
parser.add_argument('--output_scene_dir', default='../output/scenes/',
    help="The directory where output JSON scene structures will be stored. " +
         "It will be created if it does not exist.")
parser.add_argument('--output_scene_file', default='../output/CLEVR_scenes.json',
    help="Path to write a single JSON file containing all scene information")
parser.add_argument('--output_blend_dir', default='output/blendfiles',
    help="The directory where blender scene files will be stored, if the " +
         "user requested that these files be saved using the " +
         "--save_blendfiles flag; in this case it will be created if it does " +
         "not already exist.")
parser.add_argument('--save_blendfiles', type=int, default=0,
    help="Setting --save_blendfiles 1 will cause the blender scene file for " +
         "each generated image to be stored in the directory specified by " +
         "the --output_blend_dir flag. These files are not saved by default " +
         "because they take up ~5-10MB each.")
parser.add_argument('--version', default='1.0',
    help="String to store in the \"version\" field of the generated JSON file")
parser.add_argument('--license',
    default="Creative Commons Attribution (CC-BY 4.0)",
    help="String to store in the \"license\" field of the generated JSON file")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
    help="String to store in the \"date\" field of the generated JSON file; " +
         "defaults to today's date")
parser.add_argument('--mode', default='rel_change', choices=['rel_change','removal', 'swap', 'addition', 'replacement'],
    help="Image manipulation mode")

# Rendering options
parser.add_argument('--use_gpu', default=0, type=int,
    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
         "to work.")
parser.add_argument('--width', default=320, type=int,
    help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=240, type=int,
    help="The height (in pixels) for the rendered images")
parser.add_argument('--key_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--camera_jitter', default=0.5, type=float,
    help="The magnitude of random jitter to add to the camera position")
parser.add_argument('--render_num_samples', default=512, type=int,
    help="The number of samples to use when rendering. Larger values will " +
         "result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_min_bounces', default=8, type=int,
    help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
    help="The maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=256, type=int,
    help="The tile size to use for rendering. This should not affect the " +
         "quality of the rendered image but may affect the speed; CPU-based " +
         "rendering may achieve better performance using smaller tile sizes " +
         "while larger tile sizes may be optimal for GPU-based rendering.")


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
    
    if hasattr(cycles_prefs, 'get_devices'):
      cycles_prefs.get_devices()
    # Try the common GPU backends in order of typical availability.
    for backend in ('CUDA', 'OPTIX', 'HIP', 'METAL', 'ONEAPI'):
      try:
        cycles_prefs.compute_device_type = backend
        break
      except TypeError:
        continue
    if hasattr(cycles_prefs, 'get_devices'):
      cycles_prefs.get_devices()
    for device in getattr(cycles_prefs, 'devices', []):
      device.use = True
    scene.cycles.device = 'GPU'
  else:
    scene.cycles.device = 'CPU'


def resolve_object_scale(shape_key, shape_scales, size_dict, fallback_sizes):
  """
  Determine the numeric scale to use for a given shape.

  Returns a float scale value. Shape-specific overrides in shape_scales may
  reference named sizes from size_dict or provide explicit numeric values.
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
  The stored custom property 'clevr_scale' is preferred when available so that
  downstream placement logic has access to the true footprint radius.
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


def rotate_offset(offset, theta_degrees):
  if offset is None:
    return (0.0, 0.0)
  ox, oy = offset
  theta = math.radians(theta_degrees)
  print("rotate_offset:", theta_degrees, theta)
  cos_t = math.cos(theta)
  sin_t = math.sin(theta)
  return (
      ox * cos_t - oy * sin_t,
      ox * sin_t + oy * cos_t,
  )


def find_valid_xy(radius, positions, scene_struct, args,
                  min_distance_from=None, min_distance=0.0, max_attempts=None):
  """
  Sample a valid (x, y) location that keeps the new object far enough away from
  existing placements and optionally enforces a minimum displacement from a
  reference point (used when relocating an object).
  """
  attempts = 0
  directions = scene_struct['directions']
  while True:
    attempts += 1
    if max_attempts is not None and attempts > max_attempts:
      raise RuntimeError("Max attempts exceeded while sampling object position")

    x = random.uniform(-3, 3)
    y = random.uniform(-3, 3)

    if min_distance_from is not None and min_distance > 0:
      dx = x - min_distance_from[0]
      dy = y - min_distance_from[1]
      if math.sqrt(dx * dx + dy * dy) < min_distance:
        continue

    valid = True
    for (xx, yy, rr) in positions:
      dx = x - xx
      dy = y - yy
      dist = math.sqrt(dx * dx + dy * dy)
      if dist - radius - rr < args.min_dist:
        valid = False
        break
      for direction_name in ['left', 'right', 'front', 'behind']:
        direction_vec = directions[direction_name]
        assert direction_vec[2] == 0
        margin = dx * direction_vec[0] + dy * direction_vec[1]
        if 0 < margin < args.margin:
          valid = False
          break
      if not valid:
        break

    if valid:
      return x, y

def main(args):
  num_digits = 6
  prefix = '%s_%s_' % (args.filename_prefix, args.split)
  img_template_source = '%s%%0%dd_source.png' % (prefix, num_digits)
  scene_template_source = '%s%%0%dd_source.json' % (prefix, num_digits)
  blend_template_source = '%s%%0%dd_source.blend' % (prefix, num_digits)

  img_template_source = os.path.join(args.output_image_dir, img_template_source)
  scene_template_source = os.path.join(args.output_scene_dir, scene_template_source)
  blend_template_source = os.path.join(args.output_blend_dir, blend_template_source)

  img_template_target = '%s%%0%dd_target.png' % (prefix, num_digits)
  scene_template_target = '%s%%0%dd_target.json' % (prefix, num_digits)
  blend_template_target = '%s%%0%dd_target.blend' % (prefix, num_digits)

  img_template_target = os.path.join(args.output_image_dir, img_template_target)
  scene_template_target = os.path.join(args.output_scene_dir, scene_template_target)
  blend_template_target = os.path.join(args.output_blend_dir, blend_template_target)

  if not os.path.isdir(args.output_image_dir):
    os.makedirs(args.output_image_dir)
  if not os.path.isdir(args.output_scene_dir):
    os.makedirs(args.output_scene_dir)
  if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
    os.makedirs(args.output_blend_dir)

  radius_cache = load_shape_radii(args.shape_radii_json)
  
  all_scene_paths = []
  for i in range(args.num_images):
    img_path_source = img_template_source % (i + args.start_idx)
    img_path_target = img_template_target % (i + args.start_idx)

    scene_path_source = scene_template_source % (i + args.start_idx)
    scene_path_target = scene_template_target % (i + args.start_idx)

    all_scene_paths.append(scene_path_source)
    all_scene_paths.append(scene_path_target)
    blend_path_source = None
    if args.save_blendfiles == 1:
      blend_path_source = blend_template_source % (i + args.start_idx)
      blend_path_target = blend_template_target % (i + args.start_idx)
    num_objects = random.randint(args.min_objects, args.max_objects)
    render_scene(args,
      num_objects=num_objects,
      output_index=(i + args.start_idx),
      output_split=args.split,
      output_image=img_path_source,
      output_image_target=img_path_target,
      output_scene_source=scene_path_source,
      output_scene_target=scene_path_target,
      output_blendfile=blend_path_source,
      radius_cache=radius_cache,
    )

  # After rendering all images, combine the JSON files for each scene into a
  # single JSON file.
  all_scenes = []
  for scene_path in all_scene_paths:
    with open(scene_path, 'r') as f:
      all_scenes.append(json.load(f))
  output = {
    'info': {
      'date': args.date,
      'version': args.version,
      'split': args.split,
      'license': args.license,
    },
    'scenes': all_scenes
  }
  with open(args.output_scene_file, 'w') as f:
    json.dump(output, f)

def render_scene(args,
    num_objects=5,
    output_index=0,
    output_split='none',
    output_image='render_source.png',
    output_image_target='render_target.png',
    output_scene_source='render_source_json',
    output_scene_target='render_target_json',
    output_blendfile=None,
    radius_cache=None,
  ):

  # Load the main blendfile
  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

  # Load materials
  utils.load_materials(args.material_dir)

  scene = bpy.context.scene

  # Set render arguments so we can get pixel coordinates later.
  # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
  # cannot be used.
  render_args = scene.render
  render_args.engine = "CYCLES"
  render_args.filepath = output_image
  render_args.resolution_x = args.width
  render_args.resolution_y = args.height
  render_args.resolution_percentage = 100
  if hasattr(render_args, 'tile_x'):
    render_args.tile_x = args.render_tile_size
    render_args.tile_y = args.render_tile_size
  else:
    cycles_tiles = getattr(scene, 'cycles', None)
    if cycles_tiles is not None:
      if hasattr(cycles_tiles, 'tile_x'):
        cycles_tiles.tile_x = args.render_tile_size
        cycles_tiles.tile_y = args.render_tile_size
      elif hasattr(cycles_tiles, 'tile_size'):
        cycles_tiles.tile_size = args.render_tile_size

  configure_cycles_device(args.use_gpu == 1)

  # Some CYCLES-specific stuff
  world = scene.world
  if world is not None and getattr(world, 'cycles', None) is not None:
    world.cycles.sample_as_light = True
  scene.cycles.blur_glossy = 2.0
  scene.cycles.samples = args.render_num_samples
  scene.cycles.transparent_min_bounces = args.render_min_bounces
  scene.cycles.transparent_max_bounces = args.render_max_bounces

  # This will give ground-truth information about the scene and its objects
  scene_struct = {
      'split': output_split,
      'image_index': output_index,
      'image_filename': os.path.basename(output_image),
      'objects': [],
      'directions': {},
  }

  # Put a plane on the ground so we can compute cardinal directions
  bpy.ops.mesh.primitive_plane_add(size=10)
  plane = bpy.context.object

  def rand(L):
    return 2.0 * L * (random.random() - 0.5)

  # Add random jitter to camera position
  if args.camera_jitter > 0:
    for i in range(3):
      bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

  # Figure out the left, up, and behind directions along the plane and record
  # them in the scene structure
  camera = bpy.data.objects['Camera']
  plane_normal = plane.data.vertices[0].normal
  cam_rot = camera.matrix_world.to_quaternion()
  cam_behind = cam_rot @ Vector((0, 0, -1))
  cam_left = cam_rot @ Vector((-1, 0, 0))
  cam_up = cam_rot @ Vector((0, 1, 0))
  plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
  plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
  plane_up = cam_up.project(plane_normal).normalized()

  # Delete the plane; we only used it for normals anyway. The base scene file
  # contains the actual ground plane.
  utils.delete_object(plane)

  # Save all six axis-aligned directions in the scene struct
  scene_struct['directions']['behind'] = tuple(plane_behind)
  scene_struct['directions']['front'] = tuple(-plane_behind)
  scene_struct['directions']['left'] = tuple(plane_left)
  scene_struct['directions']['right'] = tuple(-plane_left)
  scene_struct['directions']['above'] = tuple(plane_up)
  scene_struct['directions']['below'] = tuple(-plane_up)

  # Add random jitter to lamp positions
  if args.key_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
  if args.back_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
  if args.fill_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

  # Now make some random objects
  objects, blender_objects = add_random_objects(scene_struct, num_objects, args, camera, radius_cache)

  # Render the scene and dump the scene data structure
  scene_struct['objects'] = objects
  scene_struct['relationships'] = compute_all_relationships(scene_struct)
  while True:
    try:
      bpy.ops.render.render(write_still=True)
      break
    except Exception as e:
      print(e)

  with open(output_scene_source, 'w') as f:
    json.dump(scene_struct, f, indent=2)

  #if output_blendfile is not None:
  #  bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)

  #Add the transformations to the objects

  #'rel_change','removal', 'swap', 'addition', 'replacement'
  #replacement
  if args.mode == "replacement":
    blender_objects, scene_struct = replace_random_object(scene_struct, blender_objects, camera, args, radius_cache)
  #relationship change
  elif args.mode == "rel_change":
    blender_objects, scene_struct = relative_transform_objects(scene_struct, blender_objects, camera, args)
  elif args.mode == "removal":
    blender_objects, scene_struct = remove_random_object(scene_struct, blender_objects, camera)
  elif args.mode == "swap":
    blender_objects, scene_struct = swap_objects(scene_struct, blender_objects, camera)
  elif args.mode == "addition":
    blender_objects, scene_struct = add_objects(scene_struct, blender_objects, camera, args, radius_cache)
  else:
    print("not ready yet")
  
  #scene_struct['objects'] = objects
  scene_struct['relationships'] = compute_all_relationships(scene_struct)
  
  render_args.filepath = output_image_target
  #Render and save scene again
  while True:
    try:
      bpy.ops.render.render(write_still=True)
      break
    except Exception as e:
      print(e)

  with open(output_scene_target, 'w') as f:
    json.dump(scene_struct, f, indent=2)


def transform_random_objects(scene_struct, blender_objects, camera):
  c = 0
  objects = scene_struct['objects']
  positions = get_object_positions(blender_objects)
  for obj in blender_objects:
    loc = obj.location
    x_t = min(max(loc[0] + random.uniform(-1.5, 1.5),-3),3)
    y_t = min(max(loc[1] + random.uniform(-1.5, 1.5),-3),3)
    obj.location = Vector((x_t,y_t,loc[2]))
    pixel_coords = utils.get_camera_coords(camera, obj.location)
    objects[c]['3d_coords'] = tuple(obj.location)
    objects[c]['pixel_coords'] = pixel_coords
    c = c + 1
  scene_struct['objects'] = objects
  return blender_objects, scene_struct

def replace_random_object(scene_struct, blender_objects, camera, args, radius_cache):

  num_objs = len(blender_objects)
  n_rand = int(random.uniform(0,1) * num_objs)
  
  obj = blender_objects[n_rand]
  loc = obj.location
  x, y = loc[0], loc[1]
  old_shape = scene_struct['objects'][n_rand]['shape']
  old_color = scene_struct['objects'][n_rand]['color']
  utils.delete_object(obj)
  
  scene_struct['objects'].pop(n_rand)
  blender_objects.pop(n_rand)
  objects = scene_struct['objects']
  positions = get_object_positions(blender_objects)
  
  # Load the property file
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['colors'].items():
      rgba = [float(c) / 255.0 for c in rgb] + [1.0]
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

  # Choose random color and shape
  if shape_color_combos is None:
    obj_name_out = old_shape
    color_name = old_color
    while obj_name_out == old_shape:
      obj_name, obj_name_out = random.choice(object_mapping)
    while color_name == old_color:
      color_name, rgba = random.choice(list(color_name_to_rgba.items()))
  else:
    obj_name_out = old_shape
    color_name = old_color
    while obj_name_out == old_shape:
      obj_name_out, color_choices = random.choice(shape_color_combos)
    while color_name == old_color:
      color_name = random.choice(color_choices)
    obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
    rgba = color_name_to_rgba[color_name]

  r = resolve_object_scale(obj_name_out, shape_scales, sizes, size_mapping)
  footprint_radius, base_offset = get_shape_radius_info(obj_name_out, radius_cache)

  # Choose random orientation for the object.
  theta = 360.0 * random.random()

  # Actually add the object to the scene
  rotated_offset = rotate_offset(base_offset, theta)
  utils.add_object(
      args.shape_dir,
      obj_name,
      r,
      (x, y),
      theta=theta,
      center_offset=rotated_offset,
      footprint_radius=footprint_radius,
      base_center_offset=base_offset,
      center_position=(x, y)
  )
  obj = bpy.context.active_object
  blender_objects.append(obj)
  positions.append((x, y, footprint_radius))
  utils.add_radius_indicator((x, y, obj.location.z), footprint_radius, parent=obj)

  # Attach a random material
  # mat_name, mat_name_out = random.choice(material_mapping)
  # utils.add_material(mat_name, Color=rgba)

  # Record data about the object in the scene data structure
  pixel_coords = utils.get_camera_coords(camera, obj.location)
  bbox = camera_view_bounds_2d(bpy.context.scene, camera, obj) 

  objects.append({
      'shape': obj_name_out,
      'scale': r,
      'radius': footprint_radius,
      'center_offset': base_offset,
      'material': mat_name_out,
      '3d_coords': tuple(obj.location),
      'rotation': theta,
      'pixel_coords': pixel_coords,
      'bbox': bbox.to_tuple(),
      'color': color_name,
    })
  scene_struct['objects'] = objects
  return blender_objects, scene_struct

def relchange_random(scene_struct, blender_objects, camera, args):

  num_objs = len(blender_objects)
  n_rand = int(random.uniform(0,1) * num_objs)
  
  obj = blender_objects[n_rand]
  loc = obj.location
  r = obj.get("footprint_radius", obj.get("clevr_scale", loc[2]))
  shape_data = scene_struct['objects'][n_rand]
  shape = shape_data['shape']
  color = shape_data['color']
  mat_name = shape_data['material']
  theta = shape_data['rotation']
  footprint_radius = shape_data.get('radius', r)
  base_offset = shape_data.get('center_offset', (0.0, 0.0))
  utils.delete_object(obj)
  
  scene_struct['objects'].pop(n_rand)
  blender_objects.pop(n_rand)
  objects = scene_struct['objects']
  positions = get_object_positions(blender_objects)
  
  x, y = find_valid_xy(
      r,
      positions,
      scene_struct,
      args,
      min_distance_from=(loc[0], loc[1]),
      min_distance=4.0
  )

  # Load the property file
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['colors'].items():
      rgba = [float(c) / 255.0 for c in rgb] + [1.0]
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
  
  shape_name = properties['shapes'][shape]
  rgba = color_name_to_rgba[color]
  # Actually add the object to the scene
  rotated_offset = rotate_offset(base_offset, theta)
  utils.add_object(
      args.shape_dir,
      shape_name,
      r,
      (x, y),
      theta=theta,
      center_offset=rotated_offset,
      footprint_radius=footprint_radius,
      base_center_offset=base_offset,
      center_position=(x, y)
  )
  obj = bpy.context.active_object
  blender_objects.append(obj)
  positions.append((x, y, footprint_radius))

  # utils.add_material(properties['materials'][mat_name], Color=rgba)

  # Record data about the object in the scene data structure
  pixel_coords = utils.get_camera_coords(camera, obj.location)
  bbox = camera_view_bounds_2d(bpy.context.scene, camera, obj) 

  objects.append({
      'shape': shape, #obj_name_out
      'scale': r,
      'radius': footprint_radius,
      'center_offset': base_offset,
      'material': mat_name, #_out
      '3d_coords': tuple(obj.location),
      'rotation': theta,
      'pixel_coords': pixel_coords,
      'bbox': bbox.to_tuple(),
      'color': color, #_name
    })
  scene_struct['objects'] = objects
  return blender_objects, scene_struct

def relative_transform_objects(scene_struct, blender_objects, camera, args):

  num_objs = len(blender_objects)
  n_rand = int(random.uniform(0,1) * num_objs)

  #objects = scene_struct['objects']
  positions = get_object_positions(blender_objects)
  
  #num_objs = len(objects)
  #curr_obj = 0#int(random.uniform(0, num_objs))
  obj = blender_objects[n_rand]
  #sc_obj = scene_struct['objects'].pop(n_rand)
  objects = scene_struct['objects']
  loc = obj.location
  r = obj.get("footprint_radius", obj.get("clevr_scale", loc[2]))
  
  #loc_target = 2 * blender_objects[1].location
  #diff = loc_target - loc
  #print("diff: ", diff)
  #x_t = min(max(diff[0],-3),3)
  #y_t = min(max(diff[1],-3),3)
  
  while True:
    try:
      x, y = find_valid_xy(
          r,
          positions,
          scene_struct,
          args,
          min_distance_from=(loc[0], loc[1]),
          min_distance=4.0,
          max_attempts=10
      )
      break
    except RuntimeError:
      n_rand = int(random.uniform(0,1) * num_objs)
      obj = blender_objects[n_rand]
      loc = obj.location
      r = obj.get("footprint_radius", obj.get("clevr_scale", loc[2]))
      
      
  
  obj.location = Vector((x,y,loc[2]))
  blender_objects[n_rand] = obj
  # Render the scene
  bpy.ops.render.render(write_still=False)

  pixel_coords = utils.get_camera_coords(camera, obj.location)
  bbox = camera_view_bounds_2d(bpy.context.scene, camera, obj)
  #print(sc_obj)
  objects[n_rand]['3d_coords'] = tuple(obj.location)
  objects[n_rand]['pixel_coords'] = pixel_coords
  objects[n_rand]['bbox'] = bbox.to_tuple()
  #print(sc_obj)
  #input()
  #sc_obj['3d_coords'] = tuple(obj.location)
  #sc_obj['pixel_coords'] = pixel_coords
  #sc_obj['bbox'] = bbox.to_tuple()
  #objects.append(sc_obj)
  scene_struct['objects'] = objects
  return blender_objects, scene_struct

def remove_random_object(scene_struct, blender_objects, camera):

  #objects = scene_struct['objects']
  #positions = [obj.location for obj in blender_objects]
  
  num_objs = len(blender_objects)
  n_rand = int(random.uniform(0,1) * num_objs)
  
  print("object chosen: ", str(n_rand), " of ", str(num_objs))
  utils.delete_object(blender_objects[n_rand])
  
  scene_struct['objects'].pop(n_rand)
  blender_objects.pop(n_rand)

  return blender_objects, scene_struct

def add_objects(scene_struct, blender_objects, camera, args, radius_cache):

  num_objects = 1
  # Load the property file
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['colors'].items():
      rgba = [float(c) / 255.0 for c in rgb] + [1.0]
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

  objects = scene_struct['objects']
  positions = get_object_positions(blender_objects)
 
  for i in range(num_objects):
    
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
    footprint_radius, base_offset = get_shape_radius_info(obj_name_out, radius_cache)

    # Try to place the object, ensuring that we don't intersect any existing
    # objects and that we are more than the desired margin away from all existing
    # objects along all cardinal directions.
    while True:
      try:
        x, y = find_valid_xy(
            footprint_radius,
            positions,
            scene_struct,
            args,
            max_attempts=args.max_retries
        )
        break
      except RuntimeError:
        print("*" * 20, " Num tries reached!!")

    # For cube, adjust the size a bit
    if obj_name == 'Cube':
      r /= math.sqrt(2)

    # Choose random orientation for the object.
    theta = 360.0 * random.random()

    rotated_offset = rotate_offset(base_offset, theta)
    utils.add_object(
        args.shape_dir,
        obj_name,
        r,
        (x, y),
        theta=theta,
        center_offset=rotated_offset,
        footprint_radius=footprint_radius,
        base_center_offset=base_offset,
        center_position=(x, y)
    )
    obj = bpy.context.object
    blender_objects.append(obj)
    positions.append((x, y, footprint_radius))
    utils.add_radius_indicator((x, y, obj.location.z), footprint_radius, parent=obj)

    # # Attach a random material
    # mat_name, mat_name_out = random.choice(material_mapping)
    # utils.add_material(mat_name, Color=rgba)

    # Record data about the object in the scene data structure
    pixel_coords = utils.get_camera_coords(camera, obj.location)
    bbox = camera_view_bounds_2d(bpy.context.scene, camera, obj) #bpy_extras.object_utils.world_to_camera_view(scene, obj, coord)
    #print("*"* 10, b2, pixel_coords)
    #break
    objects.append({
      'shape': obj_name_out,
      'scale': r,
      'radius': footprint_radius,
      'center_offset': base_offset,
      'material': mat_name_out,
      '3d_coords': tuple(obj.location),
      'rotation': theta,
      'pixel_coords': pixel_coords,
      'bbox': bbox.to_tuple(),
      'color': color_name,
    })

  scene_struct['objects'] = objects
  return blender_objects, scene_struct



def swap_objects(scene_struct, blender_objects, camera):
  """ Swap two specified blender object """

  objects = scene_struct['objects']
  #positions = [obj.location for obj in blender_objects]
  
  obj2 = blender_objects[1].location.copy()
  loc2 = objects[1]['pixel_coords']

  blender_objects[1].location = blender_objects[0].location
  objects[1]['pixel_coords'] = objects[0]['pixel_coords']

  blender_objects[0].location = obj2
  objects[0]['pixel_coords'] = loc2

  scene_struct['objects'] = objects
  return blender_objects, scene_struct

class Box:

    dim_x = 1
    dim_y = 1

    def __init__(self, min_x, min_y, max_x, max_y, dim_x=dim_x, dim_y=dim_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.dim_x = dim_x
        self.dim_y = dim_y

    @property
    def x(self):
        return round(self.min_x * self.dim_x)

    @property
    def y(self):
        return round(self.dim_y - self.max_y * self.dim_y)

    @property
    def width(self):
        return round((self.max_x - self.min_x) * self.dim_x)

    @property
    def height(self):
        return round((self.max_y - self.min_y) * self.dim_y)

    def __str__(self):
        return "<Box, x=%i, y=%i, width=%i, height=%i>" % \
               (self.x, self.y, self.width, self.height)

    def to_tuple(self):
        if self.width == 0 or self.height == 0:
            return (0, 0, 0, 0)
        return (self.x, self.y, self.width, self.height)


def camera_view_bounds_2d(scene, cam_ob, me_ob):
    """
    Returns camera space bounding box of mesh object.

    Negative 'z' value means the point is behind the camera.

    Takes shift-x/y, lens angle and sensor size into account
    as well as perspective/ortho projections.

    :arg scene: Scene to use for frame size.
    :type scene: :class:`bpy.types.Scene`
    :arg obj: Camera object.
    :type obj: :class:`bpy.types.Object`
    :arg me: Untransformed Mesh.
    :type me: :class:`bpy.types.Mesh´
    :return: a Box object (call its to_tuple() method to get x, y, width and height)
    :rtype: :class:`Box`
    """

    if getattr(me_ob, "type", None) != 'MESH':
        child_meshes = utils.get_mesh_descendants(me_ob)
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
            # Does it make any sense to drop these?
            #if z <= 0.0:
            #    continue
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


def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))


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
        utils.delete_object(obj)
      return add_random_objects(scene_struct, num_objects, args, camera, radius_cache)

    # Choose random orientation for the object.
    theta = 360.0 * random.random()

    # Actually add the object to the scene (offset is applied inside utils.add_object)
    rotated_offset = rotate_offset(base_offset, theta)
    utils.add_object(
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
    utils.add_radius_indicator((x, y, obj.location.z), footprint_radius, parent=obj)

    # Record data about the object in the scene data structure
    pixel_coords = utils.get_camera_coords(camera, obj.location)
    bbox = camera_view_bounds_2d(bpy.context.scene, camera, obj) #bpy_extras.object_utils.world_to_camera_view(scene, obj, coord)

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
  all_visible = check_visibility(blender_objects, args.min_pixels_per_object)
  if not all_visible:
    # If any of the objects are fully occluded then start over; delete all
    # objects from the scene and place them all again.
    print('Some objects are occluded; replacing objects')
    for obj in blender_objects:
      utils.delete_object(obj)
    return add_random_objects(scene_struct, num_objects, args, camera, radius_cache)

  return objects, blender_objects


def compute_all_relationships(scene_struct, eps=0.2):
  """
  Computes relationships between all pairs of objects in the scene.
  
  Returns a dictionary mapping string relationship names to lists of lists of
  integers, where output[rel][i] gives a list of object indices that have the
  relationship rel with object i. For example if j is in output['left'][i] then
  object j is left of object i.
  """
  all_relationships = {}
  try:
    for name, direction_vec in scene_struct['directions'].items():
      if name == 'above' or name == 'below': continue
      all_relationships[name] = []
      for i, obj1 in enumerate(scene_struct['objects']):
        coords1 = obj1['3d_coords']
        related = set()
        for j, obj2 in enumerate(scene_struct['objects']):
          if obj1 == obj2: continue
          coords2 = obj2['3d_coords']
          diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
          dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
          if dot > eps:
            related.add(j)
        all_relationships[name].append(sorted(list(related)))
  except TypeError:
    print(scene_struct)
    raise
    
  return all_relationships


def check_visibility(blender_objects, min_pixels_per_object):
  """
  Check whether all objects in the scene have some minimum number of visible
  pixels; to accomplish this we assign random (but distinct) colors to all
  objects, and render using no lighting or shading or antialiasing; this
  ensures that each object is just a solid uniform color. We can then count
  the number of pixels of each color in the output image to check the visibility
  of each object.

  Returns True if all objects are visible and False otherwise.
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


def render_shadeless(blender_objects, path='flat.png'):
  """
  Render a version of the scene with shading disabled and unique materials
  assigned to all objects, and return a set of all colors that should be in the
  rendered image. The image itself is written to path. This is used to ensure
  that all objects will be visible in the final rendered scene.
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
    meshes = utils.get_mesh_descendants(obj)
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


if __name__ == '__main__':
  if INSIDE_BLENDER:
    # Run normally
    argv = utils.extract_args()
    args = parser.parse_args(argv)
    main(args)
  elif '--help' in sys.argv or '-h' in sys.argv:
    parser.print_help()
  else:
    print('This script is intended to be called from blender like this:')
    print()
    print('blender --background --python render_images.py -- [args]')
    print()
    print('You can also run as a standalone python script to view all')
    print('arguments like this:')
    print()
    print('python render_images.py --help')

