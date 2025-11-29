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
import math, sys, random, argparse, json, os, tempfile, time
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
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import clevr_blender_utils as utils
    import clevr_math_utils as math_utils
  except ImportError as e:
    print("\nERROR")
    print("Running render_images.py from Blender and cannot import utils.") 
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
parser.add_argument('--mode', default='rel_change', choices=['rel_change','removal', 'swap', 'addition', 'replacement', 'random'],
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
parser.add_argument('--check_visibility', default=1, type=int,
    help="Setting --check_visibility 1 enables visibility checks for objects. " +
         "If disabled (0), objects may be fully occluded but generation will be faster.")


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

  radius_cache = utils.load_shape_radii(args.shape_radii_json)

  # --- OPTIMIZATION START ---
  # Load the main blendfile ONCE
  print("Loading base scene...", flush=True)
  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

  # Configure GPU ONCE
  utils.configure_cycles_device(args.use_gpu == 1)

  # Capture base objects and their initial states
  base_objects = set([obj.name for obj in bpy.data.objects])
  initial_locations = {}
  for name in ['Camera', 'Lamp_Key', 'Lamp_Back', 'Lamp_Fill']:
      if name in bpy.data.objects:
          initial_locations[name] = bpy.data.objects[name].location.copy()
  # --- OPTIMIZATION END ---
  
  all_scene_paths = []
  for i in range(args.num_images):
    if i > 0:
      utils.reset_scene(base_objects, initial_locations)

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
    
    if i > 0:
      utils.reset_scene(base_objects, initial_locations)

    while True:
        success = render_scene(args,
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
        if success:
            break
        else:
            # If failed, reset scene and try again with new random objects
            utils.reset_scene(base_objects, initial_locations)
            num_objects = random.randint(args.min_objects, args.max_objects) # Re-roll num objects too

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

  scene = bpy.context.scene

  # Set render arguments so we can get pixel coordinates later.
  # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
  # cannot be used.
  render_args = scene.render
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
  objects, blender_objects = utils.add_random_objects(scene_struct, num_objects, args, camera, radius_cache)

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

  #Add the transformations to the objects

  #'rel_change','removal', 'swap', 'addition', 'replacement'
  success = False
  
  mode_to_use = args.mode
  if mode_to_use == 'random':
      mode_to_use = random.choice(['rel_change', 'removal', 'swap', 'addition', 'replacement'])
      print(f"Random mode selected: {mode_to_use}")

  #replacement
  if mode_to_use == "replacement":
    blender_objects, scene_struct, success = replace_random_object(scene_struct, blender_objects, camera, args, radius_cache)
  #relationship change
  elif mode_to_use == "rel_change":
    blender_objects, scene_struct, success = relative_transform_objects(scene_struct, blender_objects, camera, args)
  elif mode_to_use == "removal":
    blender_objects, scene_struct, success = remove_random_object(scene_struct, blender_objects, camera)
  elif mode_to_use == "swap":
    blender_objects, scene_struct, success = swap_objects(scene_struct, blender_objects, camera, args)
  elif mode_to_use == "addition":
    blender_objects, scene_struct, success = add_objects(scene_struct, blender_objects, camera, args, radius_cache)
  else:
    print("not ready yet")
    success = True # Assume success for unknown modes to avoid loop
  
  if not success:
      print("Mode application failed. Regenerating scene...")
      # Delete all objects to start fresh
      for obj in blender_objects:
          utils.delete_object(obj)
      return False # Signal failure to main loop

  # Log if random mode
  if args.mode == 'random':
      try:
          # output_scene_dir is like ".../scenes/"
          # dirname gives ".../scenes", dirname again gives "..."
          log_dir = os.path.dirname(args.output_scene_dir.rstrip(os.sep))
          log_path = os.path.join(log_dir, 'result_log.txt')
          with open(log_path, 'a') as f:
              f.write(f"Image {output_index}: {mode_to_use}\n")
      except Exception as e:
          print(f"Logging failed: {e}")

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
  
  return True # Signal success


def replace_random_object(scene_struct, blender_objects, camera, args, radius_cache):

  num_objs = len(blender_objects)
  # Create a list of indices and shuffle them to try random objects
  indices = list(range(num_objs))
  random.shuffle(indices)
  
  # Load the property file
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['colors'].items():
      rgba = [float(c) / 255.0 for c in rgb] + [1.0]
      color_name_to_rgba[name] = rgba
    object_mapping = [(v, k) for k, v in properties['shapes'].items()]
    sizes = properties.get('sizes', {})
    size_mapping = list(sizes.items())
    shape_scales = properties.get('shape_scales', {})

  shape_color_combos = None
  if args.shape_color_combos_json is not None:
    with open(args.shape_color_combos_json, 'r') as f:
      shape_color_combos = list(json.load(f).items())

  for n_rand in indices:
      obj = blender_objects[n_rand]
      loc = obj.location
      x, y = loc[0], loc[1]
      old_shape = scene_struct['objects'][n_rand]['shape']
      old_color = scene_struct['objects'][n_rand]['color']
      
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

      r = utils.resolve_object_scale(obj_name_out, shape_scales, sizes, size_mapping)
      footprint_radius, base_offset = utils.get_shape_radius_info(obj_name_out, radius_cache)

      # Choose random orientation for the object.
      theta = 360.0 * random.random()

      # Actually add the object to the scene
      rotated_offset = math_utils.rotate_offset(base_offset, theta)
      
      positions = utils.get_object_positions(blender_objects)
      # Exclude the object being replaced from collision check
      positions_excluding_self = positions[:n_rand] + positions[n_rand+1:]

      # Check if the new object fits in the old location
      if math_utils.check_valid_placement(x, y, footprint_radius, positions_excluding_self, scene_struct, args):
          # Valid placement found!
          utils.delete_object(obj)
          scene_struct['objects'].pop(n_rand)
          blender_objects.pop(n_rand)
          objects = scene_struct['objects']
          
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
          # positions.append((x, y, footprint_radius)) # Not needed as we return
          utils.add_radius_indicator((x, y, obj.location.z), footprint_radius, parent=obj)

          # Record data about the object in the scene data structure
          pixel_coords = utils.get_camera_coords(camera, obj.location)
          bbox = utils.camera_view_bounds_2d(bpy.context.scene, camera, obj) 

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
          scene_struct['objects'] = objects
          return blender_objects, scene_struct, True
      
      # If not valid, loop continues to next object

  # If we reach here, no object could be replaced
  return blender_objects, scene_struct, False


def relative_transform_objects(scene_struct, blender_objects, camera, args):

  num_objs = len(blender_objects)
  indices = list(range(num_objs))
  random.shuffle(indices)

  positions = utils.get_object_positions(blender_objects)
  
  for n_rand in indices:
      obj = blender_objects[n_rand]
      objects = scene_struct['objects']
      loc = obj.location
      r = obj.get("footprint_radius", obj.get("clevr_scale", loc[2]))

      # Remove the current object from positions to avoid self-collision check
      # positions is a list of (x, y, r) tuples
      positions_excluding_self = positions[:n_rand] + positions[n_rand+1:]
      
      try:
        x, y = math_utils.find_valid_xy(
            r,
            positions_excluding_self,
            scene_struct,
            args,
            min_distance_from=(loc[0], loc[1]),
            min_distance=0.5, # Reduced from 4.0 to allow movement within the scene
            max_attempts=10
        )
        
        # Success!
        obj.location = Vector((x,y,loc[2]))
        blender_objects[n_rand] = obj
        # Render the scene
        bpy.ops.render.render(write_still=False)

        pixel_coords = utils.get_camera_coords(camera, obj.location)
        bbox = utils.camera_view_bounds_2d(bpy.context.scene, camera, obj)
        objects[n_rand]['3d_coords'] = tuple(obj.location)
        objects[n_rand]['pixel_coords'] = pixel_coords
        objects[n_rand]['bbox'] = bbox.to_tuple()
        scene_struct['objects'] = objects
        return blender_objects, scene_struct, True

      except RuntimeError:
        continue # Try next object
  
  return blender_objects, scene_struct, False

def remove_random_object(scene_struct, blender_objects, camera):

  num_objs = len(blender_objects)
  if num_objs == 0:
      return blender_objects, scene_struct, False

  n_rand = int(random.uniform(0,1) * num_objs)
  
  print("object chosen: ", str(n_rand), " of ", str(num_objs))
  utils.delete_object(blender_objects[n_rand])
  
  scene_struct['objects'].pop(n_rand)
  blender_objects.pop(n_rand)

  return blender_objects, scene_struct, True

def add_objects(scene_struct, blender_objects, camera, args, radius_cache):

  num_objects = 1
  # Load the property file
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['colors'].items():
      rgba = [float(c) / 255.0 for c in rgb] + [1.0]
      color_name_to_rgba[name] = rgba
    object_mapping = [(v, k) for k, v in properties['shapes'].items()]
    sizes = properties.get('sizes', {})
    size_mapping = list(sizes.items())
    shape_scales = properties.get('shape_scales', {})

  shape_color_combos = None
  if args.shape_color_combos_json is not None:
    with open(args.shape_color_combos_json, 'r') as f:
      shape_color_combos = list(json.load(f).items())

  objects = scene_struct['objects']
  positions = utils.get_object_positions(blender_objects)
 
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

    r = utils.resolve_object_scale(obj_name_out, shape_scales, sizes, size_mapping)
    footprint_radius, base_offset = utils.get_shape_radius_info(obj_name_out, radius_cache)

    # Try to place the object, ensuring that we don't intersect any existing
    # objects and that we are more than the desired margin away from all existing
    # objects along all cardinal directions.
    valid_pos_found = False
    while True:
      try:
        x, y = math_utils.find_valid_xy(
            footprint_radius,
            positions,
            scene_struct,
            args,
            max_attempts=args.max_retries
        )
        valid_pos_found = True
        break
      except RuntimeError:
        print("*" * 20, " Num tries reached!!")
        break
    
    if not valid_pos_found:
        print("Skipping object addition due to lack of space.")
        return blender_objects, scene_struct, False

    # For cube, adjust the size a bit
    if obj_name == 'Cube':
      r /= math.sqrt(2)

    # Choose random orientation for the object.
    theta = 360.0 * random.random()

    rotated_offset = math_utils.rotate_offset(base_offset, theta)
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

    # Record data about the object in the scene data structure
    pixel_coords = utils.get_camera_coords(camera, obj.location)
    bbox = utils.camera_view_bounds_2d(bpy.context.scene, camera, obj) 
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

  scene_struct['objects'] = objects
  return blender_objects, scene_struct, True



def swap_objects(scene_struct, blender_objects, camera, args):
  """ Swap two random blender objects """

  if len(blender_objects) < 2:
      print("Not enough objects to swap.")
      return blender_objects, scene_struct, False

  objects = scene_struct['objects']
  positions = utils.get_object_positions(blender_objects)
  
  # Try to find a valid swap pair
  max_swap_attempts = 10
  for _ in range(max_swap_attempts):
      idx1, idx2 = random.sample(range(len(blender_objects)), 2)
      
      obj1 = blender_objects[idx1]
      obj2 = blender_objects[idx2]
      
      r1 = obj1.get("footprint_radius", 0.5)
      r2 = obj2.get("footprint_radius", 0.5)
      
      loc1 = obj1.location
      loc2 = obj2.location
      
      # Check if obj1 fits at loc2 AND obj2 fits at loc1
      # We need to check against all OTHER objects
      other_positions = [p for i, p in enumerate(positions) if i != idx1 and i != idx2]
      
      # Temporarily add obj2 (at loc1) to check obj1 (at loc2) validity?
      # No, we check if obj1 fits at loc2 given others + obj2 at loc1.
      # Actually, simpler: check if obj1 fits at loc2 against others, and obj2 fits at loc1 against others.
      # Then check if they collide with EACH OTHER at new positions.
      
      # Check obj1 at loc2
      valid1 = math_utils.check_valid_placement(loc2.x, loc2.y, r1, other_positions, scene_struct, args)
      # Check obj2 at loc1
      valid2 = math_utils.check_valid_placement(loc1.x, loc1.y, r2, other_positions, scene_struct, args)
      
      # Check mutual distance
      dist = math.sqrt((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2)
      valid_mutual = (dist - r1 - r2) > args.min_dist
      
      if valid1 and valid2 and valid_mutual:
          # Perform swap
          temp_loc = obj2.location.copy()
          temp_pixel = objects[idx2]['pixel_coords']
          
          blender_objects[idx2].location = blender_objects[idx1].location
          objects[idx2]['pixel_coords'] = objects[idx1]['pixel_coords']
          
          blender_objects[idx1].location = temp_loc
          objects[idx1]['pixel_coords'] = temp_pixel
          
          print(f"Swapped objects {idx1} and {idx2}")
          scene_struct['objects'] = objects
          return blender_objects, scene_struct, True
  
  print("Could not find a valid pair to swap after multiple attempts.")
  return blender_objects, scene_struct, False


def compute_all_relationships(scene_struct, eps=0.2):
  """
  Computes relationships between all pairs of objects in the scene.
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

