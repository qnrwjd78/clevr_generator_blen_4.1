#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys


def _ensure_repo_on_path():
  repo_root = os.path.dirname(os.path.abspath(__file__))
  if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


_ensure_repo_on_path()

try:
  import bpy  # type: ignore
except ImportError:
  print("This script must be executed from Blender via `blender --background --python precompute_radii.py -- [args]`")
  sys.exit(1)

import utils  # pylint: disable=wrong-import-position


def parse_args():
  parser = argparse.ArgumentParser(description="Precompute per-shape footprint radii and preview renders.")
  parser.add_argument('--base_scene_blendfile', default='data/base_scene.blend')
  parser.add_argument('--properties_json', default='data/properties.json')
  parser.add_argument('--shape_dir', default='data/shapes')
  parser.add_argument('--output_radius_json', default='data/shape_radii.json')
  parser.add_argument('--preview_dir', default='output_precomputeradius_previews')
  parser.add_argument('--render_width', type=int, default=640)
  parser.add_argument('--render_height', type=int, default=480)
  parser.add_argument('--use_gpu', type=int, default=0)
  parser.add_argument('--samples', type=int, default=64)
  return utils.parse_args(parser)


def configure_scene(scene, args):
  scene.render.engine = 'CYCLES'
  scene.render.resolution_x = args.render_width
  scene.render.resolution_y = args.render_height
  scene.render.resolution_percentage = 100
  scene.cycles.samples = args.samples
  if hasattr(scene.cycles, 'use_adaptive_sampling'):
    scene.cycles.use_adaptive_sampling = True
  if args.use_gpu:
    scene.cycles.device = 'GPU'
    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons['cycles'].preferences
    if hasattr(cycles_prefs, 'compute_device_type'):
      cycles_prefs.compute_device_type = 'CUDA'
    if hasattr(cycles_prefs, 'get_devices'):
      cycles_prefs.get_devices()
      for device in cycles_prefs.devices:
        device.use = True
  else:
    scene.cycles.device = 'CPU'


def resolve_shape_scale(shape_key, shape_scales, size_dict):
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
  if size_dict:
    # Use the first defined size as a fallback if nothing else is specified.
    first_key = next(iter(size_dict))
    return size_dict[first_key]
  return 1.0


def _compute_xy_center(obj, depsgraph=None):
  meshes = utils.get_mesh_descendants(obj)
  if not meshes:
    return obj.location.x, obj.location.y
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
    return obj.location.x, obj.location.y

  return 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)


def precompute_radius_for_shape(shape_key, blend_name, scale, args, preview_dir):
  radius = utils.add_object(args.shape_dir, blend_name, scale, (0.0, 0.0), theta=0.0)
  obj = bpy.context.active_object
  if obj is None:
    raise RuntimeError("Failed to append object for shape '%s'" % shape_key)

  center_x, center_y = _compute_xy_center(obj)
  offset_x = center_x - obj.location.x
  offset_y = center_y - obj.location.y
  computed_radius = obj.get("clevr_scale", radius)
  ring_location = (obj.location.x + offset_x, obj.location.y + offset_y, obj.location.z)
  utils.add_radius_indicator(ring_location, computed_radius, parent=obj)

  scene = bpy.context.scene
  scene.render.filepath = os.path.join(preview_dir, f"{shape_key}.png")
  bpy.ops.render.render(write_still=True)

  utils.delete_object(obj)
  return float(computed_radius), [float(offset_x), float(offset_y)]


def main():
  args = parse_args()
  if args.base_scene_blendfile and os.path.isfile(args.base_scene_blendfile):
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

  scene = bpy.context.scene
  configure_scene(scene, args)

  os.makedirs(args.preview_dir, exist_ok=True)
  os.makedirs(os.path.dirname(args.output_radius_json), exist_ok=True)

  with open(args.properties_json, 'r') as f:
    properties = json.load(f)

  size_dict = properties.get('sizes', {})
  shape_scales = properties.get('shape_scales', {})
  shapes = properties.get('shapes', {})

  radius_data = {}
  for logical_name, blend_name in shapes.items():
    print(f"Processing shape '{logical_name}' from '{blend_name}'")
    scale = resolve_shape_scale(logical_name, shape_scales, size_dict)
    radius, offset = precompute_radius_for_shape(logical_name, blend_name, scale, args, args.preview_dir)
    radius_data[logical_name] = {
        "radius": radius,
        "center_offset": offset,
    }
    print(f"  -> radius: {radius:.4f}, offset: {offset}")

  with open(args.output_radius_json, 'w') as f:
    json.dump(radius_data, f, indent=2, sort_keys=True)
  print(f"Saved radii for {len(radius_data)} shapes to {args.output_radius_json}")


if __name__ == '__main__':
  main()
