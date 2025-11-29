# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import math
import random

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

def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))

def rotate_offset(offset, theta_degrees):
  if offset is None:
    return (0.0, 0.0)
  ox, oy = offset
  theta = math.radians(theta_degrees)
  # print("rotate_offset:", theta_degrees, theta)
  cos_t = math.cos(theta)
  sin_t = math.sin(theta)
  return (
      ox * cos_t - oy * sin_t,
      ox * sin_t + oy * cos_t,
  )

def check_valid_placement(x, y, radius, positions, scene_struct, args):
  directions = scene_struct['directions']
  for (xx, yy, rr) in positions:
    dx = x - xx
    dy = y - yy
    dist = math.sqrt(dx * dx + dy * dy)
    if dist - radius - rr < args.min_dist:
      return False
    for direction_name in ['left', 'right', 'front', 'behind']:
      direction_vec = directions[direction_name]
      assert direction_vec[2] == 0
      margin = dx * direction_vec[0] + dy * direction_vec[1]
      if 0 < margin < args.margin:
        return False
  return True

def find_valid_xy(radius, positions, scene_struct, args,
                  min_distance_from=None, min_distance=0.0, max_attempts=None):
  """
  Sample a valid (x, y) location that keeps the new object far enough away from
  existing placements and optionally enforces a minimum displacement from a
  reference point (used when relocating an object).
  """
  attempts = 0
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

    if check_valid_placement(x, y, radius, positions, scene_struct, args):
      return x, y
