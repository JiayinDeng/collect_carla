import glob
import os
import sys
import time
import cv2
import random

try:
    sys.path.append(glob.glob('../carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla, re

client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)

world = client.get_world()

vehicle_blueprints = world.get_blueprint_library().filter('vehicle')

location = random.choice(world.get_map().get_spawn_points()).location

for bp in vehicle_blueprints:
    transform = carla.Transform(location, carla.Rotation(yaw=-45.0))
    vehicle = world.spawn_actor(bp, transform)
    print("\"{}\",".format(vehicle.type_id))
    vehicle.destroy()
