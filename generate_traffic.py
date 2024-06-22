#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example script to generate traffic in the simulation"""

import glob
import os
import sys
import time
import cv2

try:
    sys.path.append(glob.glob('../carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from carla import VehicleLightState as vls

import argparse
import logging
from numpy import random
from queue import Queue, Empty
import numpy as np
import CARLA_annotator.carla_vehicle_annotator as cva

######### Can change ########
IM_WIDTH=3840
IM_HEIGHT=2160
# Capture period of sensors (s) in simulation world
tick_sensor = 10

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

def mkdir_folder(path, ext):
    if not os.path.isdir(os.path.join(path, ext)):
        os.makedirs(os.path.join(path, ext))

# modify from manual control
def _parse_image_cb(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array

def retrieve_data(sensor_queue, frame, timeout=1):
    while True:
        try:
            data = sensor_queue.get(True,timeout)
        except Empty:
            return None
        if data.frame == frame:
            return data

def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=30,
        type=int,
        help='Number of vehicles (default: 30)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=0,
        type=int,
        help='Number of walkers (default: 10)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='Avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='Filter vehicle model (default: "vehicle.*")')
    argparser.add_argument(
        '--generationv',
        metavar='G',
        default='All',
        help='restrict to certain vehicle generation (values: "1","2","All" - default: "All")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='Filter pedestrian type (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--generationw',
        metavar='G',
        default='2',
        help='restrict to certain pedestrian generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='Port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--asynch',
        action='store_true',
        help='Activate asynchronous mode execution')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Activate hybrid mode for Traffic Manager')
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        type=int,
        help='Set random device seed and deterministic mode for Traffic Manager')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=False,
        help='Enable car lights')
    argparser.add_argument(
        '--hero',
        action='store_true',
        default=False,
        help='Set one of the vehicles as hero')
    argparser.add_argument(
        '--respawn',
        action='store_true',
        default=False,
        help='Automatically respawn dormant vehicles (only in large maps)')
    argparser.add_argument(
        '--no-rendering',
        action='store_true',
        default=False,
        help='Activate no rendering mode')
    argparser.add_argument(
        '--save-path',
        default='data/', 
        help='Synchronous mode execution')
    argparser.add_argument(
        '--save-patched',
        action='store_true',
        default=False,
        help='Drawwing ground truth bbox and keypoints on images.')
    argparser.add_argument(
        '--max-frame',
        default=2000,
        type=int,
        help='Maximum frames')

    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    vehicles_list = []
    walkers_list = []
    all_id = []
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    synchronous_master = False
    random.seed(args.seed if args.seed is not None else int(time.time()))

    try:
        world = client.get_world()
        
        if args.car_lights_on:
            world.set_weather(carla.WeatherParameters.ClearNight)
        else:
            world.set_weather(carla.WeatherParameters.Default)
        #world.set_weather(carla.WeatherParameters.SoftRainSunset)

        ############# Set traffic manager ############
        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        if args.respawn:
            traffic_manager.set_respawn_dormant_vehicles(True)
        if args.hybrid:
            traffic_manager.set_hybrid_physics_mode(True)
            traffic_manager.set_hybrid_physics_radius(70.0)
        if args.seed is not None:
            traffic_manager.set_random_device_seed(args.seed)

        settings = world.get_settings()
        if not args.asynch:
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
            else:
                synchronous_master = False
        else:
            print("You are currently in asynchronous mode. If this is a traffic simulation, \
            you could experience some issues. If it's not working correctly, switch to synchronous \
            mode by using traffic_manager.set_synchronous_mode(True)")

        if args.no_rendering:
            settings.no_rendering_mode = True
            
        world.apply_settings(settings)

        ############# Get blueprints and spaw points ############
        blueprints = get_actor_blueprints(world, args.filterv, args.generationv)
        blueprintsWalkers = get_actor_blueprints(world, args.filterw, args.generationw)

        if args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]
            blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
            blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        hero = args.hero
        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            if hero:
                blueprint.set_attribute('role_name', 'hero')
                hero = False
            else:
                blueprint.set_attribute('role_name', 'autopilot')

            # prepare the light state of the cars to spawn
            light_state = vls.NONE
            if args.car_lights_on:
                light_state = vls.Position | vls.LowBeam | vls.LowBeam

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
                .then(SetVehicleLightState(FutureActor, light_state)))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(args.number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        if args.asynch or not synchronous_master:
            world.wait_for_tick()
        else:
            world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

        # Example of how to use Traffic Manager parameters
        traffic_manager.global_percentage_speed_difference(30.0)

        #################### Sensor settings ##################
        sensor_list = []
        q_list = []

        tick_queue = Queue()
        world.on_tick(tick_queue.put)
        q_list.append(tick_queue)

        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x", "{}".format(IM_WIDTH))
        cam_bp.set_attribute("image_size_y", "{}".format(IM_HEIGHT))
        cam_bp.set_attribute("fov", "60")
        cam_bp.set_attribute("sensor_tick", str(tick_sensor))

        cam_bp.set_attribute("blur_amount", "0")
        cam_bp.set_attribute("motion_blur_intensity", "0")
        cam_bp.set_attribute("motion_blur_max_distortion", "0")
        cam_bp.set_attribute("motion_blur_min_object_screen_size", "0")
        

        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
        lidar_bp.set_attribute('sensor_tick', str(tick_sensor))
        lidar_bp.set_attribute('channels', '128')
        lidar_bp.set_attribute('points_per_second', '2240000')
        lidar_bp.set_attribute('upper_fov', '50')
        lidar_bp.set_attribute('lower_fov', '-15')
        lidar_bp.set_attribute('range', '150')
        lidar_bp.set_attribute('rotation_frequency', '20')

        sem_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        sem_bp.set_attribute("image_size_x","{}".format(IM_WIDTH))
        sem_bp.set_attribute("image_size_y","{}".format(IM_HEIGHT))
        sem_bp.set_attribute("fov","60")
        sem_bp.set_attribute("sensor_tick", str(tick_sensor))

        ######## sensor 1 ########
        x1,y1,z1=-46,-63,7
        yaw1,pitch1,roll1=0.0,-20.0,0.0
        transform1=carla.Transform(carla.Location(x=x1, y=y1, z=z1),carla.Rotation(yaw=yaw1,pitch=pitch1,roll=roll1))

        cam01 = world.spawn_actor(cam_bp, transform1, attach_to=None)
        cam01_queue = Queue()
        cam01.listen(cam01_queue.put)
        q_list.append(cam01_queue)
        sensor_list.append(cam01)
        print('RGB camera{} ready'.format(1))

        lidar01 = world.spawn_actor(lidar_bp, transform1, attach_to=None)
        lidar01_queue = Queue()
        lidar01.listen(lidar01_queue.put)
        q_list.append(lidar01_queue)
        sensor_list.append(lidar01)
        print('LIDAR{} ready'.format(1))

        sem01 = world.spawn_actor(sem_bp, transform1, attach_to=None)
        sem01_queue = Queue()
        sem01.listen(sem01_queue.put)
        q_list.append(sem01_queue)
        sensor_list.append(sem01)
        print('Semantic{} ready'.format(1))

        ######## sensor 2 ########
        x2,y2,z2=-92.91,19.21,7
        yaw2,pitch2,roll2=0.0,-20.0,0.0
        transform2=carla.Transform(carla.Location(x=x2, y=y2, z=z2),carla.Rotation(yaw=yaw2,pitch=pitch2,roll=roll2))

        cam02 = world.spawn_actor(cam_bp, transform2, attach_to=None)
        cam02_queue = Queue()
        cam02.listen(cam02_queue.put)
        q_list.append(cam02_queue)
        sensor_list.append(cam02)
        print('RGB camera{} ready'.format(2))

        lidar02 = world.spawn_actor(lidar_bp, transform2, attach_to=None)
        lidar02_queue = Queue()
        lidar02.listen(lidar02_queue.put)
        q_list.append(lidar02_queue)
        sensor_list.append(lidar02)
        print('LIDAR{} ready'.format(2))

        sem02 = world.spawn_actor(sem_bp, transform2, attach_to=None)
        sem02_queue = Queue()
        sem02.listen(sem02_queue.put)
        q_list.append(sem02_queue)
        sensor_list.append(sem02)
        print('Semantic{} ready'.format(2))

        ######## sensor 3 ########
        x3,y3,z3=-62,133.67,7
        yaw3,pitch3,roll3=0.0,-20.0,0.0
        transform3=carla.Transform(carla.Location(x=x3, y=y3, z=z3),carla.Rotation(yaw=yaw3,pitch=pitch3,roll=roll3))

        cam03 = world.spawn_actor(cam_bp, transform3, attach_to=None)
        cam03_queue = Queue()
        cam03.listen(cam03_queue.put)
        q_list.append(cam03_queue)
        sensor_list.append(cam03)
        print('RGB camera{} ready'.format(3))

        lidar03 = world.spawn_actor(lidar_bp, transform3, attach_to=None)
        lidar03_queue = Queue()
        lidar03.listen(lidar03_queue.put)
        q_list.append(lidar03_queue)
        sensor_list.append(lidar03)
        print('LIDAR{} ready'.format(3))

        sem03 = world.spawn_actor(sem_bp, transform3, attach_to=None)
        sem03_queue = Queue()
        sem03.listen(sem03_queue.put)
        q_list.append(sem03_queue)
        sensor_list.append(sem03)
        print('Semantic{} ready'.format(3))

        ############ Set save path #############
        # imageExt='image_2'
        # mkdir_folder(args.save_path, imageExt)
        savePath = os.path.join(args.save_path, 
                    '{}_{}_{}_{}_{}_{}_{}'.format(os.path.basename(world.get_map().name),
                                                x1,y1,z1,
                                                yaw1,pitch1,roll1))
        savePath02 = os.path.join(args.save_path, 
                    '{}_{}_{}_{}_{}_{}_{}'.format(os.path.basename(world.get_map().name),
                                                x2,y2,z2,
                                                yaw2,pitch2,roll2))
        savePath03 = os.path.join(args.save_path, 
                    '{}_{}_{}_{}_{}_{}_{}'.format(os.path.basename(world.get_map().name),
                                                x3,y3,z3,
                                                yaw3,pitch3,roll3))

        ############### main loop ##############
        time_sim = 0
        cnt = 0
        while True:
            if not args.asynch and synchronous_master:
                nowFrame = world.tick()
            else:
                nowFrame = world.wait_for_tick()

            if time_sim >= tick_sensor:
                data = [retrieve_data(q, nowFrame, timeout=1) for q in q_list]
                assert all(x.frame == nowFrame for x in data if x is not None)

                # Skip if any sensor data is not available
                if None in data:
                    continue

                vehicles_raw = world.get_actors().filter('vehicle.*')
                snap = data[0]
                rgb_img = data[1]
                lidar_img = data[2]
                sem_img = data[3]

                rgb_img02 = data[4]
                lidar_img02 = data[5]
                sem_img02 = data[6]

                rgb_img03 = data[7]
                lidar_img03 = data[8]
                sem_img03 = data[9]

                # Attach additional information to the snapshot
                vehicles = cva.snap_processing(vehicles_raw, snap)

                # Calculating visible bounding boxes
                filtered_out,_ = cva.auto_annotate_lidar(vehicles, cam01, lidar_img, show_img = None, json_path = 'CARLA_annotator/vehicle_parameters.json', max_dist=200, min_detect=2)
                filtered_out02,_ = cva.auto_annotate_lidar(vehicles, cam02, lidar_img02, show_img = None, json_path = 'CARLA_annotator/vehicle_parameters.json', max_dist=200, min_detect=2)
                filtered_out03,_ = cva.auto_annotate_lidar(vehicles, cam03, lidar_img03, show_img = None, json_path = 'CARLA_annotator/vehicle_parameters.json', max_dist=200, min_detect=2)
                # Save the results
                savePatched=args.save_patched
                cva.save_output(rgb_img, filtered_out['bbox'], filtered_out['class'], bboxes3D=filtered_out['bbox_3d'], bboxes3D_world=filtered_out['bbox_3d_world'], bboxes3D_quality=filtered_out['bbox_3d_quality'], keypoints=filtered_out['keypoints'], keypoints_world=filtered_out['keypoints_world'], camera_k_matrix=filtered_out['camera_k_matrix'], velocity=filtered_out['velocity'], acceleration=filtered_out['acceleration'], semantic_img=sem_img, save_patched=savePatched, out_format='json', path=savePath, config_json='CARLA_annotator/vehicle_parameters.json')
                cva.save_output(rgb_img02, filtered_out02['bbox'], filtered_out02['class'], bboxes3D=filtered_out02['bbox_3d'], bboxes3D_world=filtered_out02['bbox_3d_world'], bboxes3D_quality=filtered_out02['bbox_3d_quality'], keypoints=filtered_out02['keypoints'], keypoints_world=filtered_out02['keypoints_world'], camera_k_matrix=filtered_out02['camera_k_matrix'], velocity=filtered_out02['velocity'], acceleration=filtered_out02['acceleration'], semantic_img=sem_img02, save_patched=savePatched, out_format='json', path=savePath02, config_json='CARLA_annotator/vehicle_parameters.json')
                cva.save_output(rgb_img03, filtered_out03['bbox'], filtered_out03['class'], bboxes3D=filtered_out03['bbox_3d'], bboxes3D_world=filtered_out03['bbox_3d_world'], bboxes3D_quality=filtered_out03['bbox_3d_quality'], keypoints=filtered_out03['keypoints'], keypoints_world=filtered_out03['keypoints_world'], camera_k_matrix=filtered_out03['camera_k_matrix'], velocity=filtered_out03['velocity'], acceleration=filtered_out03['acceleration'],  semantic_img=sem_img03, save_patched=savePatched, out_format='json', path=savePath03, config_json='CARLA_annotator/vehicle_parameters.json')
                print('{} saved'.format(nowFrame))
                time_sim = 0

                cnt+=1
                if cnt >= args.max_frame:
                    break

            time_sim = time_sim + settings.fixed_delta_seconds

    finally:

        if not args.asynch and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        print('\ndestroying %d sensors' % len(sensor_list))
        for x in sensor_list:
            x.destroy()

        time.sleep(3)

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
