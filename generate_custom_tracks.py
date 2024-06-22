import sys
import glob
import os
try:
    sys.path.append(glob.glob('../carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import time
from queue import Queue, Empty

import CARLA_annotator.carla_vehicle_annotator as cva

IM_WIDTH=3840
IM_HEIGHT=2160

def main():
    # Connect to the client and retrieve the world object
    client = carla.Client('localhost', 2000)
    world = client.get_world()

    # Set up the simulator in synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True # Enables synchronous mode
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Set up the TM in synchronous mode
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    # Set a seed so behaviour can be repeated if necessary
    traffic_manager.set_random_device_seed(0)
    random.seed(0)

    # We will aslo set up the spectator so we can see what we do
    spectator = world.get_spectator()

    # Set delay to create gap between spawn times
    spawn_delay = 20
    counter = spawn_delay

    # Set max vehicles (set smaller for low hardward spec)
    max_vehicles = 10
    # Alternate between spawn points
    alt = False

    # Select some models from the blueprint library
    # models = ['dodge', 'audi', 'model3', 'mini', 'mustang', 'lincoln', 'prius', 'nissan', 'crown', 'impala']
    # blueprints = []
    # for vehicle in world.get_blueprint_library().filter('*vehicle*'):
    #     if any(model in vehicle.id for model in models):
    #         blueprints.append(vehicle)

    blueprints = world.get_blueprint_library().filter('vehicle.audi.tt')

    spawn_points = world.get_map().get_spawn_points()
    vehicle_list = []
    sensor_list = []
    q_list = []

    try:
        ##################### vehicle settings ####################
        # Route
        spawn_point =  spawn_points[10]
        # Create route from the chosen spawn points
        route_indices = [10, 144, 147, 74, 145]
        route = []
        for ind in route_indices:
            route.append(spawn_points[ind].location)

        vehicle_bp = random.choice(blueprints)
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        vehicle_list.append(vehicle)

        vehicle.set_autopilot(True) # Give TM control over vehicle

        # Set parameters of TM vehicle control, we don't want lane changes
        traffic_manager.update_vehicle_lights(vehicle, True)
        traffic_manager.random_left_lanechange_percentage(vehicle, 50)
        traffic_manager.random_right_lanechange_percentage(vehicle, 50)
        traffic_manager.ignore_lights_percentage(vehicle, 100)
        traffic_manager.auto_lane_change(vehicle, True)

        # # Alternate between routes
        traffic_manager.set_path(vehicle, route)

        ##################### sensor settings ####################
        tick_queue = Queue()
        world.on_tick(tick_queue.put)
        q_list.append(tick_queue)

        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute("image_size_x", "{}".format(IM_WIDTH))
        cam_bp.set_attribute("image_size_y", "{}".format(IM_HEIGHT))
        cam_bp.set_attribute("fov", "60")
        # cam_bp.set_attribute("sensor_tick", str(tick_sensor))

        cam_bp.set_attribute("blur_amount", "0")
        cam_bp.set_attribute("motion_blur_intensity", "0")
        cam_bp.set_attribute("motion_blur_max_distortion", "0")
        cam_bp.set_attribute("motion_blur_min_object_screen_size", "0")

        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
        # lidar_bp.set_attribute('sensor_tick', str(tick_sensor))
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
        # sem_bp.set_attribute("sensor_tick", str(tick_sensor))

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

        ############ Set save path #############
        savePath = os.path.join('data', 
                    '{}_{}_{}_{}_{}_{}_{}'.format(os.path.basename(world.get_map().name),
                                                x1,y1,z1,
                                                yaw1,pitch1,roll1))

        ############ Main Loop ############
        while True:
            nowFrame = world.tick()
            data = [cva.retrieve_data(q, nowFrame, timeout=1) for q in q_list]
            assert all(x.frame == nowFrame for x in data if x is not None)

            # Skip if any sensor data is not available
            if None in data:
                continue

            vehicles_raw = world.get_actors().filter('vehicle.*')
            snap = data[0]
            rgb_img = data[1]
            lidar_img = data[2]
            sem_img = data[3]

            # Attach additional information to the snapshot
            vehicles = cva.snap_processing(vehicles_raw, snap)

            # Calculating visible bounding boxes
            filtered_out,_ = cva.auto_annotate_lidar(vehicles, cam01, lidar_img, show_img = None, json_path = 'CARLA_annotator/vehicle_parameters.json', max_dist=200, min_detect=2)
            # Save the results
            savePatched=False
            cva.save_output(rgb_img, filtered_out['bbox'], filtered_out['class'], bboxes3D=filtered_out['bbox_3d'], bboxes3D_world=filtered_out['bbox_3d_world'], bboxes3D_quality=filtered_out['bbox_3d_quality'], keypoints=filtered_out['keypoints'], keypoints_world=filtered_out['keypoints_world'], keypoints_world_all=filtered_out['keypoints_world_all'], camera_k_matrix=filtered_out['camera_k_matrix'], velocity=filtered_out['velocity'], acceleration=filtered_out['acceleration'], vehicle_ids=filtered_out['vehicle_ids'], vehicle_pos=filtered_out['vehicle_pos'], vehicle_quats=filtered_out['vehicle_quats'], semantic_img=None, save_patched=savePatched, out_format='json', path=savePath, config_json='CARLA_annotator/vehicle_parameters.json')
            print('{} saved'.format(nowFrame))

            if vehicle.get_location().distance(route[-1]) < 2.0:
                break

    finally:
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.no_rendering_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

        client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])
        print('\ndestroying %d vehicles' % len(vehicle_list))
        
        for x in sensor_list:
            x.destroy()
        print('\ndestroying %d sensors' % len(sensor_list))
        
        time.sleep(3)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
