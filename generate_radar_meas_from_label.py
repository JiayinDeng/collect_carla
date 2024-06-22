import os
import numpy as np
import argparse
import json

if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add the input folder path argument
    parser.add_argument('root_path', type=str, help='Path of the root of the data')
    parser.add_argument('--noise', type=float, help='Variance of noise.', default=0.5)
    parser.add_argument('--density', type=float, help='Poisson density of the number of points', default=10)

    # Parse the arguments
    args = parser.parse_args()

    # Access the root path
    root_path = args.root_path
    noise = args.noise
    density = args.density

    if not os.path.exists(os.path.join(root_path, 'out_label')):
        print('out_label folder not exists!')
        exit(0)
    out_label_path = os.path.join(root_path, 'out_label')
    
    if not os.path.exists(os.path.join(root_path, 'radar')):
        os.makedirs(os.path.join(root_path, 'radar'))
    radar_path = os.path.join(root_path, 'radar')

    for file in os.listdir(out_label_path):
        file_path = os.path.join(out_label_path, file)
        file_name_no_ext = os.path.splitext(file)[0]
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        keypoints = data['keypoints_world_all']
        measurements = []
        for kp in keypoints:
            kp_pos = np.asarray(kp)[:, 1:]

            disPow = np.sum(np.square(kp_pos), axis=1)
            probRel = 1 / disPow
            probRel /= np.sum(probRel)

            num = np.random.poisson(density)
            num = max(1, num)

            measurementID = np.random.choice(range(len(probRel)), size=num, replace=True, p=probRel)
            measurements.extend([list(np.random.multivariate_normal(kp_pos[idx], np.eye(3) * noise)) for idx in measurementID])

        np.save(os.path.join(radar_path, file_name_no_ext + '.npy'), measurements)
