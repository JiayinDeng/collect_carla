import os
import numpy as np
import argparse
import json
import cv2

def clean_keypoints(kps):
    n = len(kps)
    ret = np.hstack((np.arange(n).reshape([-1, 1]), kps))
    idx_det = np.where(ret[:, 3] != 0)[0]
    return ret[idx_det]
    

if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add the input folder path argument
    parser.add_argument('root_path', type=str, help='Path of the root of the data')
    parser.add_argument('--video', action='store_true', default=False, help='Combine the rgbs.')
    parser.add_argument('--radar', action='store_true', default=False, help='Combine the radar data')
    parser.add_argument('--label', action='store_true', default=False, help='Combine the labels')
    parser.add_argument('--keypoint', action='store_true', default=False, help='Combine the keypoints_json')

    # Parse the arguments
    args = parser.parse_args()

    # Access the root path
    root_path = args.root_path
    
    ############### combine labels and radar points ###############
    label = []
    radar_point = []

    if args.label:
        if not os.path.exists(os.path.join(root_path, 'out_label')):
            print('out_label folder not exists!')
            exit(0)
        out_label_path = os.path.join(root_path, 'out_label')
        # Read label file
        for file in sorted(os.listdir(out_label_path)):
            file_path = os.path.join(out_label_path, file)
            file_name_no_ext = os.path.splitext(file)[0]
            with open(file_path, 'r') as f:
                data = json.load(f)
            for key, val in data.items():
                data[key] = np.array(val)
            label.append(data)
        np.save(os.path.join(root_path, 'labels.npy'), label)
        print('Save labels to {}'.format(os.path.join(root_path, 'labels.npy')))

    if args.radar:
        if not os.path.exists(os.path.join(root_path, 'radar')):
            print('radar folder not exists!')
            exit(0)
        radar_path = os.path.join(root_path, 'radar')
        # Read radar file
        for file in sorted(os.listdir(radar_path)):
            file_path = os.path.join(radar_path, file)
            data = np.load(file_path, allow_pickle=True)
            radar_point.append(data)
        np.save(os.path.join(root_path, 'radar_point.npy'), np.asarray(radar_point, dtype=object))
        print('Save radar data to {}'.format(os.path.join(root_path, 'radar_point.npy')))

    ############### combine keypoints_json ###############
    keypoints = []
    if args.keypoint:
        if not os.path.exists(os.path.join(root_path, 'keypoints_json')):
            print('keypoints_json folder not exists!')
            exit(0)
        keypoints_json_path = os.path.join(root_path, 'keypoints_json')
        # Read keypoints json file
        for file in sorted(os.listdir(keypoints_json_path)):
            file_path = os.path.join(keypoints_json_path, file)
            file_name_no_ext = os.path.splitext(file)[0]
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            keypoints_pixel = np.array([clean_keypoints(np.asarray(item['keypoints']).reshape([-1, 3])) for item in data], dtype=object)
            keypoints_bbox = np.array([np.asarray(item['bbox']).reshape([-1, 2]) for item in data])
            keypoints_score = np.array([item['score'] for item in data])
            keypoints_category_id = np.array([item['category_id'] for item in data])

            now = dict()
            now['keypoints'] = keypoints_pixel
            now['bbox'] = keypoints_bbox
            now['score'] = keypoints_score
            now['category_id'] = keypoints_category_id
            keypoints.append(now)

        np.save(os.path.join(root_path, 'keypoints_det.npy'), np.asarray(keypoints, dtype=object))
        print('Save keypoints_det to {}'.format(os.path.join(root_path, 'keypoints_det.npy')))

    ############### combine images ###############
    # Get the list of image files and sort them in lexicographical order
    if args.video:
        if not os.path.exists(os.path.join(root_path, 'rgb')):
            print('rgb folder not exists!')
            exit(0)
        rgb_path = os.path.join(root_path, 'rgb')

        images = sorted([img for img in os.listdir(rgb_path) if img.endswith('.jpg')])

        # Read the first image to get the width and height information
        frame = cv2.imread(os.path.join(rgb_path, images[0]))
        height, width, _ = frame.shape

        # Create a video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can choose the video codec based on your requirements
        video = cv2.VideoWriter(os.path.join(root_path, 'output.mp4'), fourcc, 20, (width, height))

        # Write each image frame to the video
        for image in images:
            print('Saving image:', image)
            frame = cv2.imread(os.path.join(rgb_path, image))
            video.write(frame)

        # Release resources
        video.release()
        cv2.destroyAllWindows()
        print('Save video to {}'.format(os.path.join(root_path, 'output.mp4')))