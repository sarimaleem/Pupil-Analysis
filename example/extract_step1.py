import os
from pathlib import Path
import glob
import numpy as np
import pickle
import yaml
import deeplabcut
from deeplabcut.generate_training_dataset import label_frames

from pypupil import pupil_extraction as pe

from IPython import embed as shell

#concatenate all *png frames to video
#ffmpeg -framerate 15 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p version1.mp4

# get all frames:
# training_set_directory = '/home/degee/DeepLabCut/dlc_training_set/'
# frames = glob.glob(os.path.join(training_set_directory, '*.png'))

# # create project:
# deeplabcut.create_new_project('version1', 'jack', '/home/degee/DeepLabCut/version1/version1.mp4')

# # in case we want to add more videos to the training data:
# deeplabcut.add_new_videos()

# path_config_file = '/home/degee/DeepLabCut/version1/version1-jack-2019-04-03/config.yaml'

# # extract frames:
# deeplabcut.extract_frames(path_config_file, 'automatic', 'uniform')

# # create train data set:
# label_frames(path_config_file)
# deeplabcut.create_training_dataset(path_config_file, Shuffles=[1])

# # train network:
# deeplabcut.train_network(path_config_file, saveiters=1000, displayiters=10, shuffle=1)

# # evaluate network:
# deeplabcut.evaluate_network(path_config_file)

# get all filenames:
network_dir = '/media/external1/deeplabcut/version1-jack-2019-04-03/'
directory = '/home/degee/DeepLabCut/VNS/single/'

# all filenames:
filenames = [str(f) for f in Path(directory).glob('**/*.mp4')]

# get crop coordinates:
for filename in filenames:
    print(filename)
    if not os.path.exists(filename.replace('.mp4', '.pickle')):
        crop_cords, video_dimension = pe.select_crop_box(os.path.join(directory, filename), nr_frames_mean=10, n_jobs=4)
        video_info = {'crop_cords': crop_cords, 'video_dimension': video_dimension}
        with open(filename.replace('.mp4', '.pickle'), 'wb') as handle:
            pickle.dump(video_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

# analyzing videos:
for filename in filenames:
    
    # change yaml:
    with open(filename.replace('.mp4', '.pickle'), 'rb') as handle:
        video_info = pickle.load(handle)
    path_config_file = '{}/config (copy).yaml'.format(network_dir)
    with open(path_config_file) as f:
        list_doc = yaml.load(f)
    
    list_doc['video_sets']['/home/degee/DeepLabCut/version1/version1.mp4']['crop'] = '0, {}, 0, {}'.format(
                                                                                        video_info['video_dimension'][0], 
                                                                                        video_info['video_dimension'][1])
    list_doc['x1'] = int(video_info['crop_cords'][1][0])
    list_doc['x2'] = int(video_info['crop_cords'][1][1])
    list_doc['y1'] = int(video_info['crop_cords'][0][0])
    list_doc['y2'] = int(video_info['crop_cords'][0][1])
    with open(path_config_file, "w") as f:
        yaml.dump(list_doc, f)

    # extract!
    deeplabcut.analyze_videos(path_config_file, [filename], shuffle=1, videotype='.mp4')

# # create labeled video
# for filename in filenames:
#     deeplabcut.create_labeled_video(path_config_file, [filename], save_frames=True)

# # plot the trajectories of the analyzed videos 
# %matplotlib notebook  #for making interactive plots.
# deeplabcut.plot_trajectories(path_config_file,videofile_path,showfigures=True)

# videofile_path = [] #Enter the list of videos to analyze.