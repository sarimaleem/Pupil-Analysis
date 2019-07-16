import os
from pathlib import Path
import time
from joblib import Parallel, delayed

from pupil_preprocess import extract_pupil
from pupil_preprocess import preprocess_pupil
from pupil_preprocess import utils

directory = '/media/external2/raw2/vns_exploration/data/'

mp4_files = [str(f) for f in Path(directory).glob('**/*.mp4')]

for mp4_file in mp4_files:

    # output directory:
    output_dir = os.path.join(os.path.dirname(mp4_file), 'pupil_preprocess')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # subject_id:
    subject_id = os.path.basename(mp4_file).split('.')[0]

    # deeplabcut output:
    hdf_file, = [str(h5) for h5 in Path(os.path.dirname(mp4_file)).glob('**/*.h5')]
    crop_file, = [str(pickle) for pickle in Path(os.path.dirname(mp4_file)).glob('**/*.pickle') if 'DeepCut' not in str(pickle)]
    
    # extract:
    df = extract_pupil.extract_timeseries(hdf_file=hdf_file, mp4_file=mp4_file, crop_file=crop_file, output_dir=output_dir, subject_id=subject_id)

    # preprocess:
    fps = utils.get_frame_rate(mp4_file)
    params = {'fs': fps, 'hp': 0.01, 'lp': 6, 'order': 3, 's_hp': 10}
    df = preprocess_pupil.preprocess_pupil(df=df, params=params, output_dir=output_dir, subject_id=subject_id)