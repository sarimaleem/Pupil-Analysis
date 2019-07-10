import pandas as pd
from UtilFuncs import get_crop_adjust
from ProcessPupil import create_time_series
from ProcessPupil import get_time_elapsed
from pathlib import Path
from UtilFuncs import get_likelihoods, get_animal_name, get_frame_rate
from PlotFuncs import plot_verify_grid, plot_axes_ratio_dist, plot_blinks, plot_rotated_rect_area
from PreprocessPupil import preprocess_pupil, plot_preprocessing
import time

times = {'loading files': 0, 'algorithm': 0, 'plot blinks': 0, 'plot verify grid': 0}


def create_files(path):
    global times

    start = time.time()
    dc, = [str(h5) for h5 in Path(path).glob('**/*.h5')]
    vid_file, = [str(mp4) for mp4 in Path(path).glob('**/*.mp4')]
    crop, = [str(pickle) for pickle in Path(path).glob('**/*.pickle') if 'DeepCut' not in str(pickle)]

    deep_cut = pd.read_hdf(dc)
    deep_cut.columns = deep_cut.columns.droplevel()

    if crop is None:
        crop_x, crop_y = 0, 0
    else:
        crop_x, crop_y = get_crop_adjust(crop)

    animal_name = get_animal_name(vid_file)

    times['loading files'] = f'some house-keeping: {time.time() - start} seconds'
    ##############################################################################################
    start = time.time()

    time_series = create_time_series(deep_cut, crop_x, crop_y)
    time_series['time'] = get_time_elapsed(time_series, vid_file)
    time_series['likelihoods'] = get_likelihoods(deep_cut)

    times['algorithm'] = f'actual bottleneck: {time.time() - start} seconds'
    ##################################################################################################

    plot_rotated_rect_area(time_series['rotated rect area'], path, animal_name)

    start = time.time()
    plot_blinks(time_series, deep_cut, vid_file, crop_x, crop_y, path, animal_name)
    times['plot blinks'] = f'plotting the blinks {time.time() - start}'

    plot_axes_ratio_dist(time_series['major minor ratio'], path, animal_name)

    plot_verify_grid(10, time_series, path, deep_cut, crop_x, crop_y, vid_file, animal_name)
    times['verify grid'] = f'verify grid {time.time() - start}'

    params = {'fs': get_frame_rate(vid_file), 'hp': 10, 'lp': 0.01, 'order': 3, 's_hp': 10}
    time_series = preprocess_pupil(params, time_series)
    plot_preprocessing(time_series, path, animal_name)

    save_csv = path + '/' + animal_name + 'time_series.csv'
    time_series.to_csv(save_csv, index=False)

    for key in times:
        print(times[key])
