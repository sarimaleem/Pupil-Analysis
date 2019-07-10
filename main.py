from joblib import Parallel, delayed
import os
from ExtractPupil import create_files
import time

start = time.time()
directory = 'C:/Users/Sarim Aleem/Documents/PupilProject/data/cropped'

folders = [x[0] for x in os.walk(directory)][1:]
#create_files(folders[0]) #if you only want to do one of the directories

Parallel(n_jobs=48)(delayed(create_files)(path) for path in folders)

end = time.time()
print(end - start)
