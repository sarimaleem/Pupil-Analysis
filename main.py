from joblib import Parallel, delayed
import os
from ExtractPupil import create_files
import time

start = time.time()

directory = 'C:/Users/Sarim Aleem/Documents/PupilProject/data/cropped'

folders = [x[0] for x in os.walk(directory)][1:]
#create_files(folders[0])

end = time.time()

print(end - start)

#Parallel(n_jobs=8)(delayed(create_files)(path) for path in folders)