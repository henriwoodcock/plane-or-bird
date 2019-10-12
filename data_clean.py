from matplotlib.image import imread
import numpy as np
import glob


plane_train_dir = "/Users/henriwoodcock/Documents/Code/data projects /plane_or_bird/data/train/airplane"
plane_test_dir = "/Users/henriwoodcock/Documents/Code/data projects /plane_or_bird/data/test/airplane"
bird_train_dir = "/Users/henriwoodcock/Documents/Code/data projects /plane_or_bird/data/train/bird"
bird_test_dir = "/Users/henriwoodcock/Documents/Code/data projects /plane_or_bird/data/test/bird"


#plane train - convert to numpy
plane_training_set = []
plane_training_y = []
file_list = glob.glob(plane_train_dir + "/*.png")
for file in file_list:
	img = imread(file)
	plane_training_set.append(img)
	plane_training_y.append(0)

plane_training_set = np.array(plane_training_set)
plane_training_y = np.array(plane_training_y)


plane_test_set = []
plane_test_y = []
file_list = glob.glob(plane_test_dir + "/*.png")
for file in file_list:
	img = imread(file)
	plane_test_set.append(img)
	plane_test_y.append(0)

plane_test_set = np.array(plane_test_set)
plane_test_y = np.array(plane_test_y)

bird_train_set = []
bird_train_y = []
file_list = glob.glob(bird_train_dir + "/*.png")
for file in file_list:
	img = imread(file)
	bird_train_set.append(img)
	bird_train_y.append(1)

bird_train_set = np.array(bird_train_set)
bird_train_y = np.array(bird_train_y)


bird_test_set = []
bird_test_y = []
file_list = glob.glob(bird_test_dir + "/*.png")
for file in file_list:
	img = imread(file)
	bird_test_set.append(img)
	bird_test_y.append(1)

bird_test_set = np.array(bird_test_set)
bird_test_y = np.array(bird_test_y)

x_train = np.concatenate([plane_training_set, bird_train_set])
y_train = np.concatenate([plane_training_y, bird_train_y])
x_test = np.concatenate([plane_test_set, bird_test_set])
y_test = np.concatenate([plane_test_y, bird_test_y])
