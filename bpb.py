try:
    import Imagee
except ImportError:
    from PIL import Image
import numpy as np 
import cv2 as cv
import numpy
import matplotlib.pyplot as plt
import os
from PIL import ImageDraw
from scipy.spatial import ConvexHull, convex_hull_plot_2d, Delaunay
import alphashape

isbi2016_ground_truth_path = '/Users/muratkara/Downloads/ISBI2016_ISIC_Part1_Training_GroundTruth/'
image_writing_folder = '/Users/muratkara/Desktop/yüksek lisans çalışmalar/501/image_writing_folder/';
sz = (256, 256)

T = 1 #number of iterations
iou_best_image = []
n = 3 # number of points

def ccw_sort(p):
    p = np.array(p)
    mean = np.mean(p,axis=0)
    d = p-mean
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]


def mean_iou(y_true, y_pred):
	inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
	union = tf.math.count_nonzero(tf.add(yt0, yp0))
	iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
	return iou

ground_truth_images_path = os.listdir(isbi2016_ground_truth_path)
SGT = Image.open(isbi2016_ground_truth_path + ground_truth_images_path[2])
# SGT.show()
SGT = np.array(SGT.resize(sz))

edges = cv.Canny(SGT, 100, 200)
edges[edges != 0] = 1
nonzero_edges = np.nonzero(edges)
contours = [ (nonzero_edges[1][i], nonzero_edges[0][i]) for i in range(len(nonzero_edges[1])) ]


random_indices = np.random.randint(len(contours), size=n)



points = np.zeros((len(random_indices),2))
for i in range(len(random_indices)):
	index = random_indices[i]
	points[i] = contours[index]



# plt.figure()
# plt.axis('equal')
# plt.plot(points[:, 0], points[:, 1], '.')
# for i, j in edges:
#     plt.plot(points[[i, j], 0], points[[i, j], 1])
# plt.show()




# contours1 = []
# for j in reversed(range(len(edges))):
# 	for i in reversed(range(len(edges))):
# 		if(edges[i][j] != 0):
# 			contours1.append((j, i))

# nonzero_edges = np.nonzero(edges)

# print(nonzero_edges)
# edges2 = np.zeros(sz)
# edges2[nonzero_edges[0]][nonzero_edges[1]] = 255

# data = Image.fromarray(edges2)
# data.resize(sz)
# data.show()


# contours1 = [( nonzero_edges[1][i], nonzero_edges[0][i]) for i in range(len(nonzero_edges[1]))]
# [edges[t[0]][t[1]] for t in contours]

# random_indices = np.random.randint(len(contours1), size=n)
# random_indices = np.sort(random_indices)

# # print(random_indices)
# print('random_indices', random_indices)
# # print('contours', contours1)


# contours = []
# for i in range(len(random_indices)):
# 	index = random_indices[i]
# 	contours.append(contours1[index])
# print(contours)


# # # nonzero_edges = np.nonzero(edges)

# # # print(contours)


# contours1 = sorted(contours1, key=lambda tup: (tup[1],tup[0]) ) 
points = ccw_sort(points)
points = [tuple(point) for point in points]
img = Image.new("RGB", sz, '#fff')
img1 = ImageDraw.Draw(img)
img1.polygon(points, fill ="#000", outline ="#000") 
img.show()

plt.imsave(image_writing_folder + 'drawn image' + str(0) + ".jpg" , img)

# # nonzero_indices = np.nonzero(edges)


# for i in range(T):
# 	# indices = np.sort(np.random.randint(len(nonzero_indices[0]), size=n))
# 	# rand_pixels_x = nonzero_indices[0][indices]
# 	# rand_pixels_y = nonzero_indices[1][indices]
# 	# contours = [(rand_pixels_x[i], rand_pixels_y[i]) for i in range(n)]
	
# 	img = Image.new("RGB", sz, '#000')
# 	img1 = ImageDraw.Draw(img)
# 	img1.polygon(contours, fill ="#eeeeff", outline ="blue") 
# 	img.show()

# 	# plt.imsave(image_writing_folder + 'drawed image' + str(i) , img)

