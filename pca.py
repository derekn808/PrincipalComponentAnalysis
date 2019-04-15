from glob import glob
import os
import cv2 as cv
import numpy as np
import fnmatch

#get training set
#convert each face into a vector of size nxm (number of pizels in a picture)
#add each vector into a matrix with each face as a different column
#Use normalization on matrix then use SVD on matrix to compute PCA:
#normalize the faces by removing all common features, leaving only unique feature
#find k nearest neighbor

# K=1,2,3,6,10,20, and 30
# 1,3,4,5,7,9 are training images, and 2,6,8,10 are testing images
#convert each image to vector of size length D=112*92=10304
#Stack 6 training image sets with 10 subjects in each to from a matrix of size 10304*60

#Calculate the recognition accuracy for different K values.
#Comment on the results of each k value

class PCA:

	def __init__(self):
		k = (1,2,3,6,10,20,30)	#different values of components for pca
		dir_path = os.path.dirname(os.path.realpath(__file__))	#get directory of current file
		self.file_dir = os.path.join(dir_path, 'att_faces_10')	#get directory of training and test files
		self.file_dir = glob(os.path.join(self.file_dir,'**','*'))
		training_matrix = None	#initialize empty training matrix
		

		#fnmatch.filter(files, pattern)
		for directory in os.listdir(self.file_dir):	#check all files in directory
			file_name = os.path.basename(directory)
			if file_name == 's1' or file_name == 's3' or file_name == 's4' or file_name == 's5' or file_name == 's7' or file_name == 's9':	
				for image in directory:	#loop through directories with training sets
					if fnmatch.fnmatch(image, '*.pgm'):
						img_name = os.path.join(img_name, image)
						imgraw = cv.imread(img_name, 0)	#read image as grayscale
						imgraw = np.array(imgraw) #change image to array of pixel values
						img_flat = imgraw.flatten()	#flatten into 1D vector
						training_matrix = np.column_stack((training_matrix, img_flat))	#append to matrix
		
		#PCA
		#Rotations of matrix - U and V, U has principal components (only variable that will be used)
		#Diagonal of matrix - S (sigma)
		normalize_mat = np.linalg.norm(training_matrix)
		U, S, V = np.linalg.svd(normalize_mat, full_matrices = True)

		for i in k:	#test each value of k components
			#highest rows in U have most important components
			#cut out k rows from U
			X = U[:, 0:k[i]]	#slice U for current k i.e. top 2 rows of U for k=2
			U_transpose = X.transpose()	#transpose the read image
			print('Displaying', k[i], 'components:\n')
			testSet(U_transpose)

	def testSet(self, U):
		for directory in os.listdir(self.file_dir):
			file_name = os.path.basename(directory)
			if file_name == 's2' or file_name == 's6' or file_name == 's8' or file_name == 's10':	
				for image in directory:
					if fnmatch.fnmatch(image, '*.pgm'):
						img_name = os.path.join(directory, image)
						imgraw = cv.imread(img_name, 0)
						imgraw = np.array(imgraw)
						img_flat = imgraw.flatten()	#create vector of read image
						img_dot = np.dot(img_flat, U)	#compute dot product of read image and selected principal components
						img_tile = np.tile(img_dot, 10304)	#copy dot for the size of the image resolution
						closest_neighbor = np.argmin(img_dot - img_tile)	#calculate k closest neighbor by finding min of weights
						print('K closest neighbor for ', img_name, 'is: ',closest_neighbor,'\n')

test = PCA()