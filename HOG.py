#import
from skimage.feature import hog

#class lbp
class HistogramofOrientedGradients:
	def __init__(self):
		pass	
	
	def describe(self,image):
			fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16,16),cells_per_block=(1, 1), visualize=True)
			return fd