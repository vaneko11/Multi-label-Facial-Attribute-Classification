# import
from LBP import LocalBinaryPatterns
from HOG import HistogramofOrientedGradients
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import cv2
import glob
import random

#lbp
LBP = LocalBinaryPatterns(256, 8)
hists = []

#hog
HOG = HistogramofOrientedGradients()
fds = []

#img
data = []
for img in glob.glob("img_align_celeba/*.jpg"):
    n= cv2.imread(img)
    data.append(n)
random.shuffle(data)	 

# loop over the training images
print(len(data))
train_data_size=int(202599*0.7)


for image in data[:train_data_size]:
		
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)		
	hist = LBP.describe(gray)
	hists.append(hist)
	fd = HOG.describe(gray)
	fds.append(fd)

# loop over the testing images
hists_test=[]
fds_test=[]

for image in data[train_data_size+1:]:
	
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)	
	hist = LBP.describe(gray)
	hists_test.append(hist)
	fd = HOG.describe(gray)
	fds_test.append(fd)	

f = open('vysl.txt','w')

for a in range(1,41):
	print(a)
	# read attr
	fname='Atr/atr_' + str(a) + '.txt'
	label = []
	file = open(fname, 'r')
	label = file.read()
	file.close()
	label=label.split(" ") 
	
	with open('sample.txt', "w") as fl:
		fl.write(str(len(label[:train_data_size]))+' '+str(len(label[:train_data_size][0]))+'\n')
	
	# train a LinearSVC and SVC
	model1 =  SVC(kernel='rbf', C=1.0)
	model1.fit(hists, label[:train_data_size])

	model2 =  SVC(kernel='rbf', C=1.0)
	model2.fit(fds, label[:train_data_size])

	model3 =  LinearSVC(C=1.0, dual=True, max_iter=5000)
	model3.fit(hists, label[:train_data_size])

	model4 =  LinearSVC(C=1.0, dual=True, max_iter=5000)
	model4.fit(fds, label[:train_data_size])

	score1=model1.score(hists_test, label[train_data_size+1:len(data)])
	score2=model2.score(fds_test, label[train_data_size+1:len(data)])
	score3=model3.score(hists_test, label[train_data_size+1:len(data)])
	score4=model4.score(fds_test, label[train_data_size+1:len(data)])


	#print('Popis LBP atribut-' + str(a) + ': ' + str(score1) + ' (SVC)')
	#print('Popis HOG artibut-' + str(a) + ': ' + str(score2) + ' (SVC)')
	#print('Popis LBP atribut-' + str(a) + ': ' + str(score3) + ' (LinearSVC)')
	#print('Popis HOG artibut-' + str(a) + ': ' + str(score4) + ' (LinearSVC)')
	f.write('Popis LBP atribut-' + str(a) + ': ' + str(score1) + ' (SVC)\n')
	f.write('Popis HOG artibut-' + str(a) + ': ' + str(score2) + ' (SVC)\n')
	f.write('Popis LBP atribut-' + str(a) + ': ' + str(score3) + ' (LinearSVC)\n')
	f.write('Popis HOG artibut-' + str(a) + ': ' + str(score4) + ' (LinearSVC)\n') 

f.close()

