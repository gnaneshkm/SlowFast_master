import cv2
import numpy as np
import glob

img_array = []
filenames = [img for img in glob.glob("/home/gnk/data_ava/data/ava/frames/train/*.png")]
filenames.sort()
for filename in filenames:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (1024,1024)
    img_array.append(img)

out = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*'DIVX'), 5, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()