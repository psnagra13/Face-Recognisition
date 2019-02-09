
# coding: utf-8

# # Face Detection System

# ### Importing required Libraries

# In[45]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
# get_ipython().magic(u'matplotlib inline')


# ### Read all jpg files in folder 

# In[25]:


images_directory = 'data/images'
images_path_list = [] 
for file in glob.glob(images_directory + '/*.jpg'):
    images_path_list.append(file)
print ('Number of images found = ' + str(len(images_path_list)))


# ### Reading images

# In[31]:


def read_image (path):
    image = cv2.imread(path,cv2.IMREAD_COLOR)
    image_copy = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert image to grayscale
    return image


# ### Load face and eyes Cascade Classifier

# In[35]:


facecascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')


# In[43]:


def face_detection(image):
    faces = facecascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)
    return faces

    


# In[46]:


for image_path in images_path_list:
    print (image_path)
    image = read_image (image_path)    
    faces = face_detection(image)   
 
    print("Total number of Faces found",len(faces))
    
    for (x, y, w, h) in faces:
        face_detect = cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 255), 2)
        roi_gray = image[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        plt.imshow(face_detect)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            eye_detect = cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,255),2)
            plt.imshow(eye_detect)

