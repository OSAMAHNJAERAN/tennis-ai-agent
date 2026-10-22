The names of the videos (after removing suffixes like .mp4 and .avi) correspond one-to-one with the folder names under the 'tennis/Labels/' directory which are from https://doi.org/10.5061/dryad.3bk3j9m13. Those folders contain the annotation information for each image frame of the videos with the same names.

The label and Video Index are from 0, here is a example to get corresponding image of the label:

```
import cv2
# Suppose Now We Have a.mp4 and a_index.txt, the index in real are 0,1,2,...,999 .etc
with open("a_index.txt",'r') as f:
	labels = f.readlines()
	f.close()
frameIndex = int("a_index.txt".split("_")[-1][0:-4])
cap = cv2.VideoCapture("a.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex) # Please Use set
ret,frame = cap.read()

# Now frame,label has get matched
```

What‘s more, some video are too big to upload once, we slice them to multi videos. i.e. a.mp4 will be slice to a\_(1).mp4, a\_(2).mp4, a\_(3).mp4,..... Please concate them first like below

```
import os
import cv2


videoNames = ['a_(1).mp4','a_(2).mp4','a_(3).mp4']
newVideoName = "a.mp4"


cap = cv2.VideoCapture(videoNames[0])

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(newVideoName, fourcc, fps, (width, height))


for videoName in videoNames:
    cap = cv2.VideoCapture(os.path.join(basicFolder,capPath))
    frameId = 0
    while True:
        ret,frame = cap.read()
        if ret is False:
    		break
        else:
            frameId += 1
            out.write(frame)
    cap.release()
out.release()
  
```
