"""
Based on turorial series:
part 1: https://www.youtube.com/watch?v=dK-KxuPi768
part 2: https://www.youtube.com/watch?v=hO6D6DEYxQE
part 3: https://www.youtube.com/watch?v=hcINR237U6E
"""

import cv2
import numpy as np
from pathlib import Path
import dlib

## load image
path = Path.cwd() / 'realperson' / 'astro.jpg'
img = cv2.imread(str(path))

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(img_gray)

## detect the face and the facial landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
faces = detector(img_gray)
for face in faces:
    landmarks = predictor(img_gray, face)
    landmarks_points = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))
        # cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

    ## find the boundaries sorrounding the face
    points = np.array(landmarks_points, np.int32)
    convexhull = cv2.convexHull(points)
    # cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
    cv2.fillConvexPoly(mask, convexhull, 255)

    ## extract the face
    face_image_1 = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("Image 1", img)
cv2.imshow("Face image 1", face_image_1)
cv2.imshow("Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
