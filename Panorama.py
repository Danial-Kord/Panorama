import cv2
import numpy as np

# Load the two images
img1 = cv2.imread("1.png")
img2 = cv2.imread("2.png")

# Initialize SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# Compute the keypoints and descriptors for both images
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# Use BFMatcher to find the best matches
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Filter the matches based on their distance
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)

# Draw the matches using drawMatches function
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the result
cv2.imshow("Matched Features", img3)
cv2.waitKey()
cv2.destroyAllWindows()
