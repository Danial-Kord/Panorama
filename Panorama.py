import cv2
import numpy as np


def RANSAC(src_pts, dst_pts, num_iterations, threshold):
    best_homography = None
    best_inliers = []
    for i in range(num_iterations):
        # Randomly select 4 pairs of matched points
        random_indices = np.random.randint(0, len(src_pts), 4)
        random_src_pts = src_pts[random_indices]
        random_dst_pts = dst_pts[random_indices]

        # Estimate the homography matrix using the random points
        homography_matrix, _ = cv2.findHomography(random_src_pts, random_dst_pts)


        # Calculate the distances between the transformed source points and the destination points
        dst_pts_transformed = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), homography_matrix)
        difference = dst_pts - dst_pts_transformed.reshape(-1, 2)
        distances = np.linalg.norm(difference, axis=1)

        # Select the inliers based on the threshold
        inliers = np.where(distances < threshold)[0]
        print(inliers)
        # Update the best homography and inliers if the current set is better
        if len(inliers) > len(best_inliers):
            best_homography = homography_matrix
            best_inliers = inliers

    return best_homography, best_inliers




# Load the two images
img1 = cv2.imread("2.png")
img2 = cv2.imread("1.png")

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
    if m.distance < 0.9*n.distance:
        good.append(m)

# Draw the matches using drawMatches function
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# save SIFT
cv2.imwrite("2_matches.png", img3)


# Convert the matched keypoints to coordinates
src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

# Homography and RANSAC
homography_matrix, inliers = RANSAC(src_pts, dst_pts, 100, 2500)

# save Inliers
inlier_matches = [good[i] for i in inliers]
img_inliers = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor = (0,255,0))
cv2.imwrite("2_inliers.png", img_inliers)

# save Outliers
outlier_matches = [good[i] for i in range(len(good)) if i not in inliers]
img_outliers = cv2.drawMatches(img1, kp1, img2, kp2, outlier_matches, None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor = (255,0,0))
cv2.imwrite("2_outliers.png", img_outliers)

homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0)

result = cv2.warpPerspective(img2, homography_matrix, (img1.shape[1] + img2.shape[1], img1.shape[0]))

# Create a separate image for the destination
dst = np.zeros((img1.shape[0], img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)

# Copy the first image to the destination image
dst[0:img1.shape[0], 0:img1.shape[1]] = img1

# Copy the transformed second image to the destination image
dst[0:img1.shape[0], img1.shape[1]:img1.shape[1]+img2.shape[1]] = result[0:img1.shape[0], 0:img2.shape[1]]
# Save the result
cv2.imwrite("panorama.png", dst)

# exit(0)

# # Get the size of the combined panorama
# rows1, cols1 = img1.shape[:2]
# rows2, cols2 = img2.shape[:2]
#
# # Apply the homography to get the panorama
# homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#
#
# # Apply the homography matrix to the second image
# h, w,_ = img1.shape
# pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
# dst = cv2.perspectiveTransform(pts,homography_matrix)
# print(dst)
# print(dst.shape)
#
# # Calculate the size and offset of the panorama
# h_panorama, w_panorama = h, w
# offset_x, offset_y = 0, 0
# for (x, y) in dst:
#     x, y = x[0] + offset_x, y[0] + offset_y
#     w_panorama = max(w_panorama, int(x))
#     h_panorama = max(h_panorama, int(y))
#
# # Create an empty panorama image
# panorama = np.zeros((h_panorama, w_panorama, 3), dtype=np.uint8)
#
# # Copy the first image to the panorama
# panorama[offset_y:offset_y + h, offset_x:offset_x + w] = img1
#
# h_transformed, w_transformed, _ = img2.shape
# pts = np.float32([[0, 0], [0, h_transformed-1], [w_transformed-1, h_transformed-1], [w_transformed-1, 0]]).reshape(-1, 1, 2)
# dst = cv2.perspectiveTransform(pts, homography_matrix)
#
# min_x, min_y = np.inf, np.inf
# for (x, y) in dst:
#     x, y = x[0] + offset_x, y[0] + offset_y
#     min_x, min_y = min(min_x, x), min(min_y, y)
#
# offset_x, offset_y = int(min_x), int(min_y)
# panorama_roi = panorama[offset_y:offset_y + h_transformed, offset_x:offset_x + w_transformed]
# gray2_warped = cv2.warpPerspective(img2, homography_matrix, (w_transformed, h_transformed))
# mask = gray2_warped != 0
# panorama_roi[mask] = gray2_warped[mask]
#
# cv2.imshow("Panorama", cv2.resize(panorama, (900,500)))
# cv2.waitKey(0)