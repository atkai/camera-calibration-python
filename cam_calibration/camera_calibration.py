# Camera calibration
# Date: 2021/4/10
# Modified:  [Add the length of the grid: 2021/5/1]
# Reference: [CSDN]Hjw_52:Python implementation of Zhang Zhengyou's camera calibration method


import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import glob


# Camera calibration based on opencv
# 1. Read pictures in a loop
# 2. Use the findChessboardCorners function to detect corner points
#    (you need to enter the number of corner points in advance).
# 3. Use the CornerSubpix function to perform sub-pixel precision on the corner points
# 4. Use drawChessboardCorners to display the corner points.
# 5. Create an ideal checkerboard based on the number and size of corners
#    (use point vectors to store all theoretical corner coordinates).
# 6. Use the calibrateCamera function to calibrate the ideal coordinates and actual \
#    image coordinates to obtain the calibration results.
# 7. Calculate the back projection error by the projectPoints function.

def show(img):
    if img.ndim == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()


# Checkerboard specifications
row = 9
col = 6

# Set the coordinates of a point in world coordinates
objp = np.zeros((row * col, 3), np.float32)
length = 30  # The length of the black and white grid
objp[:, :2] = np.mgrid[0:row * length:length, 0:col * length:length].T.reshape(-1, 2)

objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in camera space

# Here you need to change the file path of the left or right camera
imgs = glob.glob("./right/*.png")  # Get the addresses of all images

for fname in imgs:
    # Read grayscale image
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the corner point, ret is the sign of whether the corner point is found
    ret, corners = cv.findChessboardCorners(img, (row, col), None)
    if ret:  # Judge whether the corner point is found
        # Perform sub-pixel corner detection
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv.cornerSubPix(gray, corners, (row, row), (-1, -1), criteria)

        # Visualize corner points on the chessboard
        # img = cv.drawChessboardCorners(img, (row, col), corners2, ret)
        # show(img)

        # Record the coordinate information of the corner points
        objpoints.append(objp)
        imgpoints.append(corners2)

# mtx: camera internal parameters;
# dist: distortion coefficient;
# revcs: rotation matrix;
# tvecs: translation matrix
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Read one of the images
img0 = cv.imread('./right/chessboard-R0.png')

# Get the height and width information of the image
h, w = img0.shape[:2]

'''
Optimize the camera matrix. This step is optional.
Parameter 1 means that all pixels are reserved, and black pixels may be introduced at the same time.
Set to 0 to cut out unwanted pixels as much as possible, this is a scale, 0-1 can be taken.
'''
new_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

# Correct distortion
dst = cv.undistort(img0, mtx, dist, None, new_mtx)

# Output the picture after correcting the distortion
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv.imwrite('result.png', dst)

# Output camera parameters
print("Original camera internal parameters:\n", mtx)
print("Camera internal parameters after optimization:\n", new_mtx)
print("Distortion coefficient:\n", dist)

# Calculate error
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    total_error += error

print("\nTotal error: ", total_error / len(objpoints))
