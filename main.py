import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import functions as fun
from Line import Line


window_size = 10  # how many frames for line smoothing
left_line = Line(n=window_size)
right_line = Line(n=window_size)
detected = False  # did the fast line fit detect the lines?
left_curve, right_curve = 0., 0.  # radius of curvature for left and right lanes

def pipeline(frame):

    global detected

    # applay Perspective transform
    perp, m_inv = fun.warp(frame)

    # # Thresholded binary image
    sxbinary = fun.abs_sobel_thresh(perp, orient='x', sobel_kernel=3,thresh=(50,255))
    s_binary = fun.HLS_select(perp,(75,255))
    binary_warped = cv2.add(s_binary,sxbinary)

    # Polynomial fit
    if not detected:
        # Slow line fit
        ret = fun.fit_polynomial(binary_warped)
        left_fit = ret['left_fit']
        right_fit = ret['right_fit']
        # Get moving average of line fit coefficients
        left_fit = left_line.add_fit(left_fit)
        right_fit = right_line.add_fit(right_fit)
        # slow line fit always detects the line
        detected = True
    else:
        # Fast line fit
        left_fit = left_line.get_fit()
        right_fit = right_line.get_fit()
        ret = fun.search_around_poly(binary_warped, left_fit, right_fit)
        # Only make updates if we detected lines in current frame
        if ret is not None:
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            left_fit = left_line.add_fit(left_fit)
            right_fit = right_line.add_fit(right_fit)
        else:
            detected = False


    # Perform final visualization on top of original undistorted image
    result = fun.viz(frame, left_fit, right_fit, m_inv,)

    return result


## test image ##
# image = plt.imread('test_images/straight_lines2.jpg')
# image = plt.imread('bridge_shadow.jpg')
# result = pipeline(image)

# plt.imshow(result)
# plt.show()




## uncomment to test video file ##

cap = cv2.VideoCapture('challenge.mp4')
ret, frame = cap.read()
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,720))


while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        final = pipeline(frame)
        out.write(final)
        cv2.imshow('frame', final)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
out.release()
cv2.destroyAllWindows()

