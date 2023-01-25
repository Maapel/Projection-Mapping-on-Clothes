import cv2
import numpy as np
import time
src = cv2.VideoCapture(0)
#cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

POS =[]

def mouse_click(event, x, y,
                flags, param):
    # to check if left mouse
    # button was clicked

    if event == cv2.EVENT_LBUTTONDOWN:
        POS.append([x,y])


res = (1920,1080)
white_im = np.zeros((res[1],res[0]))
white_im[:,:]=   255

cv2.waitKey()
im =white_im
cv2.namedWindow("foo", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("foo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("foo", im)
cv2.waitKey()
for p in range(10):
    ret , white_rec = src.read()
cv2.imshow("foo",white_rec)
cv2.setMouseCallback("foo",mouse_click)
cv2.waitKey(0)
pts_src =np.array([[0,0],[res[0],0],[res[0],res[1]],[0,res[1]]])
POS = np.array(POS)
print(POS)

if len(POS) >= 4:
    cv2.polylines(white_rec,[POS],True,(155,155,0),2)

cv2.imshow("foo",white_rec)
cv2.waitKey()
cv2.namedWindow("foo", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("foo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

(tl, tr, br, bl) = POS
# Finding the maximum width.
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))
# Finding the maximum height.
heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxHeight = max(int(heightA), int(heightB))
# Final destination co-ordinates.
destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]

M = cv2.getPerspectiveTransform(np.float32(POS), np.float32(destination_corners))
print(M, destination_corners)

while cv2.waitKey(1)!=27:
    ret,img = src.read()
    M = cv2.getPerspectiveTransform(np.float32(POS), np.float32(destination_corners))
    # Perspective transform using homography.
    img = cv2.warpPerspective(img, M, (destination_corners[2][0], destination_corners[2][1]),
                                flags=cv2.INTER_LINEAR)
    # img = img[y_start:y_start + h_projection, x_start:x_start + w_projection]
    # cv2.imwrite("m.png",img)
    cv2.imshow("foo",img)
    cv2.waitKey(1)