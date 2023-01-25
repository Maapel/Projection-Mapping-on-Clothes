import cv2
import numpy as np
import time
src = cv2.VideoCapture(1)
#cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)


res = (1920,1080 )
white_im = np.zeros((res[1],res[0]))
white_im[:,:]=   255

cv2.waitKey()
im =white_im
cv2.namedWindow("foo", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("foo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("foo", im)
cv2.waitKey()
cv2.destroyWindow("foo")
for p in range(10):
    ret , white_rec = src.read()
white_rec = cv2.GaussianBlur(white_rec, (11, 11), 0)
white_rec = cv2.cvtColor(white_rec,cv2.COLOR_BGR2GRAY)
ret, white_rec = cv2.threshold(white_rec, 50, 255, cv2.THRESH_BINARY)
cv2.imshow("h",white_rec)
cv2.waitKey()
contours, hierarchy = cv2.findContours(white_rec, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
ctr = cv2.cvtColor(white_rec, cv2.COLOR_GRAY2BGR)

areas = [cv2.contourArea(c) for c in contours]
if len(areas) > 0:
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    x_start, y_start, w_projection, h_projection = cv2.boundingRect(cnt)

y_ = y_start +  h_projection //2
arr = white_rec[y_][x_start:x_start + w_projection]
x2 = x_start+ w_projection
c = 0
for p in range(len(arr)):
    if (arr[p] != 0 and c == 0):
        x1 = p + x_start
        c += 1
    elif (arr[p] == 0 and c == 1):
        c += 1
        x2 = p + x_start
        break
x_start = x1
w_projection=x2-x1
cv2.rectangle(white_rec,(x_start,y_start),(x_start+w_projection,y_start+h_projection),(155,0,0),2)
cv2.imshow("h",white_rec)
cv2.waitKey(5)
cv2.namedWindow("foo", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("foo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


print(x_start, y_start, w_projection, h_projection )
for i in range(200):
    ret,img = src.read()
    img = img[y_start:y_start + h_projection, x_start:x_start + w_projection]
    # cv2.imwrite("m.png",img)
    cv2.imshow("foo",img)
    cv2.waitKey(1)