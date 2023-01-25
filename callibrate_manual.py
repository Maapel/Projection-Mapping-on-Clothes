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
        POS.append((x,y))


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
# print(POS)
if len(POS) >= 2:
    x_start,y_start = POS[0]
    x2,y2 = POS[1]
    w_projection = (x2-x_start)
    h_projection = (y2-y_start)
    if w_projection<0:
        temp  =x_start
        x_start =x2
        x2 = x_start
        w_projection = (x2 - x_start)
    if h_projection<0:
        temp  =y_start
        y_start =y2
        y2 = y_start
        h_projection = (y2 - y_start)

    cv2.rectangle(white_rec,(x_start,y_start),(x_start+w_projection,y_start+h_projection),(155,155,0),2)

cv2.imshow("foo",white_rec)
cv2.waitKey()
cv2.namedWindow("foo", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("foo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


print("("+str(x_start)+"," +str(y_start)+"," +str(w_projection)+","+ str(h_projection)+")" )
while cv2.waitKey(1)!=27:
    ret,img = src.read()
    img = img[y_start:y_start + h_projection, x_start:x_start + w_projection]
    # cv2.imwrite("m.png",img)
    cv2.imshow("foo",img)
    cv2.waitKey(1)