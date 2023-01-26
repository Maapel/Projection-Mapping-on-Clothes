import numpy as np
import cv2
from matplotlib import pyplot as plt
import pickle

src = cv2.VideoCapture(0)
#cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
f1 = open("M_Pickle",'rb')
f2 = open("dest_Pickle",'rb')
M = pickle.load(f1)

destination_corners = pickle.load(f2)

f1.close()
f2.close()
ada = cv2.VideoCapture("Water Dance - 112512.mp4")
res = (1920,1080 )
# x_start, y_start, w_projection, h_projection=(98 ,102 ,497 ,280)

cv2.namedWindow("foo", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("foo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
black_im = np.zeros((res[1], res[0]))
for i in range(20):
    cv2.imshow("foo",black_im)
    cv2.waitKey(1)
cv2.waitKey()
i = 0
while cv2.waitKey(1) != 27:
    i+=1
    ret, img = src.read()
    ma,des = ada.read()
    # img = img0
    img = cv2.warpPerspective(img, M, (destination_corners[2][0], destination_corners[2][1]),
                                flags=cv2.INTER_LINEAR)
    img  = cv2.resize(img,res)
    print(img)
    # cv2.imwrite("images/" + str(i) + "_0.png", img)

    # ret = True
    # img = cv2.imread("img1.jpg")
    if not (ret):
        break
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # faces = faceCascade.detectMultiScale(
    #     gray,
    #     scaleFactor=1.3,
    #     minNeighbors=3,
    #     minSize=(30, 30)
    # )
    # img = gray

    # try:
    #     face = faces[0]
    #     # final = img_blur
    #     # img = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,10 )
    #     img[face[1]:face[1] + 7*face[3]//6, face[0]:face[2] + 7*face[0]//6] = 0
    # except Exception as e:
    #     pass
    ret, img = cv2.threshold(img, 75, 255, cv2.THRESH_BINARY)


    # print(img)ror: (-215:Assertion failed) !ssize.empty() in function 'cv::resize

    # for face in faces:
    #     cv2.rectangle(img,(face[0],face[1]),(face[2]+face[0],face[3]+face[1]),(0,0,4),2)


    # diff = cv2.absdiff(img,prev)
    # thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ctr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    areas = [cv2.contourArea(c) for c in contours]
    if len(areas) > 0:
        max_index = np.argmax(areas)
        # areas[max_index]=0
        # max_index = np.argmax(areas)

        cnt = contours[max_index]

        x, y, w, h = cv2.boundingRect(cnt)
        y_ = y +  h //2
        arr = img[y_][x:x + w]
        c = 0
        x2 = x+ w
        for p in range(len(arr)):
            if (arr[p] != 0 and c == 0):
                x1 = p + x
                c += 1
            elif (arr[p] == 0 and c == 1):
                c += 1
                x2 = p + x
                break

        # w_=x2-x1
        x_ = (x1 + x2) // 2
        x_1 = x_ - w // 2
        x_2 = x_ + w // 2
        y_1 = y_ - h // 2
        y_2 = y_ + h // 2
        # print("x1 x2 y1 y2 ",x_1,x_2,y_1,y_2)
        # cv2.rectangle(ctr, (x_1, y_1), (x_2, y_2), (0, 25, 20), -1)
        background = img
        background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        m = des  # IMREAD_UNCHANGED => open image with the alpha channel
        m = cv2.resize(m, (x_2 - x_1, y_2 - y_1))
        k = np.full(shape=(m.shape[0], m.shape[1]), fill_value=255)
        overlay = np.dstack((m, k))
        overlay[y_1:y_2, x_1:x_2,3] = 255
        alpha_channel = overlay[:, :, 3] / 255  # convert from 0-255 to 0.0-1.0
        overlay_colors = overlay[:, :, :3]
        alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

        h, w = overlay.shape[:2]
        background_subsection = background[y_1:y_2, x_1:x_2]

        composite = overlay_colors * alpha_mask

        background[y_1:y_2, x_1:x_2] = composite
        a = np.zeros_like(background)
        a[y_1:y_2, x_1:x_2] = composite
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        background = cv2.bitwise_and(img, background)
        a = cv2.bitwise_and(img, a)
        # cv2.imshow("p", img)
        # cv2.imshow("p1", img0)
        # cv2.imwrite("images/"+str(i)+"_1.png" , img)
        # cv2.imwrite("images/"+str(i)+"_2.png" , a)

        cv2.imshow("foo", a)

        #a , background1 = cv2.threshold(background, 0, 200, cv2.THRESH_BINARY)
        #cv2.imshow("Face detected", ctr)
    # cv2.imshow("window", img0)

    # cv2.imshow("gray",background)
    # cv2.imshow("foo", a)


src.release()
cv2.destroyAllWindows()