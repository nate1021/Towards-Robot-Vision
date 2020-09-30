import cv2
import json
import os
import time

mouseDown = False
mouseX = 2
mouseY = 2
mouseStartX = 2
mouseStartY = 2
mouseEndX = 2
mouseEndY = 2

FILE_PATH = '../../data/region.json'
region = {}
if os.path.isfile(FILE_PATH):
    with open(FILE_PATH, 'r') as fp:
        region = json.load(fp)

        mouseStartX = region['StartX']
        mouseStartY = region['StartY']
        mouseEndX = region['EndX']
        mouseEndY = region['EndY']

window_title = "Color Tracking"
# capturing video through webcam
cap = cv2.VideoCapture(0)
time.sleep(2)


def save():
    with open('region.json', 'w') as fp:
        json.dump(region, fp)
    print('Region Saved:\n %s' % region)


def draw_circle(event,x,y,flags,param):
    global mouseX, mouseY, mouseStartX, mouseStartY, mouseEndX, mouseEndY, mouseDown
    mouseX, mouseY = x, y

    if event == cv2.EVENT_LBUTTONDOWN:
        mouseDown = True
        mouseStartX = mouseX
        mouseStartY = mouseY

    if event == cv2.EVENT_LBUTTONUP:
        mouseDown = False
        msy = mouseStartY
        msx = mouseStartX
        mouseStartX = min(mouseX, msx)
        mouseStartY = min(mouseY, msy)
        mouseEndX = max(mouseX, msx)
        mouseEndY = max(mouseY, msy)

        region['StartX'] = mouseStartX
        region['StartY'] = mouseStartY
        region['EndX'] = mouseEndX
        region['EndY'] = mouseEndY

        save()


cv2.namedWindow(window_title)
cv2.setMouseCallback(window_title, draw_circle)

while True:
    _, img = cap.read()

    if mouseDown:
        cv2.rectangle(img, (mouseStartX, mouseStartY), (mouseX, mouseY), (0, 0, 255), 1)
    else:
        cv2.rectangle(img, (mouseStartX, mouseStartY), (mouseEndX, mouseEndY), (0, 0, 255), 1)

        color = img[mouseY,mouseX]
        cv2.putText(img, ("BGR: %s %s %s" % (color[0], color[1], color[2])), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200))

    cv2.imshow(window_title, img)

    ky = cv2.waitKey(10) & 0xFF
    if ky == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

    if ky == ord('b'):
        color = img[mouseY, mouseX]
        region['BLUE_B'] = int(color[0])
        region['BLUE_G'] = int(color[1])
        region['BLUE_R'] = int(color[2])
        print(color)
        save()
    if ky == ord('y'):
        color = img[mouseY,mouseX]
        region['YELLOW_B'] = int(color[0])
        region['YELLOW_G'] = int(color[1])
        region['YELLOW_R'] = int(color[2])
        print(color)
        save()
    if ky == ord('r'):
        color = img[mouseY,mouseX]
        region['RED_B'] = int(color[0])
        region['RED_G'] = int(color[1])
        region['RED_R'] = int(color[2])
        print(color)
        save()
