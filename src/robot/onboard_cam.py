import cv2
frame = None


def show_frame():
    cv2.imshow('Onboard Camera', frame)
    cv2.waitKey(100)


def camera_loop():
    global frame

    ip = 110
    capture = None
    while capture is None or not capture.isOpened():
        str_ip = 'http://192.168.8.%i:8080/video' % ip
        print('Connecting to device at: %s' % str_ip)
        capture = cv2.VideoCapture(str_ip)
        ip += 1

    print("Device captured correctly.", capture)

    while True:
        ret, frame = capture.read()


if __name__ == "__main__":
    import threading

    p2 = threading.Thread(target=camera_loop, args=())
    p2.start()

    while True:
        if frame is not None:
            cv2.imshow('Onboard Camera', frame)

            if cv2.waitKey(100) == 0x1b:
                print('ESC pressed. Exiting ...')
                break
