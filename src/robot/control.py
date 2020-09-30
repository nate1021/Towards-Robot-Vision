import time
import serial
import threading
import os

MAX_MOTOR_SPEED = 17500.0  # (POSITIVE OR NEGATIVE)
MIN_MOTOR_SPEED = 8000.0  # (POSITIVE OR NEGATIVE)
MAX_MOTOR_TIME = 3100
MIN_MOTOR_TIME = 300
MOTOR_TIME = 400

comi = 0
ser = None

watchdog_running = False
watchdog_heartbeat = 0
watchdog_resets = 0


def millis():
    return time.time() * 1000.0


def connect(comi):
    global ser
    try:
        port_path = '/dev/rfcomm%d' % comi
        ser = serial.Serial(
            port=port_path,
            baudrate=115200,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS
        )
        ser.isOpen()
        print('Connected on: %s' % port_path)
        return True
    except serial.serialutil.SerialException:
        return False


def watchdog_loop():
    global watchdog_heartbeat, ser, comi, watchdog_resets
    while True:
        if watchdog_heartbeat != 0 and watchdog_heartbeat < millis():
            # times up
            print('Watch dog alert! Restarting bluetooth service...')
            ser = None
            comi += 1

            print('bluetooth service restart...')
            os.popen('sudo -S ' + 'service bluetooth restart', 'w').write('cirl\n')
            time.sleep(5)
            print('bluetooth pair...')
            p = os.popen('bt-device -c 00:07:80:96:99:F6', 'w')
            time.sleep(2)
            p.write('0000\n')
            time.sleep(2)

            def connect_com():
                print('bluetooth start comm connection...')
                os.popen('sudo -S rfcomm connect rfcomm%d 00:07:80:96:99:F6' % comi, 'w').write('cirl\n')
                time.sleep(3)

            connect_com()
            while not connect(comi):
                connect_com()
            watchdog_heartbeat = 0
            watchdog_resets += 1

        time.sleep(0.05)


def open_port():
    global ser, watchdog_running, comi
    # configure the serial connections (the parameters differs on the device you are connecting to)
    if not watchdog_running:
        watchdog_running = True
        p1 = threading.Thread(target=watchdog_loop, args=())
        p1.start()

    comi = 0
    while comi < 40:
        if connect(comi):
            break
        else:
            comi += 1

    if ser is None:
        print('ERROR: Failed to connect')
        # exit(-1)
    else:
        print('Serial port opened.')


def set_speed(left, right):
    global watchdog_heartbeat

    command = 'D,l' + str(left) + ',l' + str(right) + '\n'
    watchdog_heartbeat = millis() + 1500
    while ser is None:
        time.sleep(0.2)
    try:
        ser.write(command.encode())
    except serial.serialutil.SerialException:
        print('Command waiting for connection...')
        while watchdog_heartbeat != 0:
            time.sleep(0.2)
        print('Command canceled. Program continuing...')

    watchdog_heartbeat = 0


def set_scaled_speed(left, right):
    cLeft = left * 10000
    if cLeft < 0:
        cLeft -= 3000
    elif cLeft > 0:
        cLeft += 3000

    cRight = right * 10000
    if cRight < 0:
        cRight -= 3000
    elif cRight > 0:
        cRight += 3000

    cRight = int(cRight)
    cLeft = int(cLeft)

    set_speed(cLeft, cRight)


def set_scaled_speed_classic(left, right):
    cLeft = left * 5000
    if cLeft < 0:
        cLeft -= 8000
    elif cLeft > 0:
        cLeft += 8000

    cRight = right * 5000
    if cRight < 0:
        cRight -= 8000
    elif cRight > 0:
        cRight += 8000

    cRight = int(cRight)
    cLeft = int(cLeft)

    set_speed(cLeft, cRight)


def close_port():
    if ser is not None:
        set_speed(0, 0)
        ser.write('M'.encode())
        ser.close()
        print('Serial port closed. Remember to turn off the robot.')


if __name__ == '__main__':
    open_port()
    try:
        AMOUNT = 13000
        while True:
            set_speed(AMOUNT, -AMOUNT)
            print('Turning one way.')
            time.sleep(0.5)
            set_speed(-AMOUNT, AMOUNT)
            print('Turning other way.')
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    close_port()

