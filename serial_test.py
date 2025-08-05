import serial

ser = serial.Serial('COM5', baudrate=921600, timeout=1)

while True:
    try:
        line = ser.readline().decode('utf-8', errors='ignore')
        if line:
            print(line.strip())
    except KeyboardInterrupt:
        print("Exit by Ctrl+C")
        break
