import cv2

# source data & video
cars_data = 'cars.xml'
video = 'cars.mp4'
# video = 'input your file address'

'''
detects cars in each frame & draws a yellow rect around them
takes the file address as an argument 
'''
def detectCars(file):
    # learning from xml file & reading the video file
    cascade = cv2.CascadeClassifier(cars_data)
    vc = cv2.VideoCapture(file)

    # if video is read
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    # until frames are grabbed from the video
    while rval:
        rval, frame = vc.read()

        # Resizing & gray-scaling the frame
        frame = cv2.resize(frame, (1000,  500))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Haar detection
        cars = cascade.detectMultiScale(gray, 1.3, 3)

        # drawing rects for each car detected
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        # displaying result
        cv2.imshow("Result", frame)

        # exit key
        if cv2.waitKey(33) == ord('x'):
            break

    # closing video file
    vc.release()

if __name__ == '__main__':
    detectCars(video)