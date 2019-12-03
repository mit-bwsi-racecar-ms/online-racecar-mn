#############################
#### Imports
#############################

# General 
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# ROS 
try:
    import rospy
    from rospy.numpy_msg import numpy_msg
    from sensor_msgs.msg import LaserScan
    from sensor_msgs.msg import Image
    from ackermann_msgs.msg import AckermannDriveStamped
except:
    print('ROS is not installed')

# iPython Display
import PIL.Image
from io import BytesIO
import IPython.display
import time

# Used for HSV select
import threading
try:
    import ipywidgets as widgets
except:
    print('ipywidgets is not installed')

#############################
#### Parameters
#############################

# Video Capture Port
video_port = 2

# Display ID
current_display_id = 1 # keeps track of display id

# Resize dimensions
resize_width = 640
resize_height = 480

#############################
#### Racecar ROS Class
#############################

# Starter code class that handles the fancy stuff. No need to modify this!
cap = None
released = True
class Racecar:
    SCAN_TOPIC = "/scan"
    IMAGE_TOPIC = "/camera"
    DRIVE_TOPIC = "/drive"
    
    def __init__(self):
        self.sub_scan = rospy.Subscriber(self.SCAN_TOPIC, LaserScan, callback=self.scan_callback)
        self.sub_image = rospy.Subscriber(self.IMAGE_TOPIC, Image, callback=self.image_callback)
        self.pub_drive = rospy.Publisher(self.DRIVE_TOPIC, AckermannDriveStamped, queue_size=1)
        self.last_drive = AckermannDriveStamped()
    
    def image_callback(self, msg):
        self.last_image = msg.data
        
    def show_last_image(self):
        im = np.fromstring(self.last_image,dtype=np.uint8).reshape((480,-1,3))[...,::-1]
        return im
        
    def scan_callback(self, msg):
        self.last_scan = msg.ranges
        
    def drive(self, speed, angle):
        msg = AckermannDriveStamped()
        msg.drive.speed = speed
        msg.drive.steering_angle = angle*(0.25/20)  # thresholded at 0.25 radians, approx 20 degrees
        self.last_drive = msg
    
    def stop(self):
        self.drive(0, 0) #self.last_drive.drive.steering_angle)
    
    def look(self):
        return self.last_image
    
    def scan(self):
        return self.last_scan
    
    def run(self, func, limit=10):
        global cap, released
        r = rospy.Rate(60)
        t = rospy.get_time()
        if not released:
            cap.release()
            released = True
        cap = cv2.VideoCapture(video_port)
        resize_cap(cap, resize_height, resize_width)
        released = False
        while rospy.get_time() - t < limit and not rospy.is_shutdown():
            frame = None
            try:
                frame = cap.read()[1]
            except:
                print('Video feed is in use. Please run again or restart kernel.')
            if frame is None:
                print('Video feed is in use. Please run again or restart kernel.')
                break
            else:
                func(cap.read()[1])
                self.pub_drive.publish(self.last_drive)
            r.sleep()
        cap.release()
        released = True
        print("END OF ROSPY RUN")
        self.stop()
        self.pub_drive.publish(self.last_drive)
        time.sleep(0.1)

#############################
#### General Display
#############################

def show_inline(img):
    '''Displays an image inline.'''
    b, g, r = cv2.split(img)
    rgb_img = cv2.merge([r,g,b])
    plt.imshow(rgb_img)
    plt.xticks([]), plt.yticks([])
    plt.show()

def show_frame(frame):
    global display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    f = BytesIO()
    PIL.Image.fromarray(frame).save(f, 'jpeg')
    img = IPython.display.Image(data=f.getvalue())
    display.update(img)

def resize_cap(cap, width, height):
    cap.set(3,width)
    cap.set(4,height)

#############################
#### Identify Cone
#############################

def show_video(func, time_limit, rc):
    global display, current_display_id
    display = IPython.display.display('', display_id=current_display_id)
    current_display_id += 1
    
    rc.run(func, time_limit)

def show_image(func):
    global display, current_display_id
    display = IPython.display.display('', display_id=current_display_id)
    current_display_id += 1
    
    cap = cv2.VideoCapture(video_port)
    resize_cap(cap, resize_width, resize_height)
    frame = func(cap.read()[1])  
    show_frame(frame)
    cap.release()

def show_picture(img):
    global display, current_display_id
    # setup display
    display = IPython.display.display('', display_id=current_display_id)
    current_display_id += 1
    # display image
    f = BytesIO()
    PIL.Image.fromarray(img).save(f, 'jpeg')
    display_image = IPython.display.Image(data=f.getvalue())
    display.update(display_image)
    

#############################
#### HSV Select
#############################

# Mask and display video
def hsv_select_live(limit = 10, fps = 4):
    global current_display_id
    display = IPython.display.display('', display_id=current_display_id)
    current_display_id += 1
    
    # Create sliders
    h = widgets.IntRangeSlider(value=[0, 179], min=0, max=179, description='Hue:', continuous_update=True, layout=widgets.Layout(width='100%'))
    s = widgets.IntRangeSlider(value=[0, 255], min=0, max=255, description='Saturation:', continuous_update=True, layout=widgets.Layout(width='100%'))
    v = widgets.IntRangeSlider(value=[0, 255], min=0, max=255, description='Value:', continuous_update=True, layout=widgets.Layout(width='100%'))
    display.update(h)
    display.update(s)
    display.update(v)
    
    # Live masked video for the thread
    def show_masked_video():
        global cap, released
        if not released:
            cap.release()
            released = True
        cap = cv2.VideoCapture(video_port)
        resize_cap(cap, resize_width, resize_height)
        released = False
        start = time.time()
        while time.time() - start < limit:
            frame = None
            try:
                frame = cap.read()[1]
            except:
                print('Video feed is in use. Please run again or restart kernel.')
            if frame is None:
                print('Video feed is in use. Please run again or restart kernel.')
                break
            else:
                try:
                    hsv_min = (h.value[0], s.value[0], v.value[0])
                    hsv_max = (h.value[1], s.value[1], v.value[1])
                    frame = cv2.flip(frame, 1)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                    mask = cv2.inRange(img_hsv, hsv_min, hsv_max)
                    img_masked = cv2.bitwise_and(frame, frame, mask = mask)
                    f = BytesIO()
                    PIL.Image.fromarray(img_masked).save(f, 'jpeg')
                    img_jpeg = IPython.display.Image(data=f.getvalue())
                    display.update(img_jpeg)
                    time.sleep(1.0 / fps)
                except Exception as e:
                    print(e)
                    break
        cap.release()
        released = True
        print('END OF HSV SELECT')
    
    # Open video on new thread (needed for slider update)
    hsv_thread = threading.Thread(target=show_masked_video)
    hsv_thread.start()

#############################
#### Feature Detection
#############################

def find_object(img, img_q, detected, kp_img, kp_frame, good_matches, query_columns):
    '''
    Draws an outline around a detected objects given matches and keypoints.

    If enough matches are found, extract the locations of matched keypoints in both images.
    The matched keypoints are passed to find the 3x3 perpective transformation matrix.
    Use transformation matrix to transform the corners of img to corresponding points in trainImage.
    Draw matches.
    '''
    dst = []
    if detected:
        src_pts = np.float32([ kp_img[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp_frame[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h,w,ch = img_q.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        
        if M is not None:
            dst = cv2.perspectiveTransform(pts,M)

            dst[:,:,0] += query_columns
        
            x1 = dst[:, :, 0][0]
            y1 = dst[:, :, 1][0]

            x2 = dst[:, :, 0][3]
            y2 = dst[:, :, 1][3]

            center = (x1 + abs(x1 - x2)/2, y1 - abs(y1 - y2)/2)

            return img, dst, center[0], center[1]
        else:
            matchesMask= None
            return img, dst, -1, -1
    else:
        matchesMask = None
        return img, dst, -1, -1   # if center[0] = -1 then didn't find center
    
