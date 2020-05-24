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
    import rclpy
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
import ipywidgets as widgets
  
import pyrealsense2 as rs
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy, QoSProfile 

# Start ROS node
rclpy.init(args=None)
rc = rclpy.create_node('Racecar')
print('ROS node started successfully')

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
#### ROS Driving
#############################

def withDriving(callback):
    qos_profile = QoSProfile(depth=1)
    qos_profile.history = QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST
    qos_profile.reliability = QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT
    qos_profile.durability = QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE
    publisher = rc.create_publisher(AckermannDriveStamped, '/drive', qos_profile)
    
    def drive(speed, angle):
        msg = AckermannDriveStamped()
        msg.drive.speed = speed
        msg.drive.steering_angle = angle
        publisher.publish(msg)
        
    callback(drive)
    # Send an empty message, whch will stop the car
    publisher.publish(AckermannDriveStamped())

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
#### Frame Processing
#############################

def withRealSenseImages(frameProcessor, options, limit = None):
    color = options.get('color', 'rgb')
    depth = options.get('depth', False)
    # RealSense maximum resolution
    width = options.get('width', 1920)
    height = options.get('height', 1080)
    frame_rate = options.get('frame-rate', 30)
    
    if not (color or depth):
        print('Specify color or depth')
        return
    
    pipeline = rs.pipeline()
    config = rs.config()
    
    if color:
        if color == 'bgr':
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, frame_rate)
        else:
            config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, frame_rate)
    if depth:
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, frame_rate)

    # Start streaming
    pipeline.start(config)

    try:
        start = time.time()
        while limit == None or time.time() - start < limit:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            if color:
                if depth:
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    if color_frame and depth_frame:
                        frameProcessor(np.asanyarray(color_frame.get_data()),
                                       np.asanyarray(depth_frame.get_data()))
                else:
                    frameProcessor(np.asanyarray(frames.get_color_frame().get_data()))
            elif depth:
                frameProcessor(np.asanyarray(frames.get_depth_frame().get_data()))
    finally:
        # Stop streaming
        pipeline.stop()
        
def show_video(func, time_limit = 10, use_both_frames = False, show_video = True):

    global current_display_id
    display = IPython.display.display('', display_id=current_display_id)
    current_display_id += 1
    
    def display_frame(color_image):
        processed_img = func(color_image)
        if show_video:
            f = BytesIO()
            PIL.Image.fromarray(processed_img).save(f, 'jpeg')
            img = IPython.display.Image(data=f.getvalue())
            display.update(img)
            time.sleep(0.2)
    
    def display_both_frames(color_image, depth_image):
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        processed_img = func(color_image, depth_colormap)

        if show_video:
            f = BytesIO()
            PIL.Image.fromarray(processed_img).save(f, 'jpeg')
            img = IPython.display.Image(data=f.getvalue())
            display.update(img)
            time.sleep(0.2)
            
    if use_both_frames:
        withRealSenseImages(display_both_frames, {'width': 640, 'height': 480, 'depth': use_both_frames}, time_limit)
    else:
        withRealSenseImages(display_frame, {'width': 640, 'height': 480, 'depth': use_both_frames}, time_limit)

def show_videox(func, time_limit = 10, use_both_frames = False, show_video = True):

    canvas = Canvas(width=640, height=480)

    def display_frames(color_image, depth_image):
        if use_both_frames:
            processed_img = func(color_image, depth_image)
        else:
            processed_img = func(color_image)

        if show_video:
            canvas.put_image_data(processed_image, 0, 0)
        
    withRealSenseImages(display_frames, {'width': 640, 'height': 480, 'depth': True}, time_limit)

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
def hsv_select_live(limit = 10, fps = 5):
    
    global current_display_id
    display = IPython.display.display('', display_id=current_display_id)
    current_display_id += 1

    # Create sliders
    h = widgets.IntRangeSlider(value=[0, 179], min=0, max=179,
                               description='Hue:', continuous_update=True,
                               layout=widgets.Layout(width='100%'))
    s = widgets.IntRangeSlider(value=[0, 255], min=0, max=255,
                               description='Saturation:', continuous_update=True,
                               layout=widgets.Layout(width='100%'))
    v = widgets.IntRangeSlider(value=[0, 255], min=0, max=255,
                               description='Value:', continuous_update=True,
                               layout=widgets.Layout(width='100%'))
    display.update(h)
    display.update(s)
    display.update(v)

    def show_masked_video():  
        def processFrame(color_image = None):
            hsv_min = (h.value[0], s.value[0], v.value[0])
            hsv_max = (h.value[1], s.value[1], v.value[1])
            img_hsv = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(img_hsv, hsv_min, hsv_max)
            img_masked = cv2.bitwise_and(color_image, color_image, mask = mask)

            f = BytesIO()
            PIL.Image.fromarray(img_masked).save(f, 'jpeg')
            img = IPython.display.Image(data=f.getvalue())
            display.update(img)
            time.sleep(1.0 / fps)
            
        withRealSenseImages(processFrame, {'width': 640, 'height': 480, 'frame-rate': 30}, limit)

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
    
