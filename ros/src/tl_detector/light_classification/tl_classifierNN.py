
import rospy
import cv2
import numpy as np
from styx_msgs.msg import TrafficLight
from darknet_ros_msgs.msg import BoundingBoxes

class TLClassifierNN(object):
    def __init__(self):
        #TODO load classifier
        self.lastBBox = [[0, 0], [0, 0]]
        self.lastBBoxT = rospy.get_time()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        t = rospy.get_time()
        dt = t - self.lastBBoxT
        if dt < 0.1:
            rospy.loginfo("Got traffic picture past {} seconds, bbox: x {}:{}, y {}:{}".format(dt,
                                    self.lastBBox[0][0], self.lastBBox[0][1], self.lastBBox[1][0], self.lastBBox[1][1]))
        else:
            return TrafficLight.UNKNOWN

        # Check if box is valid
        if self.lastBBox[0][0] == self.lastBBox[0][1] or self.lastBBox[1][0] == self.lastBBox[1][1]:
            return TrafficLight.UNKNOWN

        # Crop image
        bb_image = image[self.lastBBox[1][0]:self.lastBBox[1][1], self.lastBBox[0][0]:self.lastBBox[0][1]]

        height, width, channels = bb_image.shape

        # Partition into red, yellow and green areas of typical vertical traffic light on site
        red_area = bb_image[0:height//3, 0:width]
        yellow_area = bb_image[height//3: 2*height//3, 0:width]
        green_area = bb_image[2*height//3: height, 0:width]

        # Standard coefficients to convert red, yellow and green channels to grayscale
        coef_red = [0.1, 0.1, 0.8]
        coef_yellow = [0.114, 0.587, 0.299]
        coef_green = [0.1, 0.8, 0.1]

        # Apply coefficients
        red_area = cv2.transform(red_area, np.array(coef_red).reshape((1,3)))
        yellow_area = cv2.transform(yellow_area, np.array(coef_yellow).reshape((1,3)))
        green_area = cv2.transform(green_area, np.array(coef_green).reshape((1,3)))

        # Concatenate obtained grayscale images
        bb_image = np.concatenate((red_area,yellow_area,green_area),axis=0)

        # Reevaluate dimensions just in case
        height, width = bb_image.shape

        # Create mask
        mask = np.zeros((height, width), np.uint8)
        width_off = 3
        height_off = 4
        cv2.ellipse(mask, (width//2, 1*height//6), (width//2 - width_off, height//6 - height_off), 0, 0, 360, 1, -1)
        cv2.ellipse(mask, (width//2, 3*height//6), (width//2 - width_off, height//6 - height_off), 0, 0, 360, 1, -1)
        cv2.ellipse(mask, (width//2, 5*height//6), (width//2 - width_off, height//6 - height_off), 0, 0, 360, 1, -1)

        # Apply mask
        bb_image = np.multiply(bb_image, mask)

        # Cut not bright enough pixels
        bb_image = cv2.inRange(bb_image, 200, 255)

        # Partition into red, yellow and green areas
        red_area = bb_image[0:height//3, 0:width]
        yellow_area = bb_image[height//3: 2*height//3, 0:width]
        green_area = bb_image[2*height//3: height, 0:width]

        # Count the number of non-zero pixels in each area
        red_cnt = cv2.countNonZero(red_area)
        yellow_cnt = cv2.countNonZero(yellow_area)
        green_cnt = cv2.countNonZero(green_area)

        # Determine which color had max non-zero pixels
        if red_cnt > yellow_cnt and red_cnt > green_cnt:
            return TrafficLight.RED
        elif yellow_cnt > red_cnt and yellow_cnt > green_cnt:
            return TrafficLight.YELLOW
        # Do not differentiate green and unknown
        return TrafficLight.UNKNOWN

    def bboxes_cb(self, bBoxes):
        for box in bBoxes.bounding_boxes:
            # rospy.loginfo("Class: {}, prob: {}, x: {}:{}, y: {}:{}".format(box.Class, box.probability, box.xmin,
            #                                             box.xmax, box.ymin, box.ymax))
            if box.Class == 'traffic light':
                self.lastBBox = [[box.xmin, box.xmax], [box.ymin, box.ymax]]
                self.lastBBoxT = rospy.get_time()
