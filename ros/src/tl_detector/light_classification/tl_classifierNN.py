
import rospy
from styx_msgs.msg import TrafficLight
from darknet_ros_msgs.msg import BoundingBoxes
class TLClassifierNN(object):
    def __init__(self):
        #TODO load classifier
        self.lastBBox = [[0, 0], [0, 0]]
        self.lastBBoxT = rospy.get_time()
        pass

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


        # Do not differentiate green and unknown
        return TrafficLight.UNKNOWN

    def bboxes_cb(self, bBoxes):
        for box in bBoxes.bounding_boxes:
            # rospy.loginfo("Class: {}, prob: {}, x: {}:{}, y: {}:{}".format(box.Class, box.probability, box.xmin,
            #                                             box.xmax, box.ymin, box.ymax))
            if box.Class == 'traffic light':
                self.lastBBox = [[box.xmin, box.xmax], [box.ymin, box.ymax]]
                self.lastBBoxT = rospy.get_time()
