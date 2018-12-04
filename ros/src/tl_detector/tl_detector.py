#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from darknet_ros_msgs.msg import BoundingBoxes
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from light_classification.tl_classifierNN import TLClassifierNN
import tf
import cv2
import math
import yaml

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.lights_wp = []
        self.lights_wp_init = False
        self.car_wp_idx = -1

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        sub4 = rospy.Subscriber('/close_waypoint_n', Int32, self.car_waypoint_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()

        if not self.config['is_site']:
            self.light_classifier = TLClassifier()
        else:
            self.light_classifier = TLClassifierNN()
            sub7 = rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, self.light_classifier.bboxes_cb)

        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def car_waypoint_cb(self, wp_idx):
        self.car_wp_idx = wp_idx.data

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()
        rospy.loginfo(light_wp)
        rospy.loginfo(state)

        '''
        Publish upcoming red (and yellow) lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if (state == TrafficLight.RED or state == TrafficLight.YELLOW) else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        min_idx = -1
        min_dist = 10000
        if self.waypoints is None:
            return min_idx

        pos_x = pose.position.x
        pos_y = pose.position.y

        # Check all waypoints to find closest one
        for i, wp in enumerate(self.waypoints.waypoints):
            wp_x = wp.pose.pose.position.x
            wp_y = wp.pose.pose.position.y
            dist = math.sqrt((pos_x - wp_x)**2 + (pos_y - wp_y)**2)
            if dist < min_dist:
                min_dist = dist
                min_idx = i

        return min_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # Get classification
        result = self.light_classifier.get_classification(cv_image)
        rospy.loginfo("STATE: ")
        # rospy.loginfo(light.state)
        rospy.loginfo(result)

        return result

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        state = self.get_light_state(1)
        # Init closest waypoints for lights
        if not self.lights_wp_init:
            if self.lights and self.waypoints:
                # List of positions that correspond to the line to stop in front of for a given intersection
                stop_line_positions = self.config['stop_line_positions']
                for i, light in enumerate(self.lights):
                    # Stop line waypoint index
                    line = stop_line_positions[i]
                    tmp_pose = Pose()
                    tmp_pose.position.x = line[0]
                    tmp_pose.position.y = line[1]
                    temp_wp_idx = self.get_closest_waypoint(tmp_pose)
                    self.lights_wp.append(temp_wp_idx)

                self.lights_wp_init = True

        closest_light = None
        line_wp_idx = None

        car_wp_idx = self.car_wp_idx
        if self.pose and self.lights_wp_init and car_wp_idx >= 0:

            #TODO find the closest visible traffic light (if one exists)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                # Find closest stop line waypoint index
                d = self.lights_wp[i] - car_wp_idx
                if d >= 0 and d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = self.lights_wp[i]

            if closest_light and diff < 200:
                state = self.get_light_state(closest_light)
                return line_wp_idx, state

        #self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
