#!/usr/bin/env python
import copy
import json
import os
import random

import numpy as np
import rosnode
import rospy
import tf2_ros
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import TransformStamped
from scipy import interpolate, signal
from std_msgs.msg import Bool

move = False 
class ObjectsController:
    def __init__(self):
        self.real_robot = rospy.get_param("real_robot")
        
        self.reference_frame = rospy.get_param("reference_frame")
        
        # Static TF2 Broadcaster
        self.static_tf2_broadcaster = tf2_ros.StaticTransformBroadcaster()
        rospy.sleep(3.0)

    def objects_initialization(self):
        self.n_objects = int(rospy.get_param("n_objects", 1))
        # Initialization of ModelState() messages
        #rospy.sleep(6)

        if not self.real_robot:
            self.objects_model_state = [ModelState() for i in range(self.n_objects)]
            
            # Get objects model names
            for i in range(self.n_objects):
                self.objects_model_state[i].model_name = rospy.get_param("object_" + repr(i) +"_model_name")
                self.objects_model_state[i].reference_frame = self.reference_frame
        # Initialization of Objects tf frames names
        self.objects_tf_frame = [rospy.get_param("object_" + repr(i) +"_frame") for i in range(self.n_objects)]
        
        self.position_list = self._find_random_positioning(self.n_objects)
        self.move_objects2shelf()

    def move_objects2shelf(self):
        # Move objects up in the air 
        for i in range(self.n_objects):

            position = self.position_list[i]
            
            if not self.real_robot:
                
                self.objects_model_state[i].pose.position.x = position[0]
                self.objects_model_state[i].pose.position.y = position[1]
                self.objects_model_state[i].pose.position.z = position[2]
                rospy.wait_for_service('/gazebo/set_model_state')
                #self.set_model_state_pub.publish(self.objects_model_state[i]) 
                try:
                    set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
                    resp = set_state( self.objects_model_state[i] )
                except rospy.ServiceException:
                    print("Service call failed:")

            # Publish tf of objects
            t = TransformStamped()
            t.header.frame_id = self.reference_frame
            t.header.stamp = rospy.Time.now()
            t.child_frame_id = self.objects_tf_frame[i]
            t.transform.translation.x = position[0]
            t.transform.translation.y = position[1]
            t.transform.translation.z = position[2]
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0
            self.static_tf2_broadcaster.sendTransform(t)

    def _find_random_positioning(self, number_of_objects):
        #x_min = -0.375
        #x_max = 0.375
        x_min = -0.34
        x_max = 0.34
        y_min = 1.10
        #y_max = 1.26
        y_max = 1.20
        z1 = 0.03
        z2 = 0.41
        range_x = (x_max - x_min)/3

        positions = []

        for i in range(number_of_objects):
            position = []
            if i < 3:
                x_value = round(random.uniform(x_min + range_x*i + 0.02, x_min + range_x*(i+1) - 0.02), 2)
                y_value = round(random.uniform(y_min, y_max), 2)
                z_value = z2
                position.append(x_value)
                position.append(y_value)
                position.append(z_value)
                positions.append(position)
            else:
                x_value = round(random.uniform(x_min + range_x*(i-3), x_min + range_x*(i-2)), 2)
                y_value = round(random.uniform(y_min, y_max), 2)
                z_value = z1
                position.append(x_value)
                position.append(y_value)
                position.append(z_value)
                positions.append(position)

        return random.sample(positions, len(positions))



if __name__ == '__main__':
    try:
        rospy.init_node('objects_controller')
        rospy.sleep(3.0)
        oc = ObjectsController()
        oc.objects_initialization()
        #oc.objects_state_update_loop()
    except rospy.ROSInterruptException:
        pass
