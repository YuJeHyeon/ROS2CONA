#include "ros/ros.h"
#include <iostream>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/Joy.h>
#include <naoqi_bridge_msgs/JointAnglesWithSpeed.h>
#include "boost/thread/mutex.hpp"
#include "boost/thread/thread.hpp"

ros::Publisher robot_pub_;
ros::Publisher joint_pub_;
naoqi_bridge_msgs::JointAnglesWithSpeed pose;
float linear_vel = 0.0f;
float angular_vel = 0.0f;
float headyaw_ = 0.0f;
float headpitch_ = 0.0f;

float freehead;
double yaw,pitch, roll, speed_;


const static float maxyaw = 0.5;
const static float maxpitch = 0.1;
const static float maxroll = 0.1;
const static float deltayaw = 1;
const static float deltapitch = 1;

// uint8_t joint_relative = 0;

void JoyWingCallback(const sensor_msgs::Joy::ConstPtr &joy)
{
    float axes_hori_left = joy->axes[0];
    float axes_verti_left = joy->axes[1];
    int buttons_left_down = joy->buttons[6];
    int buttons_right_down = joy->buttons[7];
    int buttons_y_down = joy->buttons[0];
    int buttons_b_down = joy->buttons[1];
    int buttons_a_down = joy->buttons[2];
    bool check_moving1 = false;
    bool check_moving2 = false;
    bool check_moving3 = false;
    bool check_moving4 = false;
    bool check_moving5 = false;
    
    if(buttons_left_down)
        check_moving1 = true;
    if(check_moving1)
    {
        linear_vel = axes_verti_left;
        angular_vel = axes_hori_left;
    }
    else
    {
        linear_vel = 0.0f;
        angular_vel = 0.0f;
    }
    if(buttons_right_down)
        check_moving2 = true;
    if(check_moving2)
    {
        yaw = joy->axes[0] * maxyaw;
        pitch = joy->axes[1] * maxpitch;
        pose.joint_names.clear();
        pose.joint_names.push_back("HeadYaw");
        pose.joint_names.push_back("HeadPitch");
        pose.joint_angles.clear();
        pose.joint_angles.push_back(yaw);
        pose.joint_angles.push_back(pitch);
        pose.speed = 0.05;
        pose.relative = 0;
    }
    else
    {
        yaw = 0.0f;
        pitch = 0.0f;
    }
    if(buttons_y_down)
        check_moving3 = true;
    if(check_moving3)
    {
        roll = joy->axes[0] * maxroll;
        pitch = joy->axes[1] * maxpitch;
        pose.joint_names.clear();
        pose.joint_names.push_back("HipRoll");
        pose.joint_names.push_back("HipPitch");
        pose.joint_angles.clear();
        pose.joint_angles.push_back(roll);
        pose.joint_angles.push_back(pitch);
        pose.speed = 0.05;
        pose.relative = 0;
    }
    else
    {
        roll = 0.0f;
        pitch = 0.0f;
    }
    if(buttons_a_down)
        check_moving4 = true;
    if(check_moving4)
    {
        roll = joy->axes[0] * +1 ;
        // pitch = joy->axes[1] * 0.05;
        pose.joint_names.clear();
        // pose.joint_names.push_back("RShoulderPitch");
        pose.joint_names.push_back("RShoulderRoll");
        pose.joint_angles.clear();
        // pose.joint_angles.push_back(pitch);
        pose.joint_angles.push_back(roll);
        pose.speed = 0.05;
        pose.relative = 0;
    }
    else
    {
        pitch = 0.0f;
        roll = 0.0f;
    }
    if(buttons_b_down)
        check_moving5 = true;
    if(check_moving5)
    {
        roll = joy->axes[0] * -1 ;
        // pitch = joy->axes[1] * 1;
        pose.joint_names.clear();
        // pose.joint_names.push_back("LShoulderPitch");
        pose.joint_names.push_back("LShoulderRoll");
        pose.joint_angles.clear();
        // pose.joint_angles.push_back(pitch);
        pose.joint_angles.push_back(roll);
        pose.speed = 0.05;
        pose.relative = 0;
    }
    else
    {
        roll = 0.0f;
        pitch = 0.0f;
    }
    printf("joy->axes[0] : %f joy->axes[1] : %f joy->buttons[6] : %f joy->buttons[7] : %f\n",joy->axes[0],joy->axes[1],joy->buttons[6], joy->buttons[7]);
    
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "joywing_node");
  ros::NodeHandle n, ph_;
  ros::Subscriber joy_sub_;

//   joint_pub_ = n.advertise<naoqi_bridge_msgs::JointAnglesWithSpeed>("/joint_angles", 10);

  robot_pub_ = n.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
  joy_sub_ = n.subscribe("/joy", 10, JoyWingCallback);
  joint_pub_ = ph_.advertise<naoqi_bridge_msgs::JointAnglesWithSpeed>("/joint_angles", 1, true);
  ros::Rate loop_rate(10);
  while(ros::ok())
  {

    geometry_msgs::Twist vel_pub;
    // naoqi_bridge_msgs::JointAnglesWithSpeed joint_pub;

    // joint angle
    joint_pub_.publish(pose);
	//linear
    vel_pub.linear.x = linear_vel / 8;   // +: front, -: rear
    vel_pub.linear.y = 0.0; 		
    vel_pub.linear.z = 0.0;
	//rotation 		
    vel_pub.angular.x = 0.0; 		
    vel_pub.angular.y = 0.0; 		
    vel_pub.angular.z = angular_vel / 8; 	// +: left, -: right
	
    printf("linear_x: %f, angular_z: %f\n", vel_pub.linear.x, vel_pub.angular.z);

    robot_pub_.publish(vel_pub);
    
    ros::spinOnce();
    loop_rate.sleep();
  }  

  return 0;
}
