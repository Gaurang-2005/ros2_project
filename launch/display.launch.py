from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    pkg_path = get_package_share_directory('pick_and_drop')

    urdf_path = os.path.join(pkg_path, 'urdf', 'robot.urdf')
    rviz_config_path = os.path.join(pkg_path, 'config', 'pick_and_drop.rviz')
    controller_yaml = os.path.join(pkg_path, 'config', 'rvizcontrollers.yaml')

    with open(urdf_path, 'r') as f:
        robot_desc = f.read()

    return LaunchDescription([

        Node(
            package='controller_manager',
            executable='ros2_control_node',
            parameters=[
                {'robot_description': robot_desc},
                controller_yaml
            ],
            output='screen'
        ),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_desc}],
            output='screen'
        ),

        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster'],
            output='screen'
        ),

        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['arm_controller'],
            output='screen'
        ),

        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package='rviz2',
                    executable='rviz2',
                    arguments=['-d', rviz_config_path],
                    output='screen'
                )
            ]
        ),
        Node(
            package='rqt_gui',
            executable='rqt_gui',
            arguments=['--standalone', 'rqt_joint_trajectory_controller'],
            output='screen'
        ),    
    ])