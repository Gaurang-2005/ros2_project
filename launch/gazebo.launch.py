from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, SetEnvironmentVariable
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    pkg_path = get_package_share_directory('pick_and_drop')
    sdf_path = os.path.join(pkg_path, 'gazebo', 'robot.sdf')

    return LaunchDescription([
        SetEnvironmentVariable(
            name='LIBGL_ALWAYS_SOFTWARE',
            value='1'
        ),
        # Resource path for meshes
        SetEnvironmentVariable(
            name='IGN_GAZEBO_RESOURCE_PATH',
            value=os.path.dirname(pkg_path)
        ),

        # Start Gazebo (Ignition)
        ExecuteProcess(
            cmd=['ign', 'gazebo', 'empty.sdf'],
            output='screen'
        ),

        # Spawn robot
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package='ros_gz_sim',
                    executable='create',
                    arguments=[
                        '-name', 'pick_and_drop_arm',
                        '-file', sdf_path
                    ],
                    output='screen'
                )
            ]
        ),
    ])