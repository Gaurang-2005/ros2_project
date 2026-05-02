from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    pkg_path  = get_package_share_directory('pick_and_drop')
    world_path = os.path.join(pkg_path, 'worlds', 'pick_and_drop.sdf')
    sdf_path  = os.path.join(pkg_path, 'gazebo', 'robot.sdf')
    bridge_cfg = os.path.join(pkg_path, 'config', 'bridge.yaml')

    # 1. Start Gazebo
    gazebo = ExecuteProcess(
        cmd=['ign', 'gazebo', world_path, '-r'],
        output='screen',
        additional_env={
            'LIBGL_ALWAYS_SOFTWARE': '1',
            'IGN_GAZEBO_RESOURCE_PATH': os.path.dirname(pkg_path),
        }
    )

    # 2. Spawn robot
    spawn_robot = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='ros_gz_sim',
                executable='create',
                arguments=[
                    '-name', 'pick_and_drop_arm',
                    '-file', sdf_path,
                ],
                output='screen'
            )
        ]
    )

    # 3. Bridge (for torque topics)
    bridge = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='ros_gz_bridge',
                executable='parameter_bridge',
                name='gz_bridge',
                parameters=[{'config_file': bridge_cfg}],
                output='screen',
            )
        ]
    )

    # 4. Start IK/home node after bridge is available


    return LaunchDescription([
        gazebo,
        spawn_robot,
        bridge,
    ])
