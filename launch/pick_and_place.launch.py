"""
launch/pick_and_place.launch.py
================================
Starts everything needed for the pick-and-place demo:
  1. Ignition Gazebo (empty world)
  2. Spawn the pick_and_drop arm
  3. ros_gz_bridge  (all 7 joint topics + box pose)
  4. arm_controller node (control.py)

Usage:
  ros2 launch pick_and_drop pick_and_place.launch.py

With custom drop position:
  ros2 launch pick_and_drop pick_and_place.launch.py drop_x:=2.0 drop_y:=1.5 drop_z:=0.5
"""

from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess, TimerAction,
    SetEnvironmentVariable, DeclareLaunchArgument
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    pkg_path  = get_package_share_directory('pick_and_drop')
    world_path = os.path.join(pkg_path, 'worlds', 'pick_and_drop.sdf')
    sdf_path  = os.path.join(pkg_path, 'gazebo', 'robot.sdf')
    bridge_cfg = os.path.join(pkg_path, 'config', 'bridge.yaml')

    # ── Launch arguments (override drop position on CLI) ──────────────────────
    drop_x_arg = DeclareLaunchArgument('drop_x', default_value='0.0',
                                        description='Drop zone X (metres)')
    drop_y_arg = DeclareLaunchArgument('drop_y', default_value='4.0',
                                        description='Drop zone Y (metres)')
    drop_z_arg = DeclareLaunchArgument('drop_z', default_value='0.5',
                                        description='Drop zone Z (metres)')

    # ── 1. Ignition Gazebo ─────────────────────────────────────────────────────
    gazebo = ExecuteProcess(
        cmd=['ign', 'gazebo', world_path, '-r'],   # -r = run immediately
        output='screen',
        additional_env={
            'LIBGL_ALWAYS_SOFTWARE': '1',
            'IGN_GAZEBO_RESOURCE_PATH': os.path.dirname(pkg_path),
        }
    )

    # ── 2. Spawn robot (after 3s for Gazebo to initialise) ────────────────────
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

    # ── 3. ros_gz_bridge (after 4s so robot plugins are loaded) ───────────────
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

    # ── 4. control.py (after 6s so bridge is up) ──────────────────────────────
    # controller = TimerAction(
    #     period=6.0,
    #     actions=[
    #         Node(
    #             package='pick_and_drop',
    #             executable='control',
    #             name='arm_controller',
    #             output='screen',
    #             parameters=[{
    #                 'drop_x': LaunchConfiguration('drop_x'),
    #                 'drop_y': LaunchConfiguration('drop_y'),
    #                 'drop_z': LaunchConfiguration('drop_z'),
    #             }]
    #         )
    #     ]
    # )

    return LaunchDescription([
        drop_x_arg,
        drop_y_arg,
        drop_z_arg,
        gazebo,
        spawn_robot,
        bridge,
    ])