import math
import os
import subprocess
import tempfile

import numpy as np
import rclpy

from rclpy.node import Node

from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseArray

from sensor_msgs.msg import JointState

from std_msgs.msg import Bool
from std_msgs.msg import Float64MultiArray


class PickAndPlace(Node):

    def __init__(self):

        super().__init__('pick_and_place')

        # =====================================================
        # JOINT STATE STORAGE
        # =====================================================

        self.arm_joint_names = [
            'base_joint',
            'joint1',
            'joint2',
            'joint3',
            'joint4',
        ]

        self.q = np.zeros(5, dtype=float)

        self.have_joint_state = False

        # =====================================================
        # PARAMETERS
        # =====================================================

        self.declare_parameter('spawn_box', True)

        self.declare_parameter('box_name', 'pick_box')

        self.declare_parameter('box_size', 0.45)

        self.declare_parameter('box_x', 0.0)
        self.declare_parameter('box_y', 4.0)
        self.declare_parameter('box_z', 0.225)

        self.declare_parameter('world_name', 'empty')

        self.declare_parameter('open_finger_deg', 0.0)

        self.declare_parameter('closed_finger_deg', 40.0)

        self.declare_parameter('auto_start_delay_sec', 3.0)

        # =====================================================
        # PARAM VALUES
        # =====================================================

        self.box_name = self.get_parameter(
            'box_name'
        ).value

        self.box_size = float(
            self.get_parameter('box_size').value
        )

        self.spawn_box_enabled = bool(
            self.get_parameter('spawn_box').value
        )

        self.world_name = self.get_parameter(
            'world_name'
        ).value

        self.auto_start_delay_sec = float(
            self.get_parameter(
                'auto_start_delay_sec'
            ).value
        )

        self.open_finger = math.radians(
            float(
                self.get_parameter(
                    'open_finger_deg'
                ).value
            )
        )

        self.closed_finger = math.radians(
            float(
                self.get_parameter(
                    'closed_finger_deg'
                ).value
            )
        )

        self.box_position = np.array([
            self.get_parameter('box_x').value,
            self.get_parameter('box_y').value,
            self.get_parameter('box_z').value
        ], dtype=float)

        # =====================================================
        # PUBLISHERS
        # =====================================================

        self.ik_target_pub = self.create_publisher(
            Point,
            '/ik_target',
            10
        )

        self.gripper_pub = self.create_publisher(
            Float64MultiArray,
            '/gripper_angles',
            10
        )

        self.autonomous_pub = self.create_publisher(
            Bool,
            '/autonomous_mode',
            10
        )

        # =====================================================
        # SUBSCRIBERS
        # =====================================================

        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_cb,
            10
        )

        self.create_subscription(
            PoseArray,
            f'/world/{self.world_name}/pose/info',
            self.pose_cb,
            10
        )

        # =====================================================
        # STATE
        # =====================================================

        self.box_pose_valid = False

        self.started = False

        self.finished = False

        self.sequence = []

        self.step_index = 0

        self.current_target = None

        self.current_finger = None

        self.position_tolerance = 0.25

        self.required_stable_cycles = 15

        self.stable_counter = 0

        self.start_time = self.get_clock().now()

        # =====================================================
        # BOX SPAWNING
        # =====================================================

        self.box_spawned = False

        self.spawn_attempts = 0

        self.spawn_retry_period_sec = 2.0

        self.spawn_max_attempts = 20

        self.last_spawn_attempt_time = None

        # =====================================================
        # TIMER
        # =====================================================

        self.timer = self.create_timer(
            0.1,
            self.timer_cb
        )

        self.get_logger().info(
            'Verified pick and place node started'
        )

    # =========================================================
    # JOINT STATES
    # =========================================================

    def joint_state_cb(self, msg):

        name_to_index = {
            name: i
            for i, name in enumerate(msg.name)
        }

        for i, joint in enumerate(self.arm_joint_names):

            if joint in name_to_index:

                self.q[i] = msg.position[
                    name_to_index[joint]
                ]

        self.have_joint_state = True

    # =========================================================
    # BOX TRACKING
    # =========================================================

    def pose_cb(self, msg):

        if not hasattr(msg, 'names'):
            return

        for i, name in enumerate(msg.names):

            if name == self.box_name:

                pose = msg.poses[i]

                self.box_position = np.array([
                    pose.position.x,
                    pose.position.y,
                    pose.position.z
                ], dtype=float)

                self.box_pose_valid = True

                return

    # =========================================================
    # TIMER
    # =========================================================

    def timer_cb(self):

        if self.finished:
            return

        # =========================================
        # SPAWN BOX FIRST
        # =========================================

        if (
            self.spawn_box_enabled
            and not self.box_spawned
        ):

            self.try_spawn_box()

            return

        # =========================================
        # STARTUP DELAY
        # =========================================

        elapsed = (
            self.get_clock().now()
            - self.start_time
        ).nanoseconds / 1e9

        if elapsed < self.auto_start_delay_sec:
            return

        if not self.have_joint_state:
            return

        # =========================================
        # START SEQUENCE
        # =========================================

        if not self.started:

            self.started = True

            self.publish_autonomous_mode(True)

            self.prepare_sequence()

            self.get_logger().info(
                'Starting pick and place sequence'
            )

            self.execute_current_step()

            return

        # =========================================
        # CONTINUOUS TARGET PUBLISHING
        # =========================================

        if self.current_target is not None:

            self.publish_ik_target(
                self.current_target
            )

        if self.current_finger is not None:

            self.publish_gripper(
                self.current_finger
            )

        # =========================================
        # VERIFY CURRENT STEP
        # =========================================

        if self.verify_current_step():

            self.stable_counter += 1

        else:

            self.stable_counter = 0

        # =========================================
        # STEP COMPLETED
        # =========================================

        if (
            self.stable_counter
            >= self.required_stable_cycles
        ):

            self.get_logger().info(
                'Step verified'
            )

            self.stable_counter = 0

            self.step_index += 1

            # =====================================
            # FINISHED
            # =====================================

            if self.step_index >= len(self.sequence):

                self.finished = True

                self.publish_autonomous_mode(False)

                self.get_logger().info(
                    'Pick and place finished'
                )

                return

            self.execute_current_step()

    # =========================================================
    # SPAWN BOX
    # =========================================================

    def try_spawn_box(self):

        import time

        now = time.monotonic()

        if (
            self.last_spawn_attempt_time
            is not None
            and now
            - self.last_spawn_attempt_time
            < self.spawn_retry_period_sec
        ):
            return

        if (
            self.spawn_attempts
            >= self.spawn_max_attempts
        ):

            self.get_logger().error(
                'Failed to spawn box'
            )

            self.finished = True

            return

        self.last_spawn_attempt_time = now

        self.spawn_attempts += 1

        if self.spawn_box():

            self.box_spawned = True

            self.get_logger().info(
                'Box spawned successfully'
            )

    def spawn_box(self):

        box_x = float(self.box_position[0])

        box_y = float(self.box_position[1])

        box_z = float(self.box_position[2])

        sdf = f"""
<?xml version="1.0" ?>
<sdf version="1.9">

  <model name="{self.box_name}">

    <link name="link">

      <inertial>
        <mass>0.2</mass>
      </inertial>

      <collision name="collision">

        <geometry>
          <box>
            <size>
              {self.box_size}
              {self.box_size}
              {self.box_size}
            </size>
          </box>
        </geometry>

      </collision>

      <visual name="visual">

        <geometry>
          <box>
            <size>
              {self.box_size}
              {self.box_size}
              {self.box_size}
            </size>
          </box>
        </geometry>

      </visual>

    </link>

  </model>

</sdf>
"""

        sdf_path = os.path.join(
            tempfile.gettempdir(),
            f'{self.box_name}.sdf'
        )

        with open(sdf_path, 'w') as f:

            f.write(sdf)

        cmd = [
            'ros2',
            'run',
            'ros_gz_sim',
            'create',
            '-world',
            self.world_name,
            '-name',
            self.box_name,
            '-file',
            sdf_path,
            '-x',
            str(box_x),
            '-y',
            str(box_y),
            '-z',
            str(box_z),
        ]

        try:

            result = subprocess.run(
                cmd,
                check=False,
                text=True,
                capture_output=True,
                timeout=10.0
            )

            if result.returncode == 0:

                self.get_logger().info(
                    f'Spawned box at '
                    f'[{box_x:.2f}, '
                    f'{box_y:.2f}, '
                    f'{box_z:.2f}]'
                )

                return True

            else:

                self.get_logger().error(
                    result.stderr
                )

        except Exception as e:

            self.get_logger().error(str(e))

        return False

    # =========================================================
    # SEQUENCE
    # =========================================================

    def prepare_sequence(self):

        box = self.box_position.copy()

        place = np.array([
            -4.0,
            2.0,
            0.5
        ])

        self.sequence = [

            {
                'name': 'home',
                'position': np.array([
                    0.0,
                    0.0,
                    7.5
                ]),
                'finger': self.open_finger
            },

            {
                'name': 'pre_pick',
                'position': box + np.array([
                    0.0,
                    0.0,
                    2.0
                ]),
                'finger': self.open_finger
            },

            {
                'name': 'pick',
                'position': box + np.array([
                    0.0,
                    0.0,
                    0.4
                ]),
                'finger': self.open_finger
            },

            {
                'name': 'close',
                'position': box + np.array([
                    0.0,
                    0.0,
                    0.4
                ]),
                'finger': self.closed_finger
            },

            {
                'name': 'lift',
                'position': box + np.array([
                    0.0,
                    0.0,
                    2.0
                ]),
                'finger': self.closed_finger
            },

            {
                'name': 'pre_place',
                'position': place + np.array([
                    0.0,
                    0.0,
                    2.0
                ]),
                'finger': self.closed_finger
            },

            {
                'name': 'place',
                'position': place + np.array([
                    0.0,
                    0.0,
                    0.4
                ]),
                'finger': self.closed_finger
            },

            {
                'name': 'open',
                'position': place + np.array([
                    0.0,
                    0.0,
                    0.4
                ]),
                'finger': self.open_finger
            },

            {
                'name': 'retreat',
                'position': place + np.array([
                    0.0,
                    0.0,
                    2.0
                ]),
                'finger': self.open_finger
            },

            {
                'name': 'home',
                'position': np.array([
                    0.0,
                    0.0,
                    7.5
                ]),
                'finger': self.open_finger
            },
        ]

    # =========================================================
    # EXECUTE STEP
    # =========================================================

    def execute_current_step(self):

        step = self.sequence[self.step_index]

        name = step['name']

        position = step['position']

        finger = step['finger']

        self.current_target = position.copy()

        self.current_finger = finger

        self.publish_ik_target(
            self.current_target
        )

        self.publish_gripper(
            self.current_finger
        )

        self.get_logger().info(
            f'START STEP: {name} | '
            f'Position: '
            f'[{position[0]:.2f}, '
            f'{position[1]:.2f}, '
            f'{position[2]:.2f}]'
        )

    # =========================================================
    # VERIFY STEP
    # =========================================================

    def verify_current_step(self):

        if self.current_target is None:

            return False

        ee = self.forward_kinematics(
            self.q
        )

        error = np.linalg.norm(
            self.current_target - ee
        )

        print(
            f'Verification Error: '
            f'{error:.4f}',
            flush=True
        )

        return (
            error
            < self.position_tolerance
        )

    # =========================================================
    # IK TARGET
    # =========================================================

    def publish_ik_target(self, position):

        msg = Point()

        msg.x = float(position[0])

        msg.y = float(position[1])

        msg.z = float(position[2])

        self.ik_target_pub.publish(msg)

    # =========================================================
    # GRIPPER
    # =========================================================

    def publish_gripper(self, finger_angle):

        msg = Float64MultiArray()

        deg = math.degrees(finger_angle)

        msg.data = [deg, deg]

        self.gripper_pub.publish(msg)

    # =========================================================
    # AUTONOMOUS MODE
    # =========================================================

    def publish_autonomous_mode(self, active):

        msg = Bool()

        msg.data = bool(active)

        self.autonomous_pub.publish(msg)

    # =========================================================
    # FK
    # =========================================================

    def forward_kinematics(self, q):

        T = np.eye(4)

        T = T @ self.trans(0, 0, 0.1)

        T = T @ self.rotz(q[0])

        T = T @ self.roty(q[1])

        T = T @ self.trans(
            0,
            0,
            1.67744
        )

        T = T @ self.roty(q[2])

        T = T @ self.trans(
            0,
            0,
            2.92735
        )

        T = T @ self.roty(q[3])

        T = T @ self.trans(
            0,
            0,
            2.9244
        )

        T = T @ self.roty(q[4])

        T = T @ self.trans(
            0,
            0,
            2.12653
        )

        return T[:3, 3]

    # =========================================================
    # HELPERS
    # =========================================================

    def trans(self, x, y, z):

        T = np.eye(4)

        T[:3, 3] = [x, y, z]

        return T

    def roty(self, a):

        c = math.cos(a)

        s = math.sin(a)

        return np.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ])

    def rotz(self, a):

        c = math.cos(a)

        s = math.sin(a)

        return np.array([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])


# =============================================================
# MAIN
# =============================================================

def main():

    rclpy.init()

    node = PickAndPlace()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':

    main()