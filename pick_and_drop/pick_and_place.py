import math
import os
import subprocess
import tempfile
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseArray

class PickAndPlace(Node):
    def __init__(self):
        super().__init__('pick_and_place')

        self.arm_joint_names = [
            'base_joint',
            'joint1',
            'joint2',
            'joint3',
            'joint4',
        ]
        self.full_joint_names = [
            'base_joint',
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'finger1_joint',
            'finger2_joint',
        ]

        self.joint_limits = np.array([
            [-math.pi, math.pi],
            [-1.57, 1.57],
            [-1.57, 1.57],
            [-1.57, 1.57],
            [-1.57, 1.57],
        ], dtype=float)

        self.q = np.zeros(5, dtype=float)
        self.fingers = np.zeros(2, dtype=float)
        self.have_joint_state = False

        # ---------------- MANUAL TUNING SECTION ----------------
        # Change these ROS parameters from the command line while testing.
        # All positions are end-effector XYZ targets in the world frame.
        self.declare_parameter('spawn_box', True)
        self.declare_parameter('box_name', 'pick_box')
        self.declare_parameter('box_size', 0.45)
        self.declare_parameter('box_x', 0.0)
        self.declare_parameter('box_y', 6.1)
        self.declare_parameter('box_z', 0.225)

        self.declare_parameter('home_base_deg', 0.0)
        self.declare_parameter('home_joint1_deg', -50.0)
        self.declare_parameter('home_joint2_deg', 35.0)
        self.declare_parameter('home_joint3_deg', -75.0)
        self.declare_parameter('home_joint4_deg', -90.0)

        self.declare_parameter('pre_pick_x', 0.0)
        self.declare_parameter('pre_pick_y', 6.1)
        self.declare_parameter('pre_pick_z', 2.0)
        self.declare_parameter('pick_x', 0.0)
        self.declare_parameter('pick_y', 6.1)
        self.declare_parameter('pick_z', 1.0)
        self.declare_parameter('lift_x', 0.0)
        self.declare_parameter('lift_y', 6.1)
        self.declare_parameter('lift_z', 3.0)
        self.declare_parameter('pre_place_x', -6.1)
        self.declare_parameter('pre_place_y', 0.0)
        self.declare_parameter('pre_place_z', 3.0)
        self.declare_parameter('place_x', -6.1)
        self.declare_parameter('place_y', 0.0)
        self.declare_parameter('place_z', 1.0)
        self.declare_parameter('retreat_x', -6.1)
        self.declare_parameter('retreat_y', 0.0)
        self.declare_parameter('retreat_z', 3.0)

        self.declare_parameter('open_finger_deg', 0.0)
        self.declare_parameter('closed_finger_deg', 40.0)
        # -------------------------------------------------------

        self.declare_parameter('world_name', 'empty')
        self.declare_parameter('spawn_retry_period_sec', 2.0)
        self.declare_parameter('spawn_max_attempts', 20)
        self.declare_parameter('move_timeout_sec', 0.0)
        self.declare_parameter('position_tolerance', 0.15)
        self.declare_parameter('wrist_tolerance_deg', 8.0)
        self.declare_parameter('joint_tolerance_deg', 8.0)
        self.declare_parameter('dwell_sec', 1.0)
        self.declare_parameter('auto_start_delay_sec', 3.0)

        self.box_name = self.get_parameter('box_name').value
        self.box_size = float(self.get_parameter('box_size').value)
        self.spawn_box_enabled = bool(self.get_parameter('spawn_box').value)
        self.box_position = self.param_vec('box')
        self.pre_pick_position = self.param_vec('pre_pick')
        self.pick_position = self.param_vec('pick')
        self.lift_position = self.param_vec('lift')
        self.pre_place_position = self.param_vec('pre_place')
        self.place_position = self.param_vec('place')
        self.retreat_position = self.param_vec('retreat')
        self.open_finger = math.radians(float(self.get_parameter('open_finger_deg').value))
        self.closed_finger = math.radians(float(self.get_parameter('closed_finger_deg').value))
        self.world_name = self.get_parameter('world_name').value
        self.spawn_retry_period_sec = float(self.get_parameter('spawn_retry_period_sec').value)
        self.spawn_max_attempts = int(self.get_parameter('spawn_max_attempts').value)
        self.move_timeout = float(self.get_parameter('move_timeout_sec').value)
        self.position_tolerance = float(self.get_parameter('position_tolerance').value)
        self.wrist_tolerance = math.radians(
            float(self.get_parameter('wrist_tolerance_deg').value)
        )
        self.joint_tolerance = math.radians(
            float(self.get_parameter('joint_tolerance_deg').value)
        )
        self.dwell_sec = float(self.get_parameter('dwell_sec').value)
        self.auto_start_delay_sec = float(self.get_parameter('auto_start_delay_sec').value)

        self.home_q = np.radians(np.array([
            self.get_parameter('home_base_deg').value,
            self.get_parameter('home_joint1_deg').value,
            self.get_parameter('home_joint2_deg').value,
            self.get_parameter('home_joint3_deg').value,
            self.get_parameter('home_joint4_deg').value,
        ], dtype=float))
        self.ik_seed_templates = np.radians(np.array([
            [-50.0, 35.0, -75.0, -90.0],
            [-58.0, -31.0, -72.0, -19.0],
            [-65.0, -22.0, -49.0, -44.0],
            [-30.0, -80.0, -35.0, -35.0],
            [-70.0, 55.0, -75.0, -90.0],
        ], dtype=float))
        self.wrist_down_pitch = math.radians(180.0)
        self.orientation_weight = 3.0
        self.damping = 0.2
        self.max_dq_step = 0.04
        self.ik_iterations = 120

        self.target_pub = self.create_publisher(Float64MultiArray, '/target_angles_deg', 10)
        self.autonomous_pub = self.create_publisher(Bool, '/autonomous_mode', 10)
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)

        self.started = False
        self.finished = False
        self.box_spawned = False
        self.spawn_attempts = 0
        self.last_spawn_attempt_time = None
        self.sequence = []
        self.step_index = -1
        self.active_step = None
        self.step_start_time = None
        self.last_progress_log = 0.0
        self.settle_until = None
        self.start_time = self.get_clock().now()
        self.timer = self.create_timer(0.1, self.timer_cb)

        self.get_logger().info(
            'Pick-and-place node ready. Box at %s. Pick=%s Place=%s.'
            % (
                self._fmt_vec(self.box_position),
                self._fmt_vec(self.pick_position),
                self._fmt_vec(self.place_position),
            )
        )
        self.create_subscription(
            PoseArray,
            f'/world/{self.world_name}/pose/info',
            self.pose_cb,
            10
        )

        self.box_pose_valid = False    

    def pose_cb(self, msg):
        if not hasattr(msg, "names"):
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


    def joint_state_cb(self, msg):
        name_to_index = {name: index for index, name in enumerate(msg.name)}

        for i, joint_name in enumerate(self.arm_joint_names):
            if joint_name in name_to_index:
                self.q[i] = msg.position[name_to_index[joint_name]]

        for i, joint_name in enumerate(self.full_joint_names[5:]):
            if joint_name in name_to_index:
                self.fingers[i] = msg.position[name_to_index[joint_name]]

        self.have_joint_state = True

    def param_vec(self, prefix):
        return np.array([
            self.get_parameter(f'{prefix}_x').value,
            self.get_parameter(f'{prefix}_y').value,
            self.get_parameter(f'{prefix}_z').value,
        ], dtype=float)

    def timer_cb(self):
        if self.finished:
            return

        if self.started:
            self.publish_autonomous_mode(True)
            self.process_active_step()
            return

        if self.spawn_box_enabled and not self.box_spawned:
            self.try_spawn_box()
            return

        elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
        if elapsed < self.auto_start_delay_sec or not self.have_joint_state:
            return

        self.started = True
        self.publish_autonomous_mode(True)
        self.prepare_sequence()
        self.get_logger().info('Starting automatic pick-and-place sequence.')
        self.advance_step()

    def publish_autonomous_mode(self, active):
        msg = Bool()
        msg.data = bool(active)
        self.autonomous_pub.publish(msg)

    def try_spawn_box(self):
        now = time.monotonic()
        if (
            self.last_spawn_attempt_time is not None
            and now - self.last_spawn_attempt_time < self.spawn_retry_period_sec
        ):
            return

        if self.spawn_attempts >= self.spawn_max_attempts:
            self.get_logger().error(
                'Box did not spawn after %d attempts. Check that Gazebo is running and world_name=%s.'
                % (self.spawn_attempts, self.world_name)
            )
            self.finished = True
            return

        self.last_spawn_attempt_time = now
        self.spawn_attempts += 1
        if self.spawn_box():
            self.box_spawned = True
            self.start_time = self.get_clock().now()

    def spawn_box(self):
        box_x = float(self.box_position[0])
        box_y = float(self.box_position[1])
        box_z = float(self.box_position[2])

        sdf = f"""<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="{self.box_name}">
    <pose>0 0 0 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>0.2</mass>
        <inertia>
          <ixx>0.003375</ixx>
          <iyy>0.003375</iyy>
          <izz>0.003375</izz>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <box>
            <size>{self.box_size} {self.box_size} {self.box_size}</size>
          </box>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>2.0</mu>
              <mu2>2.0</mu2>
            </ode>
          </friction>
        </surface>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>{self.box_size} {self.box_size} {self.box_size}</size>
          </box>
        </geometry>
        <material>
          <ambient>0.8 0.15 0.05 1</ambient>
          <diffuse>0.9 0.2 0.05 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
        sdf_path = os.path.join(tempfile.gettempdir(), f'{self.box_name}.sdf')
        with open(sdf_path, 'w', encoding='utf-8') as file:
            file.write(sdf)

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
            '-allow_renaming',
            'true',
        ]
        try:
            self.get_logger().info(
                'Spawning box attempt %d/%d in world %s at %s'
                % (
                    self.spawn_attempts,
                    self.spawn_max_attempts,
                    self.world_name,
                    self._fmt_vec(self.box_position),
                )
            )
            result = subprocess.run(
                cmd,
                check=False,
                text=True,
                capture_output=True,
                timeout=10.0,
            )
            if result.returncode == 0:
                self.get_logger().info(
                    'Spawned %s in world %s at [%.3f, %.3f, %.3f]'
                    % (self.box_name, self.world_name, box_x, box_y, box_z)
                )
                return True
            else:
                self.get_logger().error(
                    'Box spawn failed with return code %d: %s %s'
                    % (result.returncode, result.stdout.strip(), result.stderr.strip())
                )
        except OSError as exc:
            self.get_logger().warn(f'Could not spawn box with ros_gz_sim create: {exc}')
        except subprocess.TimeoutExpired:
            self.get_logger().error('Box spawn command timed out before Gazebo responded.')

        return False

    def prepare_sequence(self):
        self.sequence = [
            {'name': 'home_open', 'kind': 'joints', 'q': self.home_q, 'position': self.FK(self.home_q), 'finger': self.open_finger},
            {'name': 'pre_pick_open', 'kind': 'pose', 'position': self.pre_pick_position, 'finger': self.open_finger},
            {'name': 'pick_open', 'kind': 'pose', 'position': self.pick_position, 'finger': self.open_finger},
            {'name': 'pick_closed', 'kind': 'pose', 'position': self.pick_position, 'finger': self.closed_finger, 'settle': True},
            {'name': 'lift_closed', 'kind': 'pose', 'position': self.lift_position, 'finger': self.closed_finger},
            {'name': 'pre_place_closed', 'kind': 'pose', 'position': self.pre_place_position, 'finger': self.closed_finger},
            {'name': 'place_closed', 'kind': 'pose', 'position': self.place_position, 'finger': self.closed_finger},
            {'name': 'place_open', 'kind': 'pose', 'position': self.place_position, 'finger': self.open_finger, 'settle': True},
            {'name': 'retreat_open', 'kind': 'pose', 'position': self.retreat_position, 'finger': self.open_finger},
            {'name': 'home_open', 'kind': 'joints', 'q': self.home_q, 'position': self.FK(self.home_q), 'finger': self.open_finger},
        ]

    def advance_step(self):
        self.step_index += 1
        self.active_step = None
        self.settle_until = None

        if self.step_index >= len(self.sequence):
            self.finished = True
            self.publish_autonomous_mode(False)
            self.get_logger().info('Pick-and-place sequence complete. Teleop IK may resume.')
            return

        step = self.sequence[self.step_index]

        # -------- SAFE APPROACH FIX --------

        if not hasattr(self, "locked_pick_position"):
            self.locked_pick_position = None

        step_name = step['name']

        if step['kind'] == 'pose':

            # ---- STEP 1: GO ABOVE BOX (safe height) ----
            if step_name == 'pre_pick_open':
                if self.box_pose_valid:
                    position = self.box_position + np.array([0.0, 0.0, 2.0])
                else:
                    position = step['position']

            # ---- STEP 2: ALIGN XY ABOVE BOX ----
            elif step_name == 'pick_open':
                if self.box_pose_valid:
                    # stay slightly above before descending
                    position = self.box_position + np.array([0.0, 0.0, 0.3])

                    # lock XY position here
                    self.locked_pick_position = self.box_position.copy()
                else:
                    position = step['position']

            # ---- STEP 3: MOVE STRAIGHT DOWN ----
            elif step_name == 'pick_closed':
                if self.locked_pick_position is not None:
                    position = self.locked_pick_position.copy()
                else:
                    position = step['position']

            # ---- STEP 4: LIFT STRAIGHT UP ----
            elif step_name == 'lift_closed':
                if self.locked_pick_position is not None:
                    position = self.locked_pick_position + np.array([0.0, 0.0, 2.0])
                else:
                    position = step['position']

            # ---- OTHER STEPS ----
            else:
                position = step['position']

        else:
            position = step['position']

        finger = step['finger']

        # -------- IK --------
        if step['kind'] == 'pose':
            q_seed = self.q.copy() if self.have_joint_state else self.home_q.copy()
            arm_angles = self.computeIK(q_seed, position)
        else:
            arm_angles = step['q']

        # -------- LOGGING / VALIDATION --------
        ik_error = float(np.linalg.norm(position - self.FK(arm_angles)))
        ik_wrist_error = abs(self.wrap_angle(self.wrist_down_pitch - self.wrist_pitch(arm_angles)))

        if ik_error > self.position_tolerance:
            self.get_logger().warn(
                '%s IK is %.3f m from requested target %s. The target may be unreachable with wrist-down and joint limits.'
                % (step['name'], ik_error, self._fmt_vec(position))
            )

        # -------- TARGET --------
        target = np.concatenate((arm_angles, np.array([finger, finger])))

        self.active_step = {
            'name': step['name'],
            'target': target,
            'arm_angles': arm_angles,
            'position': position,
            'finger': finger,
            'settle': bool(step.get('settle', False)),
        }

        self.step_start_time = time.monotonic()
        self.last_progress_log = 0.0

        self.get_logger().info(
            '%s target position=%s joints=%s ik_error=%.3f wrist_error=%.1f deg finger=%.1f deg'
            % (
                step['name'],
                self._fmt_vec(position),
                self._fmt_angles_deg(arm_angles),
                ik_error,
                math.degrees(ik_wrist_error),
                math.degrees(finger),
            )
        )

    def process_active_step(self):
        if self.active_step is None:
            return

        self.publish_target(self.active_step['target'])

        if self.settle_until is not None:
            if time.monotonic() >= self.settle_until:
                self.advance_step()
            return

        name = self.active_step['name']
        arm_angles = self.active_step['arm_angles']
        target_position = self.active_step['position']
        finger_angle = self.active_step['finger']

        arm_error = float(np.max(np.abs(arm_angles - self.q)))
        position_error = float(np.linalg.norm(target_position - self.FK(self.q)))
        wrist_error = abs(self.wrap_angle(self.wrist_down_pitch - self.wrist_pitch(self.q)))
        finger_error = float(
            np.max(np.abs(np.array([finger_angle, finger_angle]) - self.fingers))
        )

        if (
            position_error < self.position_tolerance
            and wrist_error < self.wrist_tolerance
            and finger_error < self.joint_tolerance
        ):
            self.get_logger().info(
                '%s reached: ee_error=%.3f wrist_error=%.1f deg finger_error=%.1f deg'
                % (
                    name,
                    position_error,
                    math.degrees(wrist_error),
                    math.degrees(finger_error),
                )
            )
            settle_duration = self.dwell_sec if self.active_step['settle'] else 0.3
            self.settle_until = time.monotonic() + settle_duration
            return

        now = time.monotonic()
        if now - self.last_progress_log > 1.0:
            self.get_logger().info(
                '%s waiting: ee_error=%.3f wrist_error=%.1f deg arm_error=%.1f deg finger_error=%.1f deg'
                % (
                    name,
                    position_error,
                    math.degrees(wrist_error),
                    math.degrees(arm_error),
                    math.degrees(finger_error),
                )
            )
            self.last_progress_log = now

        if self.move_timeout > 0.0 and now - self.step_start_time > self.move_timeout:
            self.get_logger().error(
                '%s not reached: ee_error=%.3f wrist_error=%.1f deg arm_error=%.1f deg finger_error=%.1f deg. Holding this target.'
                % (
                    name,
                    position_error,
                    math.degrees(wrist_error),
                    math.degrees(arm_error),
                    math.degrees(finger_error),
                )
            )
            self.step_start_time = now

    def publish_target(self, target_angles_rad):
        msg = Float64MultiArray()
        msg.data = [math.degrees(value) for value in target_angles_rad]
        self.target_pub.publish(msg)

    def computeIK(self, q_seed, target_position):
        seeds = [q_seed.copy()]
        base_guess = math.atan2(-target_position[0], target_position[1])
        base_guess = max(min(base_guess, self.joint_limits[0, 1]), self.joint_limits[0, 0])

        for template in self.ik_seed_templates:
            seeds.append(np.concatenate((np.array([base_guess]), template)))

        best_q = None
        best_cost = float('inf')

        for seed in seeds:
            q = self.solveIKFromSeed(seed, target_position)
            position_error = float(np.linalg.norm(target_position - self.FK(q)))
            pitch_error = abs(self.wrap_angle(self.wrist_down_pitch - self.wrist_pitch(q)))
            limit_cost = float(np.sum(np.maximum(np.abs(q[1:]) - 1.45, 0.0)))
            cost = position_error + 0.5 * pitch_error + 0.1 * limit_cost

            if cost < best_cost:
                best_cost = cost
                best_q = q

        return best_q

    def solveIKFromSeed(self, q_seed, target_position):
        q = q_seed.copy()

        for _ in range(self.ik_iterations):
            end_position = self.FK(q)
            position_error = target_position - end_position
            pitch_error = self.wrap_angle(self.wrist_down_pitch - self.wrist_pitch(q))

            error = np.concatenate((
                position_error,
                np.array([self.orientation_weight * pitch_error], dtype=float),
            ))

            jacobian = np.zeros((4, 5), dtype=float)
            jacobian[:3, :] = self.computeJacobian(q)
            jacobian[3, :] = self.orientation_weight * np.array(
                [0.0, 1.0, 1.0, 1.0, 1.0],
                dtype=float,
            )

            lhs = jacobian @ jacobian.T + (self.damping ** 2) * np.eye(4)
            dq = jacobian.T @ np.linalg.solve(lhs, error)
            dq = self.clamp_vector_norm(dq, self.max_dq_step)

            q = q + dq
            q = np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

            if np.linalg.norm(position_error) < 0.02 and abs(pitch_error) < math.radians(3.0):
                break

        return q

    def FK(self, q):
        transform = np.eye(4)

        transform = transform @ self.translate(0.0, 0.0, 0.1) @ self.rot_z(q[0])
        transform = transform @ self.translate(0.0, 0.0, 1.67744) @ self.rot_x(q[1])
        transform = transform @ self.translate(0.0, -0.007232, 2.92735) @ self.rot_x(q[2])
        transform = transform @ self.translate(0.0, 0.002234, 2.9244) @ self.rot_x(q[3])
        transform = transform @ self.translate(0.0, 0.012929, 2.12653) @ self.rot_x(q[4])

        return transform[:3, 3].copy()

    def computeJacobian(self, q):
        end_position, origins, axes = self.fk_frames(q)
        jacobian = np.zeros((3, 5), dtype=float)

        for i in range(5):
            jacobian[:, i] = np.cross(axes[i], end_position - origins[i])

        return jacobian

    def fk_frames(self, q):
        transform = np.eye(4)
        origins = []
        axes = []

        transform = transform @ self.translate(0.0, 0.0, 0.1)
        origins.append(transform[:3, 3].copy())
        axes.append((transform[:3, :3] @ np.array([0.0, 0.0, 1.0])).copy())
        transform = transform @ self.rot_z(q[0])

        transform = transform @ self.translate(0.0, 0.0, 1.67744)
        origins.append(transform[:3, 3].copy())
        axes.append((transform[:3, :3] @ np.array([1.0, 0.0, 0.0])).copy())
        transform = transform @ self.rot_x(q[1])

        transform = transform @ self.translate(0.0, -0.007232, 2.92735)
        origins.append(transform[:3, 3].copy())
        axes.append((transform[:3, :3] @ np.array([1.0, 0.0, 0.0])).copy())
        transform = transform @ self.rot_x(q[2])

        transform = transform @ self.translate(0.0, 0.002234, 2.9244)
        origins.append(transform[:3, 3].copy())
        axes.append((transform[:3, :3] @ np.array([1.0, 0.0, 0.0])).copy())
        transform = transform @ self.rot_x(q[3])

        transform = transform @ self.translate(0.0, 0.012929, 2.12653)
        origins.append(transform[:3, 3].copy())
        axes.append((transform[:3, :3] @ np.array([1.0, 0.0, 0.0])).copy())
        transform = transform @ self.rot_x(q[4])

        return transform[:3, 3].copy(), origins, axes

    def wrist_pitch(self, q):
        return float(q[1] + q[2] + q[3] + q[4])

    def rotate_about_z(self, position, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array([
            c * position[0] - s * position[1],
            s * position[0] + c * position[1],
            position[2],
        ], dtype=float)

    def translate(self, x, y, z):
        transform = np.eye(4)
        transform[:3, 3] = [x, y, z]
        return transform

    def rot_x(self, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)

    def rot_z(self, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array([
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)

    def clamp_vector_norm(self, vector, max_norm):
        norm = float(np.linalg.norm(vector))
        if norm > max_norm and norm > 0.0:
            return vector * (max_norm / norm)
        return vector

    def wrap_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _fmt_vec(self, vector):
        return '[%.3f, %.3f, %.3f]' % (vector[0], vector[1], vector[2])

    def _fmt_angles_deg(self, vector):
        return [round(math.degrees(value), 1) for value in vector]


def main():
    rclpy.init()
    node = PickAndPlace()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
