import math

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_msgs.msg import Float64MultiArray


class TeleopIK(Node):
    def __init__(self):
        super().__init__('teleop_ik')

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
        self.qd = np.zeros(5, dtype=float)
        self.finger_targets = np.zeros(2, dtype=float)
        self.q_des = None
        self.target_position = None
        self.last_cmd_active = False
        self.cmd = np.zeros(3, dtype=float)
        self.have_joint_state = False
        self.homing = True
        self.autonomous_active = False
        self.autonomous_pause_logged = False
        self.debug_count = 0
        self.home_start_time = None

        self.damping = 0.2
        self.teleop_scale = 0.5
        self.max_cartesian_step = 0.03
        self.max_dq_step = 0.04
        self.max_target_error = 0.15
        self.ik_iterations = 4
        self.position_tolerance = 0.005
        self.wrist_down_pitch = math.radians(180.0)
        self.orientation_weight = 3.0
        self.home_tolerance = math.radians(10.0)
        self.home_timeout_sec = 8.0
        self.edge_warning_margin = 0.05
        self.last_workspace_warning_time = 0
        self.last_home_log_time = 0
        self.last_homing_cmd_warning_time = 0

        self.declare_parameter('home_base_deg', 0.0)
        self.declare_parameter('home_joint1_deg', -50.0)
        self.declare_parameter('home_joint2_deg', 35.0)
        self.declare_parameter('home_joint3_deg', -75.0)
        self.declare_parameter('home_joint4_deg', -90.0)
        self.declare_parameter('home_tolerance_deg', 10.0)
        self.declare_parameter('home_timeout_sec', self.home_timeout_sec)
        self.declare_parameter('wrist_down_pitch_deg', 180.0)
        self.declare_parameter('orientation_weight', self.orientation_weight)
        self.declare_parameter('workspace_min_x', -4.0)
        self.declare_parameter('workspace_max_x', 4.0)
        self.declare_parameter('workspace_min_y', -5.8)
        self.declare_parameter('workspace_max_y', 5.8)
        self.declare_parameter('workspace_min_z', 0.8)
        self.declare_parameter('workspace_max_z', 8.5)
        self.declare_parameter('edge_warning_margin', self.edge_warning_margin)

        self.home_q = np.radians(np.array([
            self.get_parameter('home_base_deg').value,
            self.get_parameter('home_joint1_deg').value,
            self.get_parameter('home_joint2_deg').value,
            self.get_parameter('home_joint3_deg').value,
            self.get_parameter('home_joint4_deg').value,
        ], dtype=float))
        self.home_q = np.clip(
            self.home_q,
            self.joint_limits[:, 0],
            self.joint_limits[:, 1],
        )
        self.home_tolerance = math.radians(
            float(self.get_parameter('home_tolerance_deg').value)
        )
        self.home_timeout_sec = float(self.get_parameter('home_timeout_sec').value)
        self.wrist_down_pitch = math.radians(
            float(self.get_parameter('wrist_down_pitch_deg').value)
        )
        self.orientation_weight = float(self.get_parameter('orientation_weight').value)

        self.workspace_min = np.array([
            self.get_parameter('workspace_min_x').value,
            self.get_parameter('workspace_min_y').value,
            self.get_parameter('workspace_min_z').value,
        ], dtype=float)
        self.workspace_max = np.array([
            self.get_parameter('workspace_max_x').value,
            self.get_parameter('workspace_max_y').value,
            self.get_parameter('workspace_max_z').value,
        ], dtype=float)
        self.edge_warning_margin = float(
            self.get_parameter('edge_warning_margin').value
        )

        self.target_pub = self.create_publisher(Float64MultiArray, '/joint_angle', 10)

        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_cb, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)
        self.create_subscription(Bool, '/autonomous_mode', self.autonomous_mode_cb, 10)

        self.timer = self.create_timer(0.01, self.control_loop)

        self.get_logger().info(
            'Teleop IK node ready. Homing first to %s deg, then publishing IK targets.'
            % self._fmt_angles_deg(self.home_q)
        )

    def cmd_vel_cb(self, msg):
        self.cmd = np.array([
            msg.linear.x,
            msg.linear.y,
            msg.linear.z,
        ], dtype=float)

        if self.homing and np.linalg.norm(self.cmd) > 0.0:
            self.warn_cmd_ignored_during_homing()

    def autonomous_mode_cb(self, msg):
        self.autonomous_active = bool(msg.data)
        if not self.autonomous_active:
            self.autonomous_pause_logged = False
            if self.have_joint_state:
                self.q_des = self.q.copy()
                self.target_position = self.clamp_workspace(self.FK(self.q))
                self.cmd = np.zeros(3, dtype=float)

    def joint_state_cb(self, msg):
        name_to_index = {name: index for index, name in enumerate(msg.name)}

        for i, joint_name in enumerate(self.arm_joint_names):
            if joint_name in name_to_index:
                index = name_to_index[joint_name]
                self.q[i] = msg.position[index]
                if index < len(msg.velocity):
                    self.qd[i] = msg.velocity[index]

        for i, joint_name in enumerate(self.full_joint_names[5:]):
            if joint_name in name_to_index:
                index = name_to_index[joint_name]
                self.finger_targets[i] = msg.position[index]

        if not self.have_joint_state:
            self.q_des = self.home_q.copy()
            self.target_position = self.clamp_workspace(self.FK(self.home_q))
            self.have_joint_state = True
            self.get_logger().info(
                'Initialized home target at %s. Waiting for arm to reach home.'
                % self._fmt_vec(self.target_position)
            )

    def FK(self, q):
        transform = np.eye(4)

        transform = transform @ self._translate(0.0, 0.0, 0.1) @ self._rot_z(q[0])
        transform = transform @ self._translate(0.0, 0.0, 1.67744) @ self._rot_x(q[1])
        transform = transform @ self._translate(0.0, -0.007232, 2.92735) @ self._rot_x(q[2])
        transform = transform @ self._translate(0.0, 0.002234, 2.9244) @ self._rot_x(q[3])
        transform = transform @ self._translate(0.0, 0.012929, 2.12653) @ self._rot_x(q[4])

        return transform[:3, 3].copy()

    def _fk_frames(self, q):
        transform = np.eye(4)
        origins = []
        axes = []

        transform = transform @ self._translate(0.0, 0.0, 0.1)
        origins.append(transform[:3, 3].copy())
        axes.append((transform[:3, :3] @ np.array([0.0, 0.0, 1.0])).copy())
        transform = transform @ self._rot_z(q[0])

        transform = transform @ self._translate(0.0, 0.0, 1.67744)
        origins.append(transform[:3, 3].copy())
        axes.append((transform[:3, :3] @ np.array([1.0, 0.0, 0.0])).copy())
        transform = transform @ self._rot_x(q[1])

        transform = transform @ self._translate(0.0, -0.007232, 2.92735)
        origins.append(transform[:3, 3].copy())
        axes.append((transform[:3, :3] @ np.array([1.0, 0.0, 0.0])).copy())
        transform = transform @ self._rot_x(q[2])

        transform = transform @ self._translate(0.0, 0.002234, 2.9244)
        origins.append(transform[:3, 3].copy())
        axes.append((transform[:3, :3] @ np.array([1.0, 0.0, 0.0])).copy())
        transform = transform @ self._rot_x(q[3])

        transform = transform @ self._translate(0.0, 0.012929, 2.12653)
        origins.append(transform[:3, 3].copy())
        axes.append((transform[:3, :3] @ np.array([1.0, 0.0, 0.0])).copy())
        transform = transform @ self._rot_x(q[4])

        return transform[:3, 3].copy(), origins, axes

    def computeJacobian(self, q):
        end_position, origins, axes = self._fk_frames(q)
        jacobian = np.zeros((3, 5), dtype=float)

        for i in range(5):
            jacobian[:, i] = np.cross(axes[i], end_position - origins[i])

        return jacobian

    def computeIK(self, q_seed, target_position):
        q = q_seed.copy()

        for _ in range(self.ik_iterations):
            end_position = self.FK(q)
            position_error = target_position - end_position
            pitch_error = self._wrap_angle(
                self.wrist_down_pitch - self.wrist_pitch(q)
            )
            error = np.concatenate((
                position_error,
                np.array([self.orientation_weight * pitch_error], dtype=float),
            ))

            if (
                np.linalg.norm(position_error) < self.position_tolerance
                and abs(pitch_error) < math.radians(2.0)
            ):
                break

            jacobian = np.zeros((4, 5), dtype=float)
            jacobian[:3, :] = self.computeJacobian(q)
            jacobian[3, :] = self.orientation_weight * np.array(
                [0.0, 1.0, 1.0, 1.0, 1.0],
                dtype=float,
            )
            lhs = jacobian @ jacobian.T + (self.damping ** 2) * np.eye(4)
            dq = jacobian.T @ np.linalg.solve(lhs, error)
            dq = self._clamp_vector_norm(dq, self.max_dq_step)

            q = q + dq
            q = np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

        return q

    def wrist_pitch(self, q):
        return float(q[1] + q[2] + q[3] + q[4])

    def limit_target_error(self):
        reference_position = self.FK(self.q_des)
        error = self.target_position - reference_position
        error_norm = float(np.linalg.norm(error))

        if error_norm > self.max_target_error and error_norm > 0.0:
            self.target_position = (
                reference_position + error * (self.max_target_error / error_norm)
            )

    def clamp_workspace(self, target_position):
        clamped_position = np.clip(target_position, self.workspace_min, self.workspace_max)
        self.warn_if_near_workspace_edge(target_position, clamped_position)
        return clamped_position

    def warn_if_near_workspace_edge(self, target_position, clamped_position):
        warning_faces = []

        if not np.allclose(target_position, clamped_position):
            if target_position[0] < self.workspace_min[0]:
                warning_faces.append('min X')
            elif target_position[0] > self.workspace_max[0]:
                warning_faces.append('max X')

            if target_position[1] < self.workspace_min[1]:
                warning_faces.append('min Y')
            elif target_position[1] > self.workspace_max[1]:
                warning_faces.append('max Y')

            if target_position[2] < self.workspace_min[2]:
                warning_faces.append('min Z')
            elif target_position[2] > self.workspace_max[2]:
                warning_faces.append('max Z')
        else:
            margin = self.edge_warning_margin
            if self.cmd[0] < 0.0 and clamped_position[0] - self.workspace_min[0] <= margin:
                warning_faces.append('min X')
            elif self.cmd[0] > 0.0 and self.workspace_max[0] - clamped_position[0] <= margin:
                warning_faces.append('max X')

            if self.cmd[1] < 0.0 and clamped_position[1] - self.workspace_min[1] <= margin:
                warning_faces.append('min Y')
            elif self.cmd[1] > 0.0 and self.workspace_max[1] - clamped_position[1] <= margin:
                warning_faces.append('max Y')

            if self.cmd[2] < 0.0 and clamped_position[2] - self.workspace_min[2] <= margin:
                warning_faces.append('min Z')
            elif self.cmd[2] > 0.0 and self.workspace_max[2] - clamped_position[2] <= margin:
                warning_faces.append('max Z')

        if warning_faces:
            self.warn_workspace_edge(warning_faces, clamped_position)

    def warn_workspace_edge(self, warning_faces, clamped_position):
        now = self.get_clock().now().nanoseconds
        if now - self.last_workspace_warning_time < 1_000_000_000:
            return

        self.last_workspace_warning_time = now
        self.get_logger().warn(
            'Workspace edge reached: %s. Target clamped to %s inside box min=%s max=%s'
            % (
                ', '.join(warning_faces),
                self._fmt_vec(clamped_position),
                self._fmt_vec(self.workspace_min),
                self._fmt_vec(self.workspace_max),
            )
        )

    def control_loop(self):
        if not self.have_joint_state:
            return

        if self.autonomous_active:
            if not self.autonomous_pause_logged:
                self.get_logger().info(
                    'Autonomous mode active. Teleop IK is paused and will not publish /target_angles_deg.'
                )
                self.autonomous_pause_logged = True
            return

        dt = 0.01

        if self.homing:
            if self.home_start_time is None:
                self.home_start_time = self.get_clock().now()

            self.publish_target(self.home_q)
            home_error = float(np.max(np.abs(self.home_q - self.q)))
            elapsed = (
                self.get_clock().now() - self.home_start_time
            ).nanoseconds / 1_000_000_000.0

            if home_error <= self.home_tolerance or elapsed >= self.home_timeout_sec:
                self.homing = False
                self.q_des = self.home_q.copy()
                self.target_position = self.clamp_workspace(self.FK(self.home_q))
                self.cmd = np.zeros(3, dtype=float)
                self.last_cmd_active = False
                if home_error <= self.home_tolerance:
                    self.get_logger().info(
                        'Home reached. Holding home pose at %s. Cartesian teleop enabled.'
                        % self._fmt_vec(self.target_position)
                    )
                else:
                    self.get_logger().warn(
                        'Home timeout after %.1f sec with %.1f deg max error. Holding commanded home pose at %s and enabling Cartesian teleop.'
                        % (
                            elapsed,
                            math.degrees(home_error),
                            self._fmt_vec(self.target_position),
                        )
                    )
            else:
                self.log_homing_status(home_error, elapsed)

            return

        cartesian_step = self.cmd * self.teleop_scale * dt
        cartesian_step = self._clamp_vector_norm(cartesian_step, self.max_cartesian_step)
        cmd_active = float(np.linalg.norm(cartesian_step)) > 1e-9

        if cmd_active:
            self.target_position = self.target_position + cartesian_step
            self.target_position = self.clamp_workspace(self.target_position)
            self.limit_target_error()
            self.target_position = self.clamp_workspace(self.target_position)
            self.q_des = self.computeIK(self.q_des, self.target_position)
            self.publish_target(self.q_des)
            self.last_cmd_active = True
        elif self.last_cmd_active:
            self.target_position = self.clamp_workspace(self.FK(self.q_des))
            self.publish_target(self.q_des)
            self.last_cmd_active = False

        self.debug_count += 1
        if self.debug_count >= 50:
            current_position = self.FK(self.q)
            error = self.target_position - current_position
            q_deg = [round(math.degrees(value), 1) for value in self.q]
            q_des_deg = [round(math.degrees(value), 1) for value in self.q_des]
            wrist_pitch_deg = round(math.degrees(self.wrist_pitch(self.q_des)), 1)
            self.get_logger().info(
                'cmd=%s target=%s current=%s error_norm=%.4f wrist_pitch_des=%.1f q=%s q_des=%s'
                % (
                    self._fmt_vec(self.cmd),
                    self._fmt_vec(self.target_position),
                    self._fmt_vec(current_position),
                    float(np.linalg.norm(error)),
                    wrist_pitch_deg,
                    q_deg,
                    q_des_deg,
                )
            )
            self.debug_count = 0

    def publish_target(self, arm_angles):
        msg = Float64MultiArray()
        target_angles_rad = np.concatenate((arm_angles, self.finger_targets))
        msg.data = [math.degrees(value) for value in target_angles_rad]
        self.target_pub.publish(msg)

    def log_homing_status(self, home_error, elapsed):
        now = self.get_clock().now().nanoseconds
        if now - self.last_home_log_time < 1_000_000_000:
            return

        self.last_home_log_time = now
        self.get_logger().info(
            'Homing... %.1f/%.1f sec, max joint error %.1f deg, current=%s target=%s'
            % (
                elapsed,
                self.home_timeout_sec,
                math.degrees(home_error),
                self._fmt_angles_deg(self.q),
                self._fmt_angles_deg(self.home_q),
            )
        )

    def warn_cmd_ignored_during_homing(self):
        now = self.get_clock().now().nanoseconds
        if now - self.last_homing_cmd_warning_time < 1_000_000_000:
            return

        self.last_homing_cmd_warning_time = now
        self.get_logger().warn(
            'Keyboard command received, but teleop is disabled until homing finishes.'
        )

    def _translate(self, x, y, z):
        transform = np.eye(4)
        transform[:3, 3] = [x, y, z]
        return transform

    def _rot_x(self, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)

    def _rot_z(self, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array([
            [c, -s, 0.0, 0.0],
            [s, c, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)

    def _clamp_vector_norm(self, vector, max_norm):
        norm = float(np.linalg.norm(vector))
        if norm > max_norm and norm > 0.0:
            return vector * (max_norm / norm)
        return vector

    def _wrap_angle(self, angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _fmt_vec(self, vector):
        return '[%.3f, %.3f, %.3f]' % (vector[0], vector[1], vector[2])

    def _fmt_angles_deg(self, vector):
        return [round(math.degrees(value), 1) for value in vector]


def main():
    rclpy.init()
    node = TeleopIK()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
