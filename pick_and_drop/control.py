import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float64
from sensor_msgs.msg import JointState
import math

class AngleToTorque(Node):
    def __init__(self):
        super().__init__('angle_to_torque')

        self.joint_names = [
            'base_joint','joint1','joint2','joint3','joint4',
            'finger1_joint','finger2_joint'
        ]

        # state
        self.pos = [0.0]*7
        self.vel = [0.0]*7

        # target (in radians)
        self.target = None

        # gains (start conservative)
        self.kp = [5000, 16000, 11000, 7000, 4000, 80, 80]

        # critical damping
        self.kd = [
            20 * math.sqrt(self.kp[0]),                                      # base
            25 * math.sqrt(self.kp[1]),               # joint1
            17 * math.sqrt(self.kp[2]),               # joint2
            12 * math.sqrt(self.kp[3]),               # joint3
            9  * math.sqrt(self.kp[4]),               # joint4
            3  * math.sqrt(self.kp[5]),               # finger1
            3  * math.sqrt(self.kp[6])                # finger2
        ]

        # Integral action removes the steady holding error left by PD under gravity/friction.
        self.ki = [20, 800, 600, 400, 250, 8, 8]
        self.integral_error = [0.0]*7
        self.integral_limit = [
            1.0,
            0.8,
            0.8,
            0.7,
            0.7,
            0.3,
            0.3,
        ]
        self.max_torque = [500, 2500, 2200, 1600, 1200, 80, 80]

        # finetuned steps
        self.step = 8
        self.trajectory_target = [0.0]*7
        self.trajectory_vel = [0.0]*7

        self.max_vel = [0.6, 0.5, 0.4, 0.3, 0.3, 1.0, 1.0]
        self.max_acc = [1.5, 1.2, 1.0, 0.8, 0.8, 2.0, 2.0]
        # publisher → torque controller
        self.pubs = []
        for j in self.joint_names:
            topic = f'/model/pick_and_drop_arm/joint/{j}/cmd_force'
            self.pubs.append(self.create_publisher(Float64, topic, 10))

        # subscribers
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.create_subscription(
            Float64MultiArray,
            '/joint_angles',
            self.angle_cb,
            10
        )

        self.timer = self.create_timer(0.01, self.control_loop)

        self.get_logger().info("Angle-to-torque node ready")
        self.trajectory = None
        self.trajectory_start_time = None
        self.motion_start = None
        self.motion_duration = 3.0

        self.start_target = [0.0]*7
        self.goal_target = [0.0]*7

        self.last_debug_time = self.get_clock().now()
        self.debug_interval = 1.0

    # ----------------------------------------
    def joint_cb(self, msg):
        name_to_index = {n: i for i, n in enumerate(msg.name)}

        for i, j in enumerate(self.joint_names):
            if j in name_to_index:
                idx = name_to_index[j]
                self.pos[i] = msg.position[idx]
                self.vel[i] = msg.velocity[idx]
        if self.target is None:
            self.target = self.pos.copy()
            self.trajectory_target = self.pos.copy()

        # deg_vals = [round(math.degrees(p), 2) for p in self.pos]
        # self.get_logger().info(f"Current (deg): {deg_vals}")

    # ----------------------------------------
    def angle_cb(self, msg):

        if self.target is None:
            return

        n = min(len(msg.data), len(self.joint_names))

        self.start_target = self.target.copy()

        for i in range(n):
            self.goal_target[i] = msg.data[i]

        self.motion_start = self.get_clock().now()

        self.get_logger().info("Received new trajectory target")
    # ----------------------------------------
    def control_loop(self):

        if self.target is None:
            return

        dt = 0.01

        # ----------------------------------------
        # 🔥 CUSTOM TRAJECTORY INTERPOLATION
        # ----------------------------------------

        if self.motion_start is not None:

            now = self.get_clock().now()

            t = (
                now - self.motion_start
            ).nanoseconds * 1e-9

            ratio = min(t / self.motion_duration, 1.0)

            # cubic smoothstep interpolation
            s = 3 * (ratio ** 2) - 2 * (ratio ** 3)

            for i in range(len(self.joint_names)):

                self.target[i] = (
                    self.start_target[i]
                    + s * (
                        self.goal_target[i]
                        - self.start_target[i]
                    )
                )

            if ratio >= 1.0:
                self.motion_start = None

        # ----------------------------------------
        # 🔄 TRAJECTORY SMOOTHING
        # ----------------------------------------

        for i in range(len(self.joint_names)):

            error = self.target[i] - self.trajectory_target[i]

            desired_vel = max(
                min(error / dt, self.max_vel[i]),
                -self.max_vel[i]
            )

            vel_diff = desired_vel - self.trajectory_vel[i]

            max_step = self.max_acc[i] * dt

            if abs(vel_diff) > max_step:
                self.trajectory_vel[i] += math.copysign(
                    max_step,
                    vel_diff
                )
            else:
                self.trajectory_vel[i] = desired_vel

            self.trajectory_target[i] += (
                self.trajectory_vel[i] * dt
            )

        # ----------------------------------------
        # 🔧 PID TORQUE CONTROL
        # ----------------------------------------

        torques = [0.0] * len(self.joint_names)

        # ---- arm joints ----

        for i in range(len(self.joint_names) - 2):

            error = (
                self.trajectory_target[i]
                - self.pos[i]
            )

            if abs(error) > math.radians(self.step):
                error = math.copysign(
                    math.radians(self.step),
                    error
                )

            self.integral_error[i] += error * dt

            self.integral_error[i] = max(
                min(
                    self.integral_error[i],
                    self.integral_limit[i]
                ),
                -self.integral_limit[i]
            )

            tau = (
                self.kp[i] * error
                + self.ki[i] * self.integral_error[i]
                - self.kd[i] * self.vel[i]
            )

            torques[i] = float(
                max(
                    min(tau, self.max_torque[i]),
                    -self.max_torque[i]
                )
            )

        # ----------------------------------------
        # 🤏 GRIPPER CONTROL
        # ----------------------------------------

        grip = self.target[5]

        error1 = grip - self.pos[5]
        error2 = -grip - self.pos[6]

        finger_step = 1

        if abs(error1) > math.radians(finger_step):
            error1 = math.copysign(
                math.radians(finger_step),
                error1
            )

        if abs(error2) > math.radians(finger_step):
            error2 = math.copysign(
                math.radians(finger_step),
                error2
            )

        avg_vel = (
            self.vel[5] + self.vel[6]
        ) / 2.0

        # finger 1

        self.integral_error[5] += error1 * dt

        self.integral_error[5] = max(
            min(
                self.integral_error[5],
                self.integral_limit[5]
            ),
            -self.integral_limit[5]
        )

        tau1 = (
            self.kp[5] * error1
            + self.ki[5] * self.integral_error[5]
            - self.kd[5] * avg_vel
        )

        # finger 2

        self.integral_error[6] += error2 * dt

        self.integral_error[6] = max(
            min(
                self.integral_error[6],
                self.integral_limit[6]
            ),
            -self.integral_limit[6]
        )

        tau2 = (
            self.kp[6] * error2
            + self.ki[6] * self.integral_error[6]
            - self.kd[6] * avg_vel
        )

        torques[5] = float(
            max(
                min(tau1, self.max_torque[5]),
                -self.max_torque[5]
            )
        )

        torques[6] = float(
            max(
                min(tau2, self.max_torque[6]),
                -self.max_torque[6]
            )
        )
        # ----------------------------------------
        # 🖨 DEBUG PRINT
        # ----------------------------------------
        error_deg = [
            round(
                math.degrees(
                    self.trajectory_target[i]
                    - self.pos[i]
                ),
                2
            )
            for i in range(len(self.pos))
        ]

        print(f"Error (deg): {error_deg}", flush=True)
        # ----------------------------------------
        # 📤 PUBLISH TORQUES
        # ----------------------------------------

        for i in range(len(self.joint_names)):

            msg = Float64()

            msg.data = float(torques[i])

            self.pubs[i].publish(msg)

# ----------------------------------------
def main():
    rclpy.init()
    node = AngleToTorque()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
