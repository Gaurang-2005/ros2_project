import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
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
        self.kp = [500, 10000, 5000, 2500, 1250, 50, 50]

        # critical damping
        self.kd = [12.5 * math.sqrt(k) for k in self.kp]
        # finetuned steps
        self.step = 8
        self.trajectory_target = [0.0]*7
        self.trajectory_vel = [0.0]*7

        self.max_vel = [0.6, 0.5, 0.4, 0.3, 0.3, 1.0, 1.0]
        self.max_acc = [1.5, 1.2, 1.0, 0.8, 0.8, 2.0, 2.0]
        # publisher → torque controller
        self.pub = self.create_publisher(Float64MultiArray, '/joint_torque', 10)

        # subscribers
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.create_subscription(Float64MultiArray, '/target_angles_deg', self.target_cb, 10)

        self.timer = self.create_timer(0.01, self.control_loop)

        self.get_logger().info("Angle-to-torque node ready")

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
    def target_cb(self, msg):

        if len(msg.data) != len(self.joint_names):
            self.get_logger().warn("Wrong target size")
            return

        # convert degrees → radians
        self.target = [math.radians(x) for x in msg.data]

        # self.get_logger().info(f"Target (rad): {self.target}")

    # ----------------------------------------
    def control_loop(self):

        if self.target is None:
            return

        dt = 0.01

        for i in range(len(self.joint_names)):

            error = self.target[i] - self.trajectory_target[i]

            # desired velocity (bounded)
            desired_vel = max(min(error / dt, self.max_vel[i]), -self.max_vel[i])

            # smooth acceleration toward desired velocity
            vel_diff = desired_vel - self.trajectory_vel[i]
            max_step = self.max_acc[i] * dt

            if abs(vel_diff) > max_step:
                self.trajectory_vel[i] += math.copysign(max_step, vel_diff)
            else:
                self.trajectory_vel[i] = desired_vel

            # update position
            self.trajectory_target[i] += self.trajectory_vel[i] * dt
        
        torques = [0.0]*len(self.joint_names)
        error_deg = [round(math.degrees(self.target[i] - self.pos[i]), 2) for i in range(len(self.pos))]
        self.get_logger().info(f"Error (deg): {error_deg}")
        for i in range(len(self.joint_names) - 2):
            
            error = self.trajectory_target[i] - self.pos[i]
            if abs(error) > math.radians(self.step):
                error = math.copysign(math.radians(self.step), error)
            tau = (
                self.kp[i]*error
                - self.kd[i] * self.vel[i]
            )


            torques[i] = float(tau)
        # compute errors for both fingers
        error1 = self.trajectory_target[5] - self.pos[5]
        error2 = self.trajectory_target[6] - self.pos[6]
        finger_step = 1
        # average error (best for symmetry)
        if abs(error1) > math.radians(finger_step):
            error1 = math.copysign(math.radians(finger_step), error1)
        if abs(error2) > math.radians(finger_step):
            error2 = math.copysign(math.radians(finger_step), error2)

        # compute torque
        grip_tau = self.kp[5] * error1 - self.kd[5] * self.vel[5]
        torques[5] = float(grip_tau)
        grip_tau = self.kp[5] * error2 - self.kd[5] * self.vel[6]
        torques[6] = float(grip_tau)
        msg = Float64MultiArray()
        msg.data = torques
        self.pub.publish(msg)


# ----------------------------------------
def main():
    rclpy.init()
    node = AngleToTorque()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()