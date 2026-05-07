import math
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class IKNode(Node):

    def __init__(self):

        super().__init__('ik_node')

        self.joint_names = [
            'base_joint',
            'joint1',
            'joint2',
            'joint3',
            'joint4',
        ]

        self.q = np.zeros(5, dtype=float)

        self.have_joint_state = False

        # joint limits
        self.joint_limits = np.array([
            [-math.pi, math.pi],
            [-1.57, 1.57],
            [-1.57, 1.57],
            [-1.57, 1.57],
            [-1.57, 1.57],
        ])

        # IK settings
        self.damping = 0.2
        self.max_dq = 0.05
        self.iterations = 100
        self.tolerance = 0.01

        # publisher
        self.angle_pub = self.create_publisher(
            Float64MultiArray,
            '/joint_angles',
            10
        )

        # subscribers
        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_cb,
            10
        )

        self.create_subscription(
            Point,
            '/ik_target',
            self.target_cb,
            10
        )

        self.get_logger().info('Simple IK node started')

    # =========================================================
    # JOINT STATES
    # =========================================================

    def joint_cb(self, msg):

        name_to_index = {
            name: i
            for i, name in enumerate(msg.name)
        }

        for i, joint in enumerate(self.joint_names):

            if joint in name_to_index:

                idx = name_to_index[joint]

                self.q[i] = msg.position[idx]

        ee = self.fk(self.q)

        print(
            f"EE Position: "
            f"[{ee[0]:.2f}, {ee[1]:.2f}, {ee[2]:.2f}]",
            flush=True
        )

        self.have_joint_state = True

    # =========================================================
    # TARGET CALLBACK
    # =========================================================

    def target_cb(self, msg):

        if not self.have_joint_state:
            self.get_logger().warn('Waiting for joint states...')
            return

        target = np.array([
            msg.x,
            msg.y,
            msg.z
        ])

        self.get_logger().info(
            f'Target: [{msg.x:.2f}, {msg.y:.2f}, {msg.z:.2f}]'
        )

        q_sol = self.compute_ik(
            self.q.copy(),
            target
        )

        out = Float64MultiArray()

        # 7 joints expected by controller
        out.data = [
            q_sol[0],
            q_sol[1],
            q_sol[2],
            q_sol[3],
            q_sol[4],
            0.0,
            0.0
        ]

        self.angle_pub.publish(out)

        self.get_logger().info(
            f'Published: {[round(math.degrees(v),1) for v in q_sol]} deg'
        )

    # =========================================================
    # FK
    # =========================================================

    def fk(self, q):

        T = np.eye(4)

        T = T @ self.trans(0, 0, 0.1) @ self.rotz(q[0])
        T = T @ self.trans(0, 0, 1.67744) @ self.rotx(q[1])
        T = T @ self.trans(0, -0.007232, 2.92735) @ self.rotx(q[2])
        T = T @ self.trans(0, 0.002234, 2.9244) @ self.rotx(q[3])
        T = T @ self.trans(0, 0.012929, 2.12653) @ self.rotx(q[4])

        return T[:3, 3]

    # =========================================================
    # FK FRAMES
    # =========================================================

    def fk_frames(self, q):

        T = np.eye(4)

        origins = []
        axes = []

        T = T @ self.trans(0, 0, 0.1)

        origins.append(T[:3, 3].copy())
        axes.append(T[:3, :3] @ np.array([0, 0, 1]))

        T = T @ self.rotz(q[0])

        T = T @ self.trans(0, 0, 1.67744)

        origins.append(T[:3, 3].copy())
        axes.append(T[:3, :3] @ np.array([1, 0, 0]))

        T = T @ self.rotx(q[1])

        T = T @ self.trans(0, -0.007232, 2.92735)

        origins.append(T[:3, 3].copy())
        axes.append(T[:3, :3] @ np.array([1, 0, 0]))

        T = T @ self.rotx(q[2])

        T = T @ self.trans(0, 0.002234, 2.9244)

        origins.append(T[:3, 3].copy())
        axes.append(T[:3, :3] @ np.array([1, 0, 0]))

        T = T @ self.rotx(q[3])

        T = T @ self.trans(0, 0.012929, 2.12653)

        origins.append(T[:3, 3].copy())
        axes.append(T[:3, :3] @ np.array([1, 0, 0]))

        return T[:3, 3], origins, axes

    # =========================================================
    # JACOBIAN
    # =========================================================

    def jacobian(self, q):

        end, origins, axes = self.fk_frames(q)

        J = np.zeros((3, 5))

        for i in range(5):

            J[:, i] = np.cross(
                axes[i],
                end - origins[i]
            )

        return J

    # =========================================================
    # IK
    # =========================================================

    def compute_ik(self, q, target):

        for i in range(self.iterations):

            current = self.fk(q)

            error = target - current

            error_norm = np.linalg.norm(error)

            print(
                f"Iter {i} | "
                f"Current: {current.round(3)} | "
                f"Target: {target.round(3)} | "
                f"Error: {error.round(3)} | "
                f"Norm: {error_norm:.4f}",
                flush=True
            )

            if error_norm < self.tolerance:
                print("IK CONVERGED", flush=True)
                break

            J = self.jacobian(q)

            lhs = (
                J @ J.T
                + (self.damping ** 2) * np.eye(3)
            )

            dq = J.T @ np.linalg.solve(lhs, error)

            norm = np.linalg.norm(dq)

            if norm > self.max_dq:
                dq *= self.max_dq / norm

            q += dq

            q = np.clip(
                q,
                self.joint_limits[:, 0],
                self.joint_limits[:, 1]
            )

        return q

    # =========================================================
    # HELPERS
    # =========================================================

    def trans(self, x, y, z):

        T = np.eye(4)

        T[:3, 3] = [x, y, z]

        return T

    def rotx(self, a):

        c = math.cos(a)
        s = math.sin(a)

        return np.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
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


def main():

    rclpy.init()

    node = IKNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()