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

        # =====================================================
        # JOINT NAMES
        # =====================================================

        self.joint_names = [
            'base_joint',
            'joint1',
            'joint2',
            'joint3',
            'joint4',
        ]

        # =====================================================
        # CURRENT JOINT STATE
        # =====================================================

        self.q = np.zeros(5, dtype=float)

        self.have_joint_state = False

        # =====================================================
        # JOINT LIMITS
        # =====================================================

        self.joint_limits = np.array([

            [-math.pi, math.pi],

            [-1.57, 1.0],

            [-1.57, 1.0],

            [-1.57, 1.0],

            [-1.57, 1.0],

        ])

        # =====================================================
        # IK SETTINGS
        # =====================================================

        self.damping = 0.04

        self.max_dq = 0.03

        self.alpha = 0.3

        self.iterations = 300

        self.tolerance = 0.01

        # =====================================================
        # PUBLISHER
        # =====================================================

        self.angle_pub = self.create_publisher(
            Float64MultiArray,
            '/joint_angles',
            10
        )

        # =====================================================
        # SUBSCRIBERS
        # =====================================================

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

        self.get_logger().info(
            'Stable IK node started'
        )

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

                self.q[i] = msg.position[
                    name_to_index[joint]
                ]

        ee, _ = self.fk(self.q)

        print(
            f"EE Position: "
            f"[{ee[0]:.2f}, "
            f"{ee[1]:.2f}, "
            f"{ee[2]:.2f}]",
            flush=True
        )

        self.have_joint_state = True

    # =========================================================
    # TARGET CALLBACK
    # =========================================================

    def target_cb(self, msg):

        if not self.have_joint_state:

            self.get_logger().warn(
                'Waiting for joint states...'
            )

            return

        target = np.array([
            msg.x,
            msg.y,
            msg.z
        ])

        self.get_logger().info(
            f'Target: '
            f'[{msg.x:.2f}, '
            f'{msg.y:.2f}, '
            f'{msg.z:.2f}]'
        )

        # =====================================================
        # IMPORTANT:
        # CONTINUOUS LOCAL IK
        # =====================================================

        q_sol = self.compute_ik(
            self.q.copy(),
            target
        )

        # =====================================================
        # PUBLISH
        # =====================================================

        out = Float64MultiArray()

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
            f'Published: '
            f'{[round(math.degrees(v),1) for v in q_sol]} deg'
        )

    # =========================================================
    # FORWARD KINEMATICS
    # =========================================================

    def fk(self, q):

        T = np.eye(4)

        # BASE
        T = T @ self.trans(0, 0, 0.1)
        T = T @ self.rotz(q[0])

        # LINK 1
        T = T @ self.roty(q[1])
        T = T @ self.trans(0, 0, 1.67744)

        # LINK 2
        T = T @ self.roty(q[2])
        T = T @ self.trans(0, 0, 2.92735)

        # LINK 3
        T = T @ self.roty(q[3])
        T = T @ self.trans(0, 0, 2.9244)

        # LINK 4
        T = T @ self.roty(q[4])
        T = T @ self.trans(0, 0, 2.12653)

        position = T[:3, 3]

        rotation = T[:3, :3]

        return position, rotation

    # =========================================================
    # FK FRAMES
    # =========================================================

    def fk_frames(self, q):

        T = np.eye(4)

        origins = []

        axes = []

        # BASE

        T = T @ self.trans(0, 0, 0.1)

        origins.append(
            T[:3, 3].copy()
        )

        axes.append(
            T[:3, :3] @ np.array([0, 0, 1])
        )

        T = T @ self.rotz(q[0])

        # JOINT 1

        origins.append(
            T[:3, 3].copy()
        )

        axes.append(
            T[:3, :3] @ np.array([0, 1, 0])
        )

        T = T @ self.roty(q[1])

        T = T @ self.trans(
            0,
            0,
            1.67744
        )

        # JOINT 2

        origins.append(
            T[:3, 3].copy()
        )

        axes.append(
            T[:3, :3] @ np.array([0, 1, 0])
        )

        T = T @ self.roty(q[2])

        T = T @ self.trans(
            0,
            0,
            2.92735
        )

        # JOINT 3

        origins.append(
            T[:3, 3].copy()
        )

        axes.append(
            T[:3, :3] @ np.array([0, 1, 0])
        )

        T = T @ self.roty(q[3])

        T = T @ self.trans(
            0,
            0,
            2.9244
        )

        # JOINT 4

        origins.append(
            T[:3, 3].copy()
        )

        axes.append(
            T[:3, :3] @ np.array([0, 1, 0])
        )

        T = T @ self.roty(q[4])

        T = T @ self.trans(
            0,
            0,
            2.12653
        )

        return T[:3, 3], origins, axes

    # =========================================================
    # POSITION JACOBIAN
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

            # =================================================
            # CURRENT FK
            # =================================================

            current, rotation = self.fk(q)

            # =================================================
            # POSITION ERROR
            # =================================================

            pos_error = target - current

            # =================================================
            # REAL END EFFECTOR DIRECTION
            # =================================================

            ee_z = rotation[:, 2]

            # =================================================
            # DESIRED TOOL DIRECTION
            # =================================================

            desired_down = np.array([
                0.0,
                0.0,
                -1.0
            ])

            # =================================================
            # ORIENTATION ERROR
            # =================================================

            orient_error = np.cross(
                ee_z,
                desired_down
            )

            # =================================================
            # COMBINED ERROR
            # =================================================

            orientation_weight = 0.15

            error = np.concatenate([
                pos_error,
                orientation_weight
                * orient_error[0:1]
            ])

            # =================================================
            # CONVERGENCE
            # =================================================

            error_norm = np.linalg.norm(
                pos_error
            )

            print(
                f"Iter {i} | "
                f"Pos Error: "
                f"{error_norm:.4f}",
                flush=True
            )

            if error_norm < self.tolerance:

                print(
                    "IK CONVERGED",
                    flush=True
                )

                break

            # =================================================
            # POSITION JACOBIAN
            # =================================================

            J_pos = self.jacobian(q)

            # =================================================
            # ORIENTATION JACOBIAN
            # =================================================

            J_orient = np.array([
                [0, 1, 1, 1, 1]
            ])

            # =================================================
            # FULL JACOBIAN
            # =================================================

            J = np.vstack([
                J_pos,
                orientation_weight
                * J_orient
            ])

            # =================================================
            # DAMPED LEAST SQUARES
            # =================================================

            lhs = (
                J @ J.T
                + (self.damping ** 2)
                * np.eye(4)
            )

            dq = J.T @ np.linalg.solve(
                lhs,
                error
            )

            # =================================================
            # SMALL LOCAL MOTION
            # =================================================

            norm = np.linalg.norm(dq)

            if norm > self.max_dq:

                dq *= (
                    self.max_dq / norm
                )

            # =================================================
            # CONTINUOUS LOCAL UPDATE
            # =================================================

            q += self.alpha * dq

            # =================================================
            # JOINT LIMITS
            # =================================================

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

    node = IKNode()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':

    main()