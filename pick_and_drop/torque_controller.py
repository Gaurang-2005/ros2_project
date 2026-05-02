import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float64


class DirectTorque(Node):
    def __init__(self):
        super().__init__('direct_torque')

        self.joint_names = [
            'base_joint','joint1','joint2','joint3','joint4',
            'finger1_joint','finger2_joint'
        ]

        # publishers
        self.pubs = []
        for j in self.joint_names:
            topic = f'/model/pick_and_drop_arm/joint/{j}/cmd_force'
            self.pubs.append(self.create_publisher(Float64, topic, 10))

        # subscriber
        self.create_subscription(
            Float64MultiArray,
            '/joint_torque',
            self.torque_cb,
            10
        )

        self.get_logger().info("Direct torque node ready")

    def torque_cb(self, msg):
        self.get_logger().info(f"Received: {msg.data}")
        data = msg.data

        # safety: length check
        if len(data) != len(self.joint_names):
            self.get_logger().warn("Wrong torque array size")
            return

        for i in range(len(self.joint_names)):
            t = float(data[i])

            msg_out = Float64()
            msg_out.data = t
            self.pubs[i].publish(msg_out)


def main():
    rclpy.init()
    node = DirectTorque()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()