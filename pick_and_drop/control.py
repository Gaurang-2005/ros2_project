#!/usr/bin/env python3
"""
control.py  —  Pick-and-Place Controller for pick_and_drop arm
Fixes: arm glitching (removed hold_position spam) + box snapping (physics pause)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose
import subprocess
import math
import time
import sys
import os
import tempfile

from pick_and_drop.arm_ik import get_pose_for_target, FALLBACK_POSES, JOINT_LIMITS

JOINT_TOPICS = {
    "base_joint"   : "/base_joint_cmd",
    "joint1"       : "/joint1_cmd",
    "joint2"       : "/joint2_cmd",
    "joint3"       : "/joint3_cmd",
    "joint4"       : "/joint4_cmd",
    "finger1_joint": "/finger1_cmd",
    "finger2_joint": "/finger2_cmd",
}

BOX_DEFAULT_X = 4.0
BOX_DEFAULT_Y = 0.0
BOX_DEFAULT_Z = 0.5

BOX_SDF = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="box">
    <static>false</static>
    <link name="box_link">
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.167</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>0.167</iyy><iyz>0</iyz><izz>0.167</izz>
        </inertia>
      </inertial>
      <collision name="col">
        <geometry><box><size>1.0 1.0 1.0</size></box></geometry>
        <surface>
          <friction><ode><mu>0.8</mu><mu2>0.8</mu2></ode></friction>
        </surface>
      </collision>
      <visual name="vis">
        <geometry><box><size>1.0 1.0 1.0</size></box></geometry>
        <material>
          <ambient>0.8 0.1 0.1 1</ambient>
          <diffuse>0.9 0.1 0.1 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


def ign_pause_physics():
    subprocess.run([
        "ign", "service", "-s", "/world/empty/control",
        "--reqtype", "ignition.msgs.WorldControl",
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", "3000", "--req", "pause: true"
    ], capture_output=True)


def ign_resume_physics():
    subprocess.run([
        "ign", "service", "-s", "/world/empty/control",
        "--reqtype", "ignition.msgs.WorldControl",
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", "3000", "--req", "pause: false"
    ], capture_output=True)


class ArmController(Node):

    def __init__(self):
        super().__init__('arm_controller')

        self.declare_parameter('drop_x', 0.0)
        self.declare_parameter('drop_y', 4.0)
        self.declare_parameter('drop_z', 0.5)
        self.declare_parameter('motion_steps', 80)
        self.declare_parameter('step_delay', 0.02)

        self.drop_x = self.get_parameter('drop_x').value
        self.drop_y = self.get_parameter('drop_y').value
        self.drop_z = self.get_parameter('drop_z').value
        self.steps  = self.get_parameter('motion_steps').value
        self.dt     = self.get_parameter('step_delay').value

        self.pubs = {}
        for joint, topic in JOINT_TOPICS.items():
            self.pubs[joint] = self.create_publisher(Float64, topic, 10)

        self.box_x = BOX_DEFAULT_X
        self.box_y = BOX_DEFAULT_Y
        self.box_z = BOX_DEFAULT_Z
        self.pose_sub = self.create_subscription(
            Pose, '/model/box/pose', self._box_pose_cb, 10
        )

        self.current_angles = {j: 0.0 for j in JOINT_TOPICS}
        self.get_logger().info("✅ ArmController node started")

    def _box_pose_cb(self, msg: Pose):
        self.box_x = msg.position.x
        self.box_y = msg.position.y
        self.box_z = msg.position.z

    def publish_angles(self, angles: dict):
        for joint, angle in angles.items():
            if joint in self.pubs:
                msg = Float64()
                msg.data = float(angle)
                self.pubs[joint].publish(msg)

    def move_to_pose(self, target: dict, label: str = ""):
        """
        Smooth cosine-eased motion to target.
        No continuous republishing — JointPositionController holds position internally.
        """
        if label:
            self.get_logger().info(f"  ▶ {label}")

        start = dict(self.current_angles)

        for step in range(self.steps + 1):
            t = step / self.steps
            alpha = (1 - math.cos(math.pi * t)) / 2
            blended = {
                joint: start.get(joint, 0.0) + alpha * (target.get(joint, start.get(joint, 0.0)) - start.get(joint, 0.0))
                for joint in JOINT_TOPICS
            }
            self.publish_angles(blended)
            rclpy.spin_once(self, timeout_sec=0)
            time.sleep(self.dt)

        # Publish final position a few times to confirm
        for _ in range(5):
            self.publish_angles(target)
            rclpy.spin_once(self, timeout_sec=0)
            time.sleep(0.02)

        self.current_angles = dict(target)
        time.sleep(0.3)   # physics settle — no republishing

    def spawn_box(self):
        self.get_logger().info(f"📦 Spawning box at ({BOX_DEFAULT_X}, {BOX_DEFAULT_Y}, {BOX_DEFAULT_Z})...")

        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False)
        tmp.write(BOX_SDF)
        tmp.flush()
        tmp.close()
        sdf_file = tmp.name

        cmd = [
            "ign", "service", "-s", "/world/empty/create",
            "--reqtype", "ignition.msgs.EntityFactory",
            "--reptype", "ignition.msgs.Boolean",
            "--timeout", "5000",
            "--req",
            f'sdf_filename: "{sdf_file}" name: "box" '
            f'pose: {{ position: {{ x: {BOX_DEFAULT_X} y: {BOX_DEFAULT_Y} z: {BOX_DEFAULT_Z} }} }}'
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.get_logger().info("✅ Box spawned successfully")
            else:
                self.get_logger().warn(f"⚠️  Box spawn: {result.stderr}")
        except Exception as e:
            self.get_logger().error(f"❌ Spawn failed: {e}")

        time.sleep(2.0)
        try:
            os.unlink(sdf_file)
        except Exception:
            pass

    def run_pick_and_place(self):
        bx, by, bz = self.box_x, self.box_y, self.box_z
        dx, dy, dz = self.drop_x, self.drop_y, self.drop_z

        self.get_logger().info(
            f"\n{'='*50}\n  PICK → ({bx:.2f}, {by:.2f}, {bz:.2f})\n"
            f"  DROP → ({dx:.2f}, {dy:.2f}, {dz:.2f})\n{'='*50}"
        )

        self.move_to_pose(FALLBACK_POSES["home"], "HOME")

        pose, ik = get_pose_for_target(bx, by, bz + 1.5, "pre_pick", False)
        self.move_to_pose(pose, f"PRE-PICK ({'IK' if ik else 'fallback'})")

        pose, ik = get_pose_for_target(bx, by, bz + 0.3, "pick", False)
        self.move_to_pose(pose, f"PICK ({'IK' if ik else 'fallback'})")

        grip = dict(pose)
        grip["finger1_joint"] = 0.70
        grip["finger2_joint"] = 0.70
        self.move_to_pose(grip, "GRIP")
        time.sleep(0.5)

        pose, ik = get_pose_for_target(bx, by, bz + 2.5, "pre_pick", True)
        pose["finger1_joint"] = 0.70
        pose["finger2_joint"] = 0.70
        self.move_to_pose(pose, f"LIFT ({'IK' if ik else 'fallback'})")

        pose, ik = get_pose_for_target(dx, dy, dz + 1.5, "pre_pick", True)
        pose["finger1_joint"] = 0.70
        pose["finger2_joint"] = 0.70
        self.move_to_pose(pose, f"PRE-PLACE ({'IK' if ik else 'fallback'})")

        pose, ik = get_pose_for_target(dx, dy, dz + 0.3, "pick", True)
        pose["finger1_joint"] = 0.70
        pose["finger2_joint"] = 0.70
        self.move_to_pose(pose, f"PLACE ({'IK' if ik else 'fallback'})")

        release = dict(pose)
        release["finger1_joint"] = 0.0
        release["finger2_joint"] = 0.0
        self.move_to_pose(release, "RELEASE")
        time.sleep(0.5)

        self.move_to_pose(FALLBACK_POSES["home"], "HOME")
        self.get_logger().info("✅ Pick-and-place complete!")


def main(args=None):
    rclpy.init(args=args)
    node = ArmController()

    print("\n" + "="*55)
    print("  🤖  pick_and_drop — Control Node")
    print("="*55)

    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.05)

    print("\n[1/4] Moving arm to HOME position...")
    node.move_to_pose(FALLBACK_POSES["home"], "HOME")

    print(f"\n[2/4] Spawning red box...")
    node.spawn_box()

    print("      Waiting for box pose from bridge...")
    deadline = time.time() + 5.0
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    # Pause physics so user can freely move the box
    print("\n[3/4] ⏸️  Pausing physics — box is now FREE to move!")
    ign_pause_physics()

    print("\n      📦 Box spawned!")
    print("      ➜  Drag it anywhere in Gazebo GUI (physics is paused, it won't snap back).")
    print("      ➜  Press  ENTER  when ready.\n")

    input("      >> Press ENTER to start pick-and-place... ")

    print("\n      ▶️  Resuming physics...")
    ign_resume_physics()

    print("      ⏳ Starting in 2 seconds...")
    deadline = time.time() + 2.0
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

    rclpy.spin_once(node, timeout_sec=0.2)
    print(f"\n      📍 Box at:  x={node.box_x:.3f}  y={node.box_y:.3f}  z={node.box_z:.3f}")
    print(f"      🎯 Drop at: x={node.drop_x:.3f}  y={node.drop_y:.3f}  z={node.drop_z:.3f}\n")

    print("[4/4] 🚀 Executing pick-and-place...\n")
    node.run_pick_and_place()

    print("\n" + "="*55)
    print("  ✅  Done!")
    print("="*55 + "\n")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()