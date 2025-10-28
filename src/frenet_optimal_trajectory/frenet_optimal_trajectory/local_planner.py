#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math, numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Point, PointStamped
from visualization_msgs.msg import Marker

from coordinate.src.coordinate import coordinate_converter
from cubic_spline_planner import CubicSpline1D
from quintic_polynomials_planner import QuinticPolynomial


class LocalPlanner(Node):
    def __init__(self):
        super().__init__("frenet_dwa_planner")
        qos = QoSProfile(depth=1,
                         reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)

        # ===== Parameters =====
        self.frame_id = str(self.declare_parameter("frame_id", "map").value)
        plan_hz = float(self.declare_parameter("plan_hz", 20.0).value)
        self.MAX_SPEED = float(self.declare_parameter("MAX_SPEED", 9.0).value)
        self.MAX_ROAD_WIDTH = float(self.declare_parameter("MAX_ROAD_WIDTH", 7.0).value)
        self.ROBOT_RADIUS = float(self.declare_parameter("ROBOT_RADIUS", 2.0).value)
        self.FOOTPRINT_PADDING = float(self.declare_parameter("FOOTPRINT_PADDING", 0.0).value)

        # ===== I/O =====
        self.converter = coordinate_converter(self)
        self.sub_odom = self.create_subscription(Odometry, "/odom", self.cb_odom, qos)
        self.sub_static = self.create_subscription(Odometry, "/static_odom", self.cb_static, qos)
        self.sub_dynamic = self.create_subscription(Odometry, "/dynamic_odom", self.cb_dynamic, qos)
        self.sub_flag = self.create_subscription(PointStamped, "/obstacle_flag", self.cb_flag, qos)
        
        self.pub_path = self.create_publisher(Path, "/Path", 1)
        self.marker_pub = self.create_publisher(Marker, "/local_planner/markers", 10)

        # ===== State =====
        self.odom = None
        self.ego_s = self.ego_d = None
        self.static_xy = None
        self.dynamic_xy = None
        self.dynamic_v = None
        self.flag_dynamic = 0
        self.flag_static = 0
        self.priority = 0
        self.mode = 0  # 0=normal, 1=static, 2=dynamic, 3=return

        self.create_timer(1.0 / plan_hz, self.planner)
        self.get_logger().info("LocalPlanner (with reference spline correction).")

    # ------------------- Callbacks -------------------
    def cb_odom(self, msg):
        self.odom = msg
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.ego_s, self.ego_d = self.converter.global_to_frenet_point(x, y)

    def cb_static(self, msg):
        self.static_xy = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y], dtype=float)

    def cb_dynamic(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.dynamic_xy = np.array([x, y], dtype=float)
        self.dynamic_v = np.array([vx, vy], dtype=float)

    def cb_flag(self, msg: PointStamped):
        self.flag_dynamic = int(msg.point.x)
        self.flag_static = int(msg.point.y)
        self.priority = int(msg.point.z)

    # ------------------- 회피 조건 함수들 -------------------
    def should_avoid(self, obs_s, ego_s, ego_v, obs_v, flag):
        if not flag:
            return False
        relative_s = obs_s - ego_s
        closing_speed = ego_v - obs_v
        if relative_s < 0:
            return False
        if relative_s < 5.0 or (closing_speed > 0 and relative_s / max(closing_speed, 0.01) < 1.5):
            return True
        return False

    def is_obstacle_cleared(self):
        cleared = False
        for obs in [self.static_xy, self.dynamic_xy]:
            if obs is None:
                continue
            s_obs, _ = self.converter.global_to_frenet_point(*obs)
            if s_obs and s_obs - self.ego_s < 0:
                cleared = True
        return cleared

    def is_path_safe(self, path, obstacles, min_clear=0.5):
        for (x, y) in path:
            for (ox, oy) in obstacles:
                if math.hypot(x - ox, y - oy) < (self.ROBOT_RADIUS + min_clear):
                    self.get_logger().warn("[SAFETY] Path too close to obstacle!")
                    return False
        return True

    # ------------------- Reference 보정 spline -------------------
    def generate_ref_spline_path(self, forward_length=8.0):
        """ego_d ≠ 0일 때 global 기준으로 복원 spline"""
        s_points = np.linspace(self.ego_s, self.ego_s + forward_length, 40)
        qp = QuinticPolynomial(self.ego_d, 0.0, 0.0, 0.0, 0.0, 0.0, forward_length / 2.0)
        d_vals = [qp.calc_point((s - self.ego_s) / (forward_length / 2.0)) for s in s_points]
        path = [self.converter.frenet_to_global_point(s, d)
                for s, d in zip(s_points, d_vals)]
        return path

    def return_to_center_path(self):
        s_points = np.linspace(self.ego_s, self.ego_s + 4.0, 40)
        qp = QuinticPolynomial(self.ego_d, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0)
        d_vals = [qp.calc_point((s - self.ego_s) / 4.0) for s in s_points]
        path = [self.converter.frenet_to_global_point(s, d)
                for s, d in zip(s_points, d_vals)]
        return path

    # ------------------- Avoidance functions -------------------
    def static_avoidance(self):
        if self.static_xy is None:
            return None
        s_obs, d_obs = self.converter.global_to_frenet_point(*self.static_xy)
        if s_obs is None:
            return None
        offset = 1.2
        if abs(d_obs + offset) > self.MAX_ROAD_WIDTH / 2.0:
            return None
        s_points = [self.ego_s, s_obs - 1.0, s_obs, s_obs + 2.0, s_obs + 4.0]
        qp = QuinticPolynomial(self.ego_d, 0.0, 0.0, d_obs + offset, 0.0, 0.0, 3.0)
        d_vals = [qp.calc_point((s - self.ego_s) / 3.0) for s in s_points]
        spline_d = CubicSpline1D(np.array(s_points), np.array(d_vals))
        s_samples = np.linspace(self.ego_s, s_points[-1], 50)
        path = [self.converter.frenet_to_global_point(s, spline_d.calc_position(s))
                for s in s_samples]
        if not self.is_path_safe(path, [self.static_xy]):
            return None
        return path

    def dynamic_behavior(self):
        if self.dynamic_xy is None or self.dynamic_v is None:
            return None
        opp_speed = math.hypot(*self.dynamic_v)
        if opp_speed > 2.0:
            return None
        horizon = 2.0
        pred_x = self.dynamic_xy[0] + self.dynamic_v[0] * horizon
        pred_y = self.dynamic_xy[1] + self.dynamic_v[1] * horizon
        pred_s, pred_d = self.converter.global_to_frenet_point(pred_x, pred_y)
        if pred_s is None:
            return None
        offset = 1.0
        if abs(pred_d + offset) > self.MAX_ROAD_WIDTH / 2.0:
            return None
        s_points = [self.ego_s, pred_s - 1.0, pred_s, pred_s + 2.0, pred_s + 4.0]
        qp = QuinticPolynomial(self.ego_d, 0.0, 0.0, pred_d + offset, 0.0, 0.0, 3.0)
        d_vals = [qp.calc_point((s - self.ego_s) / 3.0) for s in s_points]
        spline_d = CubicSpline1D(np.array(s_points), np.array(d_vals))
        s_samples = np.linspace(self.ego_s, s_points[-1], 50)
        path = [self.converter.frenet_to_global_point(s, spline_d.calc_position(s))
                for s in s_samples]
        if not self.is_path_safe(path, [self.dynamic_xy]):
            return None
        return path

    # ------------------- Main Planner -------------------
    def planner(self):
        if self.odom is None or not self.converter.path_recived:
            return

        ego_v = self.odom.twist.twist.linear.x
        dyn_v = math.hypot(*(self.dynamic_v or [0.0, 0.0]))
        stat_s, _ = self.converter.global_to_frenet_point(*self.static_xy) if self.static_xy is not None else (None, None)
        dyn_s, _ = self.converter.global_to_frenet_point(*self.dynamic_xy) if self.dynamic_xy is not None else (None, None)

        static_cond = bool(self.flag_static)
        dynamic_cond = self.should_avoid(dyn_s or 0, self.ego_s or 0, ego_v, dyn_v, self.flag_dynamic)

        static_p = self.static_avoidance() if static_cond else None
        dynamic_p = self.dynamic_behavior() if dynamic_cond else None
        recover_p = self.return_to_center_path() if self.is_obstacle_cleared() and abs(self.ego_d) > 0.3 else None
        ref_corrected = self.generate_ref_spline_path() if self.mode == 0 and abs(self.ego_d) > 0.1 else None

        # 모드 결정
        if recover_p:
            selected_path, path_type, self.mode = recover_p, 0, 3
        elif static_cond and (self.priority == 0 or not dynamic_cond):
            selected_path, path_type, self.mode = static_p or ref_corrected, 1, 1
        elif dynamic_cond and (self.priority == 1 or not static_cond):
            selected_path, path_type, self.mode = dynamic_p or ref_corrected, 2, 2
        else:
            selected_path, path_type, self.mode = ref_corrected or self.generate_ref_spline_path(), 0, 0

        # visualization
        self.publish_path_colored(selected_path, (1, 1, 1, 0.7), "active_path")

        # Publish
        msg = Path()
        msg.header.frame_id = self.frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        for (x, y) in selected_path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.orientation.z = float(path_type)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.pub_path.publish(msg)

    # ------------------- Visualization -------------------
    def publish_path_colored(self, path_points, color_rgba, ns):
        if not path_points:
            return
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = hash(ns) % 65536
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.07
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color_rgba
        for (x, y) in path_points:
            p = Point()
            p.x, p.y, p.z = x, y, 0.0
            marker.points.append(p)
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = LocalPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
