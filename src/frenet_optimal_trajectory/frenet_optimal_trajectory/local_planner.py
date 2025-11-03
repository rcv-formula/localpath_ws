#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math, numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, Point, PointStamped
from visualization_msgs.msg import Marker

from .coordinate import coordinate_converter
from cubic_spline_planner.cubic_spline_planner import CubicSpline2D
from quintic_polynomials_planner.quintic_polynomials_planner import QuinticPolynomial


class LocalPlanner(Node):
    def __init__(self):
        super().__init__("frenet_dwa_planner")
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        # ===== Parameters =====
        self.frame_id = str(self.declare_parameter("frame_id", "map").value)
        plan_hz = float(self.declare_parameter("plan_hz", 20.0).value)
        self.MAX_SPEED = float(self.declare_parameter("MAX_SPEED", 4.5).value)
        self.MAX_ROAD_WIDTH = float(self.declare_parameter("MAX_ROAD_WIDTH", 2.0).value)
        self.ROBOT_RADIUS = float(self.declare_parameter("ROBOT_RADIUS", 0.3).value)
        self.FOOTPRINT_PADDING = float(self.declare_parameter("FOOTPRINT_PADDING", 0.0).value)

        # Frenet target d samples (dynamic behavior 사용)
        self.FOT_D_SAMPLES = np.linspace(-0.2, 0.4, 5)

        # 비용 가중치
        self.W_SAFETY = 100.0
        self.W_LATERAL = 1.0
        self.W_JERK_D = 0.5
        self.W_DYNAMIC = 50.0

        self.MAX_LATERAL_ACCEL = 3.0  # m/s^2
        self.MAX_ACCEL = 2.0          # m/s^2
        self.MAX_DECEL = -5.0         # m/s^2

        self.W_AVG_SPEED = 10.0
        self.W_TIME = 1.0

        # ===== I/O =====
        self.converter = coordinate_converter(self)
        self.sub_odom    = self.create_subscription(Odometry, "/odom", self.cb_odom, qos)
        self.sub_static  = self.create_subscription(PointStamped, "/static_obstacle", self.cb_static, qos)
        self.sub_dynamic = self.create_subscription(Odometry, "/dynamic_obstacle", self.cb_dynamic, qos)
        self.sub_flag    = self.create_subscription(PointStamped, "/obj_flag", self.cb_flag, qos)

        self.pub_path   = self.create_publisher(Path, "/Path", 1)
        self.marker_pub = self.create_publisher(Marker, "/local_path", 1)

        # ===== State =====
        self.odom = None
        self.x = self.y = 0.0
        self.ego_s = self.ego_d = None
        self.static_xy = None
        self.dynamic_xy = None
        self.static_s, self.static_d = None, None
        self.dynamic_v = None
        self.flag_dynamic = 0
        self.flag_static = 0
        self.priority = 0
        self.mode = 0  # 0=normal, 1=static, 2=dynamic, 3=return, 4=acc
        self.forward_length = 5.0

        self.global_path = None
        self.global_frenet_path = None
        self.path_length = 0

        self.selected_path = None
        self.selected_path_frenet = None

        self.DEBUG_PATH_LOG = True
        self._JUMP_THRESH = 2.0

        self._gp_cache_ready = False

        self._active_idx_list = []
        self._active_idx_set  = set()

        self.can_go_left = False
        self.can_go_right = False

        self.create_timer(1.0 / plan_hz, self.planner)

    # ------------------- Helpers -------------------
    def _yaw_from_quat(self, q):
        x, y, z, w = q.x, q.y, q.z, q.w
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(t3, t4)

    def _nearest_idx_and_tangent(self, gp, px, py):
        xs = np.array([p[0] for p in gp], dtype=float)
        ys = np.array([p[1] for p in gp], dtype=float)
        i = int(np.argmin((xs - px) ** 2 + (ys - py) ** 2))
        i0 = max(0, i - 1)
        i1 = min(len(gp) - 1, i + 1)
        vx = gp[i1][0] - gp[i0][0]
        vy = gp[i1][1] - gp[i0][1]
        n = math.hypot(vx, vy) or 1.0
        return (vx / n, vy / n)

    def _heading_dir_sign(self):
        gp = getattr(self.converter, "global_path", None)
        if self.odom is None or not gp or len(gp) < 3:
            return +1.0
        yaw = self._yaw_from_quat(self.odom.pose.pose.orientation)
        ex, ey = math.cos(yaw), math.sin(yaw)
        px = self.odom.pose.pose.position.x
        py = self.odom.pose.pose.position.y
        tx, ty = self._nearest_idx_and_tangent(gp, px, py)
        return +1.0 if (ex * tx + ey * ty) >= 0.0 else -1.0

    # ------------------- Callbacks -------------------
    def cb_odom(self, msg):
        self.odom = msg
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.ego_s, d = self.converter.global_to_frenet_point(self.x, self.y)
        self.ego_d = d

    def cb_static(self, msg):
        self.x_obs, self.y_obs = msg.point.x, msg.point.y
        self.static_xy = np.array([msg.point.x, msg.point.y], dtype=float)
        self.static_s, self.static_d = self.converter.global_to_frenet_point(msg.point.x, msg.point.y)

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

    def wall_safe(self, idx,
                  safety_margin: float = 0.2,
                  dir_sign: float = +1.0):
        r, l = self.converter.get_wall_distance(idx)  # (right, left)
        r, l = float(r), float(l)
        if dir_sign < 0:
            r, l = l, r
        r = max(0.0, r)
        l = max(0.0, l)

        dmin = -l + safety_margin
        dmax = r - safety_margin

        # if dmax - dmin < min_width:
        #     mid = 0.5 * (dmin + dmax)
        #     half = 0.5 * min_width
        #     dmin, dmax = mid - half, mid + half

        return dmin, dmax, r, l

    def safe_path(self, path, idx_list):
        if not path:
            return []
        frenet_path = self.converter.global_to_frenet(path)  # Nx3 [s,d,v]
        if not frenet_path:
            return []

        fr_np = np.asarray(frenet_path, dtype=float)  # (N,3)
        d_vals = fr_np[:, 1]

        dmins = np.empty_like(d_vals)
        dmaxs = np.empty_like(d_vals)
        for k, gi in enumerate(idx_list):
            dmin, dmax, _, _ = self.wall_safe(int(gi))
            dmins[k] = dmin
            dmaxs[k] = dmax

        fr_np[:, 1] = np.clip(d_vals, dmins, dmaxs)

        safe_path = self.converter.frenet_to_global(fr_np.tolist())
        return safe_path

    def get_closest_index(self, x, y, path):
        if len(path) == 0:
            self.get_logger().warn("Cannot find closest index: Global path is empty!")
            return 0
        idx = 0
        
        closest_dist = self.converter._calc_distance([x, y], path[0][:2])
        for i in range(1, len(path)):
            dist = self.converter._calc_distance([x, y], path[i][:2])
            if dist < closest_dist:
                idx = i
                closest_dist = dist
        return idx

    def generate_global_path(self):
        if not self.global_path:
            self.get_logger().warn("[RefPath] Cannot generate: global_path is empty.")
            return []

        path_size = len(self.global_path)
        if path_size == 0:
            self.get_logger().warn("[RefPath] Path size is zero after guard.")
            return []

        cur_idx = self.converter._get_closest_index(self.x, self.y)
        cur_idx = int(cur_idx % path_size)

        span = min(30, path_size)  # lookahead 길이

        idx_list = [(cur_idx + offset) % path_size for offset in range(span)]
        self._active_idx_list = idx_list
        self._active_idx_set  = set(idx_list)

        self.get_logger().info(f"[RefPath] Generating frenet path: start={cur_idx}, span={span}")

        global_path = [self.global_path[i] for i in idx_list]
        if len(global_path) == 0:
            self._active_idx_list = []
            self._active_idx_set  = set()
            self.get_logger().warn("PATH EMPTY in safe_path!")
            return []

        safe_global_path = self.safe_path(global_path, idx_list)
        if not safe_global_path:
            self.get_logger().warn("[RefPath] safe_path produced empty output.")
            return []

        safe_global_frenet_path = self.converter.global_to_frenet(safe_global_path)
        self.selected_path_frenet = np.asarray(safe_global_frenet_path, dtype=float)
        self.selected_path = np.asarray(safe_global_path, dtype=float)
        return safe_global_path

    def apply_d_bump(self, base_d_points, obs_s,
                            bump, half_width_idx, side, smooth=True):
        d_points = np.copy(base_d_points)
        i_center = int(obs_s)
        i0 = max(0, i_center - half_width_idx)
        i1 = min(d_points.size - 1, i_center + half_width_idx)
        if i1 < i0:
            return d_points

        n = int(i1 - i0 + 1)
        if smooth and n > 1:
            win = 0.5 * (1.0 + np.cos(np.linspace(-(math.pi), (math.pi), n)))
        else:
            win = np.ones(n)

        signed = float(side) * float(bump)
        d_points[i0: i1 + 1] += signed * win
        return d_points

    def static_avoidance(self, selected_path, stat_s, stat_d, margin=0.3):
        if selected_path is None or len(selected_path) == 0:
            return None

        frenet_path = self.selected_path_frenet
        if frenet_path is None or len(frenet_path) == 0:
            self.get_logger().warn("[STATIC] Frenet snippet missing or length mismatch.")
            return None

        if not getattr(self, "_active_idx_set", None):
            return None

        try:
            if getattr(self, "x_obs", None) is not None and getattr(self, "y_obs", None) is not None:
                obs_gidx = self.converter._get_closest_index(self.x_obs, self.y_obs)
            else:
                xo, yo = self.converter.frenet_to_global_point(float(stat_s), float(stat_d))
                obs_gidx = self.converter._get_closest_index(xo, yo)
        except Exception:
            return None

        if obs_gidx not in self._active_idx_set:
            self.get_logger().warn("no need to avoid: obs outside active path.")
            return None

        s_obs, d_obs = float(stat_s), float(stat_d)
        x_obs, y_obs = self.converter.frenet_to_global_point(s_obs, d_obs)

        obs_idx = self.get_closest_index(x_obs, y_obs, selected_path)

        d_min, d_max, right_distance, left_distance = self.wall_safe(obs_gidx)

        left_target  = d_obs - margin
        right_target = d_obs + margin

        if left_target < d_min:
            left_target = d_min

        if right_target > d_max:
            right_target = d_max

        if right_distance > left_distance:
            side = -1.0
            d_target = right_target
        elif left_distance > right_distance:
            side = +1.0
            d_target = left_target
        else:
            self.get_logger().warn("No safe side to avoid static obstacle.")
            return None

        s_points = np.asarray(frenet_path[:, 0], dtype=float)
        base_d   = np.asarray(frenet_path[:, 1], dtype=float)

        bump_mag = float(d_target - d_obs)

        d_mod = self.apply_d_bump(
            base_d, obs_idx, bump=bump_mag,
            half_width_idx=5, side=side, smooth=True
        )

        static_path = []
        for s, d in zip(s_points, d_mod):
            x, y = self.converter.frenet_to_global_point(s, d)
            v_ref = self.converter.get_velocity_at_s(s)
            static_path.append([x, y, v_ref])

        if len(static_path) == 0:
            self.get_logger().warn("static_path is empty after avoidance.")
            return None

        return static_path

    def planner(self):
        if self.odom is None or not self.converter.path_recived:
            return
        
        if not self._gp_cache_ready:
            self.global_path = self.converter.get_global_path()
            if self.global_path:
                self.global_frenet_path = self.converter.global_to_frenet(self.global_path)
                self.path_length = self.converter.get_path_length()
                self._gp_cache_ready = True
            else:
                return

        default_path = self.generate_global_path()
        if not default_path:
            self.get_logger().warn("[Planner] default_path is empty; skip this cycle.")
            return

        static_cond = bool(self.flag_static)
        dynamic_cond = bool(self.flag_dynamic)

        static_p = self.static_avoidance(
            self.selected_path.tolist() if isinstance(self.selected_path, np.ndarray) else self.selected_path,
            self.static_s, self.static_d
        ) if static_cond else None

        if static_cond:
            if static_p:
                selected_path, path_type, self.mode = static_p, 1, 1
                self.get_logger().info("[Planner] Static avoidance path selected.")
            else:
                selected_path, path_type, self.mode = default_path, 0, 0
        elif dynamic_cond and (self.priority == 1 or not static_cond):
            selected_path, path_type, self.mode = default_path, 0, 4
        else:
            selected_path, path_type, self.mode = default_path, 0, 0


        selected_xy = [(p[0], p[1]) for p in (selected_path or [])]
        self.publish_path_colored(selected_xy, (0.0, 1.0, 0.0, 0.5), "active_path")

        if selected_path:
            msg = Path()
            msg.header.frame_id = self.frame_id
            msg.header.stamp = self.get_clock().now().to_msg()
            for (x, y, v) in selected_path:
                ps = PoseStamped()
                ps.header = msg.header
                ps.pose.position.x = x
                ps.pose.position.y = y
                ps.pose.position.z = v
                ps.pose.orientation.z = float(self.mode)  # 0:ref/보정, 1:static, 2:dynamic
                ps.pose.orientation.w = 1.0
                msg.poses.append(ps)
            self.pub_path.publish(msg)

    def publish_path_colored(self, path_points, color_rgba, ns):
        if not path_points:
            return
        JUMP_THRESH = self._JUMP_THRESH

        segments, cur = [], []
        for (x, y) in path_points:
            if cur:
                px, py = cur[-1]
                if math.hypot(x - px, y - py) > JUMP_THRESH:
                    if len(cur) >= 2:
                        segments.append(cur)
                    cur = []
            cur.append((x, y))
        if len(cur) >= 2:
            segments.append(cur)

        for k, seg in enumerate(segments):
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = ns
            marker.id = (hash(ns) + k) % 65536
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.07
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = color_rgba
            for (x, y) in seg:
                p = Point(); p.x, p.y, p.z = x, y, 0.0
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
