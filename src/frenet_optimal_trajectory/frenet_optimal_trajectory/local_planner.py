#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math, numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration
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
        self.sub_odom = self.create_subscription(Odometry, "/odom", self.cb_odom, qos)
        self.sub_static = self.create_subscription(PointStamped, "/static_obstacle", self.cb_static, qos)
        self.sub_dynamic = self.create_subscription(Odometry, "/dynamic_obstacle", self.cb_dynamic, qos)
        self.sub_flag = self.create_subscription(PointStamped, "/obj_flag", self.cb_flag, qos)
        
        self.pub_path = self.create_publisher(Path, "/Path", 1)
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
        self.path_length = 0.0

        self.selected_path = None
        self.selected_path_frenet = None

        self.DEBUG_PATH_LOG = True
        self._JUMP_THRESH   = 2.0 

        self._center_enter = 0.35 
        self._center_exit  = 0.15  
        self._recover_cooldown_until = self.get_clock().now()

        self.create_timer(1.0 / plan_hz, self.planner)

    # ------------------- Helpers -------------------
    def _yaw_from_quat(self, q):
        x,y,z,w = q.x, q.y, q.z, q.w
        t3 = 2.0*(w*z + x*y); t4 = 1.0 - 2.0*(y*y + z*z)
        return math.atan2(t3, t4)

    def _nearest_idx_and_tangent(self, gp, px, py):
        xs = np.array([p[0] for p in gp], dtype=float)
        ys = np.array([p[1] for p in gp], dtype=float)
        i = int(np.argmin((xs-px)**2 + (ys-py)**2))
        i0 = max(0, i-1); i1 = min(len(gp)-1, i+1)
        vx = gp[i1][0]-gp[i0][0]; vy = gp[i1][1]-gp[i0][1]
        n = math.hypot(vx,vy) or 1.0
        return (vx/n, vy/n)

    def _heading_dir_sign(self):
        gp = getattr(self.converter, "global_path", None)
        if self.odom is None or not gp or len(gp) < 3:
            return +1.0
        yaw = self._yaw_from_quat(self.odom.pose.pose.orientation)
        ex, ey = math.cos(yaw), math.sin(yaw)
        px = self.odom.pose.pose.position.x
        py = self.odom.pose.pose.position.y
        tx, ty = self._nearest_idx_and_tangent(gp, px, py)
        return +1.0 if (ex*tx + ey*ty) >= 0.0 else -1.0
    


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

    # ------------------- 회피 조건 함수들 -------------------
    def should_avoid(self, obs_s, ego_s, ego_v, obs_v, flag):
        if not flag:
            return False
        relative_s = obs_s - ego_s
        if relative_s < 0:
            return False
        return False

    def is_obstacle_cleared(self):
        if self.ego_s is None:
            return False
        for obs in (self.static_xy, self.dynamic_xy):
            if obs is None:
                continue
            s_obs, _ = self.converter.global_to_frenet_point(*obs)
            if s_obs is None:
                continue
            if s_obs - self.ego_s >= -0.5:
                return False
        return True

    def is_path_safe(self, path, obstacles, min_clear=0.5):
        for (x, y) in path:
            for (ox, oy) in obstacles:
                if math.hypot(x - ox, y - oy) < (self.ROBOT_RADIUS + min_clear):
                    self.get_logger().warn("[SAFETY] Path too close to obstacle!")
                    return False
        return True

    # ------------------- 유틸(각도 정규화 & 코리도) -------------------
    def _wrap_angle(self, a: float) -> float:
        return (a + math.pi) % (2.0*math.pi) - math.pi

    def wall_safe(self, idx,
                       safety_margin: float = 0.3,
                       min_width: float = 0.40,
                       dir_sign: float = +1.0):

        r, l = self.converter.get_wall_distance(idx)  # (right, left)
        r, l = float(r), float(l)
        if dir_sign < 0: 
            r, l = l, r
        r = max(0.0, r); l = max(0.0, l)

        dmin = -l + safety_margin
        dmax =  r - safety_margin

        # 너무 좁으면 최소 폭 보장
        if dmax - dmin < min_width:
            mid = 0.5*(dmin + dmax)
            half = 0.5*min_width
            dmin, dmax = mid - half, mid + half

        return dmin, dmax, r, l

    def safe_path(self, path):
        frenet_path = self.converter.global_to_frenet(path)
        
        safe_frenet_path = []

        for point in frenet_path:
            s, d = point[0], point[1]
            x, y = self.converter.frenet_to_global_point(s, d)
            idx = self.converter._get_closest_index(x, y)
            
            dmin, dmax, _, _ = self.wall_safe(idx)
            safe_d = float(np.clip(d, dmin, dmax))

            safe_frenet_path.append([s, safe_d, point[2]])

        safe_path = self.converter.frenet_to_global(safe_frenet_path)

        return safe_path
    
    def get_closest_index(self, x, y, path):
        """가장 가까운 글로벌 경로 인덱스를 찾는 함수"""
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

    # ------------------- Reference 보정 spline -------------------
    def generate_ref_spline_path(self):
        # if self.ego_s is None or self.ego_d is None:
        #     return []

        # dir_sign = self._heading_dir_sign()
        # s0 = float(self.ego_s)
        # s1 = s0 + dir_sign * float(self.forward_length)
        # S_DIST = abs(s1 - s0)
        # if S_DIST < 1.0:
        #     self.get_logger().warn("[RefPath] Path generation distance too short.")
        #     return []

        # # 파라미터 (맵/트랙에 맞춰 조정 가능)
        # SAFETY_MARGIN = 0.30    # 0.30~0.35 권장
        # MIN_CORRIDOR  = 0.30
        # N_SAMPLES     = 50

        # # 시작점에서 현재 d를 안전 코리도로 1회 클립
        # dmin0, dmax0, _, _ = self._corridor_at_s(
        #     s0, prefer_d=self.ego_d,
        #     safety_margin=SAFETY_MARGIN, min_width=MIN_CORRIDOR, dir_sign=dir_sign
        # )
        # start_d = float(np.clip(self.ego_d, dmin0, dmax0))

        # target_d = 0.0

        # # Quintic (d' = d'' = 0)
        # qp = QuinticPolynomial(start_d, 0.0, 0.0, target_d, 0.0, 0.0, S_DIST)

        # # 샘플링
        # s_arr = np.linspace(s0, s1, N_SAMPLES, dtype=float)
        # s_rel = np.abs(s_arr - s0)
        # d_raw = np.array([qp.calc_point(s) for s in s_rel], dtype=float)

        # # 포인트별 코리도 클램프: 여기에서만 안전을 확보
        # d_vals = np.empty_like(d_raw)
        # for i, (s_i, d_i) in enumerate(zip(s_arr, d_raw)):
        #     dmin_i, dmax_i, _, _ = self._corridor_at_s(
        #         s_i, prefer_d=d_i,
        #         safety_margin=SAFETY_MARGIN, min_width=MIN_CORRIDOR, dir_sign=dir_sign
        #     )
        #     d_vals[i] = float(np.clip(d_i, dmin_i, dmax_i))

        # # Frenet & XY 생성
        # frenet_path = []
        # for s_i, d_i in zip(s_arr, d_vals):
        #     v_ref = self.converter.get_velocity_at_s(s_i)
        #     frenet_path.append([float(s_i), float(d_i), float(v_ref)])

        # if len(frenet_path) > 0:
        #     self.selected_path_frenet = np.asarray(frenet_path, dtype=float)
        #     xy_path = self.converter.frenet_to_global(frenet_path)   # [x, y, v]
        #     self.selected_path = np.asarray(xy_path, dtype=float)
        #     return xy_path

        if not self.global_path:
            self.get_logger().warn("[RefPath] Cannot generate: global_path is empty.")
            return []

        path_size = len(self.global_path)
        if path_size == 0:
            self.get_logger().warn("[RefPath] Path size is zero after guard.")
            return []

        idx = self.converter._get_closest_index(self.x, self.y)
        idx = int(idx % path_size)

        span = min(30, path_size)
        end_idx = (idx + span) % path_size
        self.get_logger().info(f"[RefPath] Generating frenet path from idx={idx} to end_idx={end_idx}")

        global_path = [
            self.global_path[(idx + offset) % path_size] for offset in range(span)
        ]

        if len(global_path) == 0:
            self.get_logger().warn("PATH EMPTY in safe_path!")

        safe_global_path = self.safe_path(global_path)
        safe_global_frenet_path = self.converter.global_to_frenet(safe_global_path)

        if len(safe_global_path) > 0:
            self.selected_path_frenet = np.asarray(safe_global_frenet_path, dtype=float)
            self.selected_path = np.asarray(safe_global_path, dtype=float)
            return safe_global_path

        self.get_logger().warn("[RefPath] frenet_path generation resulted in 0 points.")
        return []

    # ------------------- Avoidance functions -------------------
    def _apply_local_d_bump(self, base_d_points, obs_s,
                            bump, half_width_idx, side, smooth=True):

        d_points = np.copy(base_d_points)
        
        i_center = obs_s
        i0 = max(0, i_center - half_width_idx)
        i1 = min(d_points.size - 1, i_center + half_width_idx)

        self.get_logger().info(f"[BUMP] Applying bump at idx={i_center}, i0 = {i0}, i1 = {i1}")
        if i1 < i0:
            return d_points

        n = int(i1 - i0 + 1)
        if smooth and n > 1:
            # 중앙 1, 가장자리 0
            win = 0.5 * (1.0 + np.cos(np.linspace(-(math.pi), (math.pi), n)))
        else:
            win = np.ones(n)

        signed = float(side) * float(bump)
        d_points[i0 : i1 + 1] += signed * win
        return d_points

    def static_avoidance(self, selected_path, stat_s, stat_d, margin=0.3):
        if selected_path is None or len(selected_path) == 0:
            return None
        
        frenet_path = self.selected_path_frenet
        if frenet_path is None:
            self.get_logger().warn("[STATIC] Frenet snippet missing or length mismatch.")
            return None

        dir_sign = self._heading_dir_sign()

        s_obs, d_obs = float(stat_s), float(stat_d)
        x_obs, y_obs = self.converter.frenet_to_global_point(s_obs, d_obs)
        obs_idx = self.get_closest_index(x_obs, y_obs, selected_path)

        global_obs_idx = self.converter._get_closest_index(self.x_obs, self.y_obs)
        d_min, d_max, right_distance, left_distance = self.wall_safe(global_obs_idx)

        self.get_logger().info(
            f"(distance r={right_distance:.2f}, l={left_distance:.2f}, d_obs={d_obs:.2f})"
        )

        left_target  = d_obs - 0.4
        right_target = d_obs + 0.4

        can_go_right = right_target < d_max
        can_go_left  = left_target  > d_min

        if can_go_right and (right_distance > left_distance):
            side = -1.0
            d_target = right_target
        elif can_go_left:
            side = +1.0
            d_target = left_target
        else:
            self.get_logger().info("[STATIC] Can't move. Keep original path.")
            return None

        s_points = np.asarray(frenet_path[:, 0], dtype=float)
        base_d   = np.asarray(frenet_path[:, 1], dtype=float)

        bump_mag = float(d_target - d_obs)

        self.get_logger().info(f"[STATIC] Applying bump: side={side}, mag={bump_mag:.2f}")
        d_mod = self._apply_local_d_bump(
            base_d, obs_idx, bump=bump_mag,
            half_width_idx=5, side=side, smooth=True
        )

        static_path = []
        for s, d in zip(s_points, d_mod):
            x, y = self.converter.frenet_to_global_point(s, d)
            v_ref = self.converter.get_velocity_at_s(s)
            static_path.append([x, y, v_ref])

        return static_path


    # ------------------- Main Planner -------------------
    def planner(self):
        if self.odom is None or not self.converter.path_recived:
            return
        
        self.global_path = self.converter.get_global_path()
        self.global_frenet_path = self.converter.global_to_frenet(self.global_path)
        self.path_length = self.converter.get_path_length()

        lin = self.odom.twist.twist.linear
        ego_v = math.hypot(lin.x, lin.y)
        
        if self.flag_static == 1:
            stat_s, stat_d = self.static_s, self.static_d
        if self.dynamic_xy is not None:
            dyn_s, _ = self.converter.global_to_frenet_point(*self.dynamic_xy)

        default_path = self.generate_ref_spline_path()
        if not default_path:
            self.get_logger().warn("[Planner] default_path is empty; skip this cycle.")
            return
    
        static_cond = bool(self.flag_static)
        dynamic_cond = bool(self.flag_dynamic)

        static_p  = self.static_avoidance(self.selected_path, stat_s, stat_d) if static_cond else None

        if static_cond:
            if static_p:
                selected_path, path_type, self.mode = static_p, 1, 1
            else:
                self.get_logger().warn("[Planner] Static avoidance failed, falling back to global path.")
                selected_path, path_type, self.mode = default_path, 0, 0
        elif dynamic_cond and (self.priority == 1 or not static_cond):
            selected_path, path_type, self.mode = default_path, 0, 4
        else:
            selected_path, path_type, self.mode = default_path, 0, 0

        selected_xy = [(p[0], p[1]) for p in (selected_path or [])]
        self.publish_path_colored(selected_xy, (0.0, 1.0, 0.0, 0.5), "active_path")

        # Publish nav_msgs/Path
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

        if self.DEBUG_PATH_LOG:
            kind_map = {0: "reference", 1: "static", 2: "dynamic", 3: "recover"}
            kind_str = kind_map.get(path_type, str(path_type))
            extra = {
                "mode": self.mode,
                "ego_s": f"{(self.ego_s if self.ego_s is not None else float('nan')):.2f}",
                "ego_d": f"{(self.ego_d if self.ego_d is not None else float('nan')):.2f}",
                "ego_spd": f"{ego_v:.2f}",
                "flag_static": self.flag_static,
                "flag_dynamic": self.flag_dynamic,
                "priority": self.priority,
            }
            try:
                if self.static_xy is not None:
                    extra["stat_s"] = f"{float(self.converter.global_to_frenet_point(*self.static_xy)[0]):.2f}"
            except Exception:
                pass
            try:
                if self.dynamic_xy is not None:
                    extra["dyn_s"]  = f"{float(self.converter.global_to_frenet_point(*self.dynamic_xy)[0]):.2f}"
            except Exception:
                pass

    # ------------------- Visualization -------------------
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
