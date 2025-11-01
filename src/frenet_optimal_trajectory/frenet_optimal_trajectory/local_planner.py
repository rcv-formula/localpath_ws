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
        self.ROBOT_RADIUS = float(self.declare_parameter("ROBOT_RADIUS", 2.0).value)
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
        self.ego_s = self.ego_d = None
        self.static_xy = None
        self.dynamic_xy = None
        self.dynamic_v = None
        self.flag_dynamic = 0
        self.flag_static = 0
        self.priority = 0
        self.mode = 0  # 0=normal, 1=static, 2=dynamic, 3=return, 4=acc
        self.selected_path = None
        self.forward_length = 7.0

        self.selected_path = None

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
    
    def _sample_s(self, s_start: float, s_end: float, n: int):
        return np.array(np.linspace(s_start, s_end, n))

    def _path_stats(self, pts):
        n = len(pts)
        if n < 2:
            return {"n": n, "total_len": 0.0, "max_gap": 0.0, "max_gap_idx": -1}
        total = 0.0
        max_gap = 0.0
        max_idx = 0
        for i in range(n-1):
            dx = pts[i+1][0] - pts[i][0]
            dy = pts[i+1][1] - pts[i][1]
            d  = math.hypot(dx, dy)
            total += d
            if d > max_gap:
                max_gap = d
                max_idx = i
        return {"n": n, "total_len": total, "max_gap": max_gap, "max_gap_idx": max_idx}
    
    def _log_selected_path(self, kind_str, pts, extra=None):
        s = self._path_stats(pts)
        head = pts[0] if pts else (None, None)
        tail = pts[-1] if pts else (None, None)
        msg = (f"[PATH] kind={kind_str} n={s['n']} ")
        # self.get_logger().info(msg)

    # ------------------- Callbacks -------------------
    def cb_odom(self, msg):
        self.odom = msg
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.ego_s, d = self.converter.global_to_frenet_point(x, y)
        self.cur_idx = self.converter._get_closest_index(x, y)
        self.ego_d = d

    def cb_static(self, msg):
        self.static_xy = np.array([msg.point.x, msg.point.y], dtype=float)
        self.cur_obs_idx = self.converter._get_closest_index(msg.point.x, msg.point.y)

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

    def _corridor_at_s(self, s: float, prefer_d: float,
                       safety_margin: float = 0.40,
                       min_width: float = 0.30,
                       dir_sign: float = +1.0):

        idx = self.converter._get_closest_index(float(s), float(prefer_d))
        r, l = self.converter.get_wall_distance(idx)  # (right, left)
        r, l = float(r), float(l)
        if dir_sign < 0:  # 역방향이면 오른/왼 뒤집기
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

    # ------------------- Reference 보정 spline -------------------
    def generate_ref_spline_path(self):
        if self.ego_s is None or self.ego_d is None:
            return []

        dir_sign = self._heading_dir_sign()
        s0 = float(self.ego_s)
        s1 = s0 + dir_sign * float(self.forward_length)
        S_DIST = abs(s1 - s0)

        if S_DIST < 1.0:
            self.get_logger().warn("[RefPath] Path generation distance too short.")
            return []

        # 파라미터
        SAFETY_MARGIN = 0.40
        MIN_CORRIDOR  = 0.30
        N_SAMPLES     = 25
        FINAL_CHECK_MARGIN = SAFETY_MARGIN

        dmin0, dmax0, _, _ = self._corridor_at_s(s0, prefer_d=self.ego_d,
                                                 safety_margin=SAFETY_MARGIN,
                                                 min_width=MIN_CORRIDOR,
                                                 dir_sign=dir_sign)
        start_d  = float(np.clip(self.ego_d, dmin0, dmax0))

        dmin1, dmax1, _, _ = self._corridor_at_s(s1, prefer_d=0.0,
                                                 safety_margin=SAFETY_MARGIN,
                                                 min_width=MIN_CORRIDOR,
                                                 dir_sign=dir_sign)
        target_d = float(np.clip(0.0, dmin1, dmax1))

        # Quintic: 헤딩 보정 없음 (d' = d'' = 0)
        qp = QuinticPolynomial(start_d, 0.0, 0.0, target_d, 0.0, 0.0, S_DIST)

        # 샘플링
        s_points = self._sample_s(s0, s1, n=N_SAMPLES)
        if not np.asarray(s_points).any():
            return []
            
        s_arr = np.asarray(s_points, dtype=float)
        s_rel = np.abs(s_arr - s0)
        d_raw = np.asarray([qp.calc_point(s) for s in s_rel], dtype=float)

        d_vals = np.empty_like(d_raw)
        for i, (s_i, d_i) in enumerate(zip(s_arr, d_raw)):
            dmin_i, dmax_i, _, _ = self._corridor_at_s(s_i, prefer_d=d_i,
                                                       safety_margin=SAFETY_MARGIN,
                                                       min_width=MIN_CORRIDOR,
                                                       dir_sign=dir_sign)
            d_vals[i] = float(np.clip(d_i, dmin_i, dmax_i))

        # 모든 점 충돌 검사
        frenet_path = []
        path_is_safe = True
        for s_i, d_i in zip(s_arr, d_vals):
            idx = self.converter._get_closest_index(s_i, d_i)
            right_d, left_d = self.converter.get_wall_distance(idx)
            if dir_sign == -1.0:
                right_d, left_d = left_d, right_d
            right_d = float(right_d); left_d = float(left_d)

            if (d_i > right_d - FINAL_CHECK_MARGIN) or (d_i < -left_d + FINAL_CHECK_MARGIN):
                path_is_safe = False
                self.get_logger().warn(
                    f"[RefPath] Collision at s={s_i:.2f}, d={d_i:.2f} (r={right_d:.2f}, l={left_d:.2f})"
                )
                break

            v_ref = self.converter.get_velocity_at_s(s_i)
            frenet_path.append([float(s_i), float(d_i), float(v_ref)])

        if len(frenet_path) > 0:
            self.selected_path = np.asarray(frenet_path, dtype=float)
        else:
           self.get_logger().warn("[RefPath] frenet_path generation resulted in 0 points.")

        # 결과
        if path_is_safe and len(frenet_path) >= 2:
            return self.converter.frenet_to_global(frenet_path)

        # 폴백: 글로벌 스니펫
        self.get_logger().warn("[RefPath] Path unsafe, returning original global path snippet.")
        global_path = self.converter.get_global_path()
        path_len = len(global_path)
        if path_len == 0:
            return []
        end_idx = (self.cur_idx + 40)
        if end_idx >= path_len:
            path = global_path[self.cur_idx:] + global_path[:(end_idx % path_len)]
        else:
            path = global_path[self.cur_idx : end_idx]
        return path

    # ------------------- Avoidance functions -------------------
    def _apply_local_d_bump(self, s_points, base_d_points, s_center,
                            bump, half_width_idx=2, side=+1, smooth=True):

        d_points = np.copy(base_d_points)
        
        i_center = int(np.argmin(np.abs(np.array(s_points) - float(s_center))))
        i0 = max(0, i_center - half_width_idx)
        i1 = min(d_points.size - 1, i_center + half_width_idx)
        if i1 < i0:
            return d_points

        n = int(i1 - i0 + 1)
        if smooth and n > 1:
            # 중앙 1, 가장자리 0
            win = 0.5 * (1.0 + np.cos(np.linspace(-(math.pi), math.pi, n)))
        else:
            win = np.ones(n)

        signed = float(side) * float(bump)
        d_points[i0 : i1 + 1] += signed * win
        return d_points

    def static_avoidance(self, selected_path, margin=0.4):
        if selected_path is None or len(selected_path) == 0:
            return None

        s_points = np.asarray(selected_path[:,0])
        base_d   = np.asarray(selected_path[:,1])
        # v_ref   = np.asarray(selected_path[:,2])  # 미사용

        dir_sign = self._heading_dir_sign()

        if self.static_xy is None or self.ego_s is None or self.ego_d is None:
            return None

        s_obs, d_obs = self.converter.global_to_frenet_point(*self.static_xy)
        if s_obs is None:
            return None
        
        s_obs_raw = float(s_obs)
        self.get_logger().info(f"[STATIC] Obstacle at s={s_obs_raw:.2f}, d={d_obs:.2f}")

        obs_idx = self.converter._get_closest_index(s_obs_raw, d_obs)
        right_void, left_void = self.converter.get_wall_distance(obs_idx)
        if dir_sign == -1.0:
            right_void, left_void = left_void, right_void  # ← 올바른 스왑

        right_clearance = float(right_void - d_obs)
        left_clearance  = float(left_void  + d_obs)
        
        self.get_logger().info(f"[STATIC] Using safety margin: {margin:.2f}m")

        can_go_right = right_clearance > 0.6
        can_go_left  = left_clearance  > 0.6

        i_center_s_idx = int(np.argmin(np.abs(s_points - float(s_obs_raw))))
        d_base_at_obs = base_d[i_center_s_idx]

        side = 0.0
        bump_mag = 0.0

        if can_go_right and (right_clearance > left_clearance):
            side = +1.0
            d_target = d_obs + margin
            bump_mag = max(0.0, d_target - d_base_at_obs) 
        elif can_go_left:
            side = -1.0
            d_target = d_obs - margin
            bump_mag = max(0.0, d_base_at_obs - d_target)

        self.get_logger().info(f"[STATIC] Applying bump: side={side}, mag={bump_mag:.2f}")
        d_mod = self._apply_local_d_bump(
            s_points, base_d, s_center=s_obs_raw, bump=bump_mag,
            half_width_idx=3, side=side, smooth=True
        )
    
        path_xyv = []
        for s, d in zip(s_points, d_mod):
            x, y = self.converter.frenet_to_global_point(s, d)
            if x is None or y is None:
                continue
            v_ref = self.converter.get_velocity_at_s(s)
            path_xyv.append([x, y, v_ref])

        return path_xyv

    # ------------------- Main Planner -------------------
    def planner(self):
        if self.odom is None or not self.converter.path_recived:
            return

        lin = self.odom.twist.twist.linear
        ego_v = math.hypot(lin.x, lin.y)

        stat_s = dyn_s = None
        if self.static_xy is not None:
            stat_s, _ = self.converter.global_to_frenet_point(*self.static_xy)
        if self.dynamic_xy is not None:
            dyn_s, _ = self.converter.global_to_frenet_point(*self.dynamic_xy)

        default_path = self.generate_ref_spline_path()
        
        static_cond = bool(self.flag_static)
        dynamic_cond = bool(self.flag_dynamic)
        
        static_p  = self.static_avoidance(self.selected_path) if static_cond else None

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
            self._log_selected_path(kind_str, selected_path, extra=extra)

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
