#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nav_msgs.msg import Path
import csv
import numpy as np
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from typing import List, Tuple, Optional


# =========================
# CSV 기반 트랙 경계 핸들러
# =========================
class TrackBoundary:
    """
    CSV로 제공되는 트랙 중심선 (x,y)과 좌/우 폭(l,r)을 읽어들인다.
    CSV 형식(열 위치 고정, 1행 헤더 스킵):
      0: x, 1: y, 2: right(우), 3: left(좌)  [단위 m]

    내부 표현:
      - center: (cx, cy)
      - widths: wl(left), wr(right)

    제공 메서드:
      - widths_at(x,y): 중심선 '가장 가까운 세그먼트'에 대한 선형보간 폭 (left,right)
      - widths_and_offset(x,y): 위 폭 + 중심선 좌측을 +로 하는 서명 횡오프셋 d_signed
    """

    def __init__(self, csv_path: str, node: Node):
        self.node = node
        self.csv_path = csv_path

        self.cx: np.ndarray = np.empty((0,), dtype=float)  # centerline x
        self.cy: np.ndarray = np.empty((0,), dtype=float)  # centerline y
        self.wl: np.ndarray = np.empty((0,), dtype=float)  # left width (m)
        self.wr: np.ndarray = np.empty((0,), dtype=float)  # right width (m)

        self._load_csv_fixed_columns()

    def _load_csv_fixed_columns(self):
        """열 위치 고정 + 1행 헤더 스킵: 0:x, 1:y, 2:right(우), 3:left(좌)"""
        rows = []
        with open(self.csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            _header = next(reader, None)  # 무조건 스킵
            for row in reader:
                row = [c.strip() for c in row]
                if not row or len(row) < 4:
                    continue
                if row[0].startswith("#"):
                    continue
                try:
                    x = float(row[0])
                    y = float(row[1])
                    r = float(row[2])  # 우
                    l = float(row[3])  # 좌
                    rows.append((x, y, l, r))  # 내부: (x, y, left, right)
                except Exception:
                    continue

        if not rows:
            raise ValueError("CSV에서 유효한 데이터 행을 읽지 못했습니다. 형식: [x, y, 우, 좌] (단위 m)")

        arr = np.array(rows, dtype=float)
        self.cx = arr[:, 0]
        self.cy = arr[:, 1]
        self.wl = arr[:, 2]  # left
        self.wr = arr[:, 3]  # right

        self.node.get_logger().info(
            f"[TrackBoundary] Loaded {len(self.cx)} center points from {self.csv_path} (skip header row)"
        )

    @staticmethod
    def _safe_unit(vx: float, vy: float) -> Tuple[float, float]:
        n = math.hypot(vx, vy)
        if n < 1e-12:
            return 1.0, 0.0
        return vx / n, vy / n

    @staticmethod
    def _left_normal(tx: float, ty: float) -> Tuple[float, float]:
        # (tx,ty)를 좌측으로 90도 회전
        return -ty, tx

    def _project_to_segment(self, ax, ay, bx, by, px, py):
        """
        세그먼트 A(ax,ay)->B(bx,by) 에 점 P(px,py)를 투영.
        반환: (t, qx, qy, dist)
          - t: [0,1] 범위의 세그먼트 보간 계수
          - qx,qy: 투영점
          - dist: |P-Q|
        """
        ABx, ABy = (bx - ax), (by - ay)
        APx, APy = (px - ax), (py - ay)
        denom = (ABx * ABx + ABy * ABy)
        if denom < 1e-12:
            return 0.0, ax, ay, math.hypot(px - ax, py - ay)
        t = (APx * ABx + APy * ABy) / denom
        t_clamped = max(0.0, min(1.0, t))
        qx = ax + t_clamped * ABx
        qy = ay + t_clamped * ABy
        return t_clamped, qx, qy, math.hypot(px - qx, py - qy)

    # ---- 기존 유지: 폭만 필요할 때 ----
    def widths_at(self, x: float, y: float) -> Tuple[float, float]:
        """
        임의의 (x,y)에서 가장 가까운 중심선 '세그먼트'를 찾아
        그 세그먼트의 양 끝점 좌/우 폭을 t로 선형보간해 (left, right) 반환.
        """
        n = len(self.cx)
        if n == 0:
            return 0.0, 0.0
        if n == 1:
            return float(self.wl[0]), float(self.wr[0])

        d2 = (self.cx - x) ** 2 + (self.cy - y) ** 2
        j = int(np.argmin(d2))
        cand_segments = []
        if j > 0:
            cand_segments.append((j - 1, j))
        if j < n - 1:
            cand_segments.append((j, j + 1))
        if not cand_segments:
            cand_segments = [(max(0, j - 1), j)]

        best = (1e18, 0.0, 0)  # (dist, t, i0)
        for i0, i1 in cand_segments:
            t, qx, qy, dist = self._project_to_segment(self.cx[i0], self.cy[i0], self.cx[i1], self.cy[i1], x, y)
            if dist < best[0]:
                best = (dist, t, i0)

        _, t, i0 = best
        i1 = min(i0 + 1, n - 1)
        left  = (1.0 - t) * self.wl[i0] + t * self.wl[i1]
        right = (1.0 - t) * self.wr[i0] + t * self.wr[i1]
        return float(left), float(right)

    # ---- 추가: 폭 + 서명된 횡오프셋을 함께 반환 ----
    def widths_and_offset(self, x: float, y: float) -> Tuple[float, float, float]:
        """
        (x,y)에 대해:
          - left, right: 해당 위치에서의 트랙 좌/우 폭(선형보간)
          - d_signed: '중심선 좌측을 +, 우측을 −' 로 하는 서명된 횡오프셋 (단위: m)
        """
        n = len(self.cx)
        if n == 0:
            return 0.0, 0.0, 0.0
        if n == 1:
            # 탄젠트가 없으니 임시로 (1,0) 기준 좌측 법선 적용
            nx, ny = 0.0, 1.0
            d_signed = (x - self.cx[0]) * nx + (y - self.cy[0]) * ny
            return float(self.wl[0]), float(self.wr[0]), float(d_signed)

        d2 = (self.cx - x) ** 2 + (self.cy - y) ** 2
        j = int(np.argmin(d2))
        cand_segments = []
        if j > 0:
            cand_segments.append((j - 1, j))
        if j < n - 1:
            cand_segments.append((j, j + 1))
        if not cand_segments:
            cand_segments = [(max(0, j - 1), j)]

        best = (1e18, 0.0, 0, 0.0, 0.0)  # (dist, t, i0, qx, qy)
        for i0, i1 in cand_segments:
            t, qx, qy, dist = self._project_to_segment(self.cx[i0], self.cy[i0], self.cx[i1], self.cy[i1], x, y)
            if dist < best[0]:
                best = (dist, t, i0, qx, qy)

        _, t, i0, qx, qy = best
        i1 = min(i0 + 1, n - 1)

        # 세그먼트 탄젠트 -> 좌측 법선
        tx = self.cx[i1] - self.cx[i0]
        ty = self.cy[i1] - self.cy[i0]
        tx, ty = self._safe_unit(tx, ty)
        nx, ny = self._left_normal(tx, ty)

        # (x,y) 의 좌측(+)/우측(−) 서명 오프셋 (중심선 좌측이 +)
        d_signed = (x - qx) * nx + (y - qy) * ny

        # 폭 선형보간
        left  = (1.0 - t) * self.wl[i0] + t * self.wl[i1]
        right = (1.0 - t) * self.wr[i0] + t * self.wr[i1]
        return float(left), float(right), float(d_signed)


# =================================
# 좌표 변환 + CSV 기반 벽거리 계산기
# =================================
class coordinate_converter:
    """
    글로벌 (x,y) <-> 프레네 변환 + CSV 기반 트랙 폭으로 wall distance 계산.
    핵심: wall distance는 **global path의 국소 법선** 기준이 아니라,
         '트랙 중심선'에서의 서명된 횡오프셋 d_signed를 이용해
         각 측 폭에서 그만큼을 깎아 계산한다.
    """

    def __init__(self, node: Node):
        self.node: Node = node
        self.path_recived: bool = False

        # 파라미터
        self.track_csv_path: str = str(self.node.declare_parameter("track_csv_path", "/home/rcv/Desktop/localpath_ws/src/frenet_optimal_trajectory/Track/1103_track.csv").value)
        self.max_clearance: float = float(self.node.declare_parameter("max_clearance", 0.0).value)  # 0=무제한

        # 글로벌 경로 [x, y, v]
        self.global_path: list[list[float]] = []
        self.right_void_list: list[float] = []
        self.left_void_list: list[float] = []

        # 트랙 경계
        self.track: Optional[TrackBoundary] = None
        if self.track_csv_path:
            try:
                self.track = TrackBoundary(self.track_csv_path, self.node)
            except Exception as e:
                self.node.get_logger().error(f"트랙 CSV 로드 실패: {e}")

        # global_path 구독
        path_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.path_subscription = self.node.create_subscription(
            Path, "global_path", self.path_callback, path_qos
        )
        self.path_wait_timer = self.node.create_timer(0.2, self.check_path_received)

    def check_path_received(self):
        if self.path_recived:
            self.node.get_logger().info(f"Path received: {len(self.global_path)} points")
            self.path_wait_timer.cancel()

    # ----------------------------
    # 길이/거리 유틸
    # ----------------------------
    def _calc_distance(self, A, B):
        """두 점 A=[x,y], B=[x,y] 사이 거리"""
        return math.hypot(A[0] - B[0], A[1] - B[1])
    
    def get_global_path(self) -> List[List[float]]:
        return self.global_path

    def get_path_length(self, start_idx=None):
        if len(self.global_path) == 0:
            self.node.get_logger().error("Cannot calculate path length: global_path is empty!")
            return 0
        if start_idx is not None:
            return self._calc_path_distance(start_idx, len(self.global_path) - 1)
        return self._calc_path_distance(0, len(self.global_path) - 1)

    def _calc_path_distance(self, start_idx, end_idx):
        distance_counter = 0.0
        if end_idx < start_idx:
            end_idx += len(self.global_path)
        for i in range(start_idx, end_idx):
            distance_counter += self._calc_path_to_path_distance(i)
        return distance_counter

    def _calc_path_to_path_distance(self, idx):
        if idx < 0:
            idx += len(self.global_path)
        idx = idx % len(self.global_path)
        next_idx = (idx + 1) % len(self.global_path)
        cur = self.global_path[idx][:2]
        nxt = self.global_path[next_idx][:2]
        return math.hypot(cur[0] - nxt[0], cur[1] - nxt[1])

    # ----------------------------
    # CSV 기반 wall distance (global path 기준)
    # ----------------------------
    def calc_all_wall_distances(self) -> None:
        """
        global_path 각 점 (x,y)에 대해:
          1) track 중심선의 가장 가까운 세그먼트로 투영 → (left, right, d_signed) 획득
          2) global path 기준 좌/우 여유폭으로 변환:
             left_clear  = max(0, left  - max(0,  d_signed))
             right_clear = max(0, right - max(0, -d_signed))
          3) max_clearance > 0 이면 상한 적용
        """
        if not self.global_path:
            self.node.get_logger().warn("calc_all_wall_distances: global_path is empty; skip.")
            return
        if self.track is None:
            self.node.get_logger().warn("calc_all_wall_distances: track CSV not loaded; skip.")
            return

        self.right_void_list = [0.0] * len(self.global_path)
        self.left_void_list  = [0.0] * len(self.global_path)

        for i, (x, y, _) in enumerate(self.global_path):
            l, r, d = self.track.widths_and_offset(x, y)  # l,r (m), d_signed (m; left +)
            left_clear  = max(0.0, l - max(0.0,  d))   # 왼쪽으로 치우치면 왼쪽 여유폭 감소
            right_clear = max(0.0, r - max(0.0, -d))   # 오른쪽으로 치우치면 오른쪽 여유폭 감소

            if self.max_clearance > 0:
                left_clear  = min(left_clear,  self.max_clearance)
                right_clear = min(right_clear, self.max_clearance)

            self.left_void_list[i]  = left_clear
            self.right_void_list[i] = right_clear

        self.node.get_logger().info("Computed CSV-based wall distances (relative to global path).")

    def get_wall_distance(self, idx: int) -> tuple[float, float]:
        if 0 <= idx < len(self.global_path) and self.right_void_list and self.left_void_list:
            return self.right_void_list[idx], self.left_void_list[idx]
        return (0.0, 0.0)

    # ----------------------------
    # 프레네 변환 (기존 로직 유지)
    # ----------------------------
    def global_to_frenet_point(self, x, y):
        if len(self.global_path) == 0:
            self.node.get_logger().warn("global_to_frenet_point: No global path available.")
            return [0, 0]

        closest_idx = self._get_closest_index(x, y)

        if closest_idx >= len(self.global_path) - 1:
            self.node.get_logger().warn("Closest index is out of bounds!")
            return [0, 0]

        out1 = self._calc_proj(closest_idx, (closest_idx + 1), x, y)
        out2 = self._calc_proj(closest_idx - 1, closest_idx, x, y)

        if abs(out1[1]) > abs(out2[1]):
            s = out2[0] + self._calc_path_distance(0, closest_idx - 1)
            d = out2[1]
        else:
            s = out1[0] + self._calc_path_distance(0, closest_idx)
            d = out1[1]
        return [s, d]

    def global_to_frenet(self, path_list):
        output = []
        for p in path_list:
            s_d = self.global_to_frenet_point(p[0], p[1])
            v = p[2]
            output.append([s_d[0], s_d[1], v])
        return output

    def frenet_to_global_point(self, s, d):
        path_len = self.get_path_length()
        if path_len == 0 or len(self.global_path) < 2:
            self.node.get_logger().warn("Frenet_to_global: Path is too short.")
            return [0, 0]

        if s < 0.0:
            s += path_len
        s = s % path_len

        start_idx = self._get_start_path_from_frenet(s)
        next_idx = (start_idx + 1) % len(self.global_path)

        s_residual = s - self._calc_path_distance(0, start_idx)

        start_point = np.array(self.global_path[start_idx][:2])
        next_point = np.array(self.global_path[next_idx][:2])

        segment_vector = next_point - start_point
        segment_length = np.linalg.norm(segment_vector)

        unit_vector = np.array([0.0, 0.0])
        if segment_length > 1e-6:
            unit_vector = segment_vector / segment_length

        proj_point = start_point + (unit_vector * s_residual)
        normal_vector = self.rotate_right_90(unit_vector)
        global_point = proj_point + (normal_vector * d)
        return list(global_point)

    def frenet_to_global(self, path_list: list):
        output = []
        for p in path_list:
            global_point = self.frenet_to_global_point(p[0], p[1])
            output.append(global_point + [p[2]])
        return output

    @staticmethod
    def rotate_right_90(v):
        return np.array([v[1], -v[0]])

    def _get_start_path_from_frenet(self, s):
        idx = 0
        while self._calc_path_distance(0, idx + 1) <= s and (idx + 1) < len(self.global_path):
            idx += 1
        return idx

    def _get_closest_index(self, x, y):
        if len(self.global_path) == 0:
            self.node.get_logger().warn("Cannot find closest index: Global path is empty!")
            return 0
        pts = np.asarray(self.global_path, dtype=float)[:, :2]
        d2 = np.sum((pts - np.array([x, y])) ** 2, axis=1)
        return int(np.argmin(d2))

    def _calc_proj(self, idx, next_idx, x, y):
        path_size = len(self.global_path)
        idx = (idx % path_size + path_size) % path_size
        next_idx = (next_idx % path_size + path_size) % path_size

        pointA = np.array(self.global_path[idx][:2])
        pointB = np.array(self.global_path[next_idx][:2])
        pointC = np.array([x, y])

        vectorA = pointB - pointA
        vectorB = pointC - pointA

        denom = np.dot(vectorA, vectorA)
        if denom < 1e-12:
            proj_t = 0.0
            proj_point = pointA
        else:
            proj_t = np.dot(vectorB, vectorA) / denom
            proj_point = pointA + (proj_t * vectorA)

        d = np.linalg.norm(proj_point - pointC)
        if np.cross(vectorA, vectorB) > 0:
            d = -d
        s = proj_t * np.linalg.norm(vectorA)  # 길이 스케일로 s를 정의
        return [s, d]

    # ----------------------------
    # 속도 헬퍼 (하위 호환 포함)
    # ----------------------------
    def get_velocity_at_s(self, s: float) -> float:
        """전역 경로의 호 길이 s 위치에서 참조 속도를 반환."""
        if not self.global_path:
            return 0.0
        path_len = self.get_path_length()
        if path_len == 0:
            return float(self.global_path[0][2])
        s = s % path_len
        idx = self._get_start_path_from_frenet(s)
        return float(self.global_path[idx][2])

    # 과거 코드 호환용
    def _get_velocity_at_s(self, s: float) -> float:
        return self.get_velocity_at_s(s)

    # ----------------------------
    # 콜백
    # ----------------------------
    def path_callback(self, msg: Path):
        if not msg.poses:
            self.node.get_logger().warn("Received empty global path!")
            return
        self.global_path = [
            [p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in msg.poses
        ]
        self.path_recived = True
        self.node.get_logger().info(f"Received global path with {len(self.global_path)} points")

        # 트랙 CSV가 준비되어 있으면 즉시 계산
        if self.track is not None:
            self.calc_all_wall_distances()


def main():
    rclpy.init()
    node = Node("coordinate_converter_csv_node")
    converter = coordinate_converter(node)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
