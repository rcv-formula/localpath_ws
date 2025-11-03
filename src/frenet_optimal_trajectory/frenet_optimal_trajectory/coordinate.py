from nav_msgs.msg import Path
import numpy as np
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from typing import List, Tuple, Optional

try:
    # nav_msgs.msg may not be available in non‑ROS environments; guard import
    from nav_msgs.msg import OccupancyGrid
except Exception:
    OccupancyGrid = None  # type: ignore


class MapHandler:
    """
    A helper class that wraps a nav_msgs/msg/OccupancyGrid and exposes
    methods to convert between world coordinates and map indices as well
    as query individual cell occupancy.

    This class makes minimal assumptions about the coordinate frame of the
    occupancy grid: we honour the `origin` pose and `resolution` fields.
    Orientation is assumed to be around the Z axis only (2D rotation),
    consistent with the ROS map conventions.  Unknown cells (with value
    -1) are treated as occupied for the purposes of wall detection.
    """
    def __init__(self, msg: OccupancyGrid):
        self.resolution: float = msg.info.resolution
        self.width: int = msg.info.width
        self.height: int = msg.info.height
        # Extract the origin translation and yaw.  The origin describes the
        # world coordinate of the map cell (0,0), which is the lower left
        # corner of the occupancy grid if the yaw is zero.
        self.origin_x: float = msg.info.origin.position.x
        self.origin_y: float = msg.info.origin.position.y
        # orientation quaternion to yaw (2D rotation about z)
        q = msg.info.origin.orientation
        # Convert quaternion to yaw.  Fallback gracefully if `math` is not
        # available (e.g., if imported outside of ROS environment).
        try:
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.yaw = math.atan2(siny_cosp, cosy_cosp)
        except Exception:
            self.yaw = 0.0
        # Precompute sine and cosine of yaw for coordinate transforms
        self.cos_yaw = math.cos(self.yaw)
        self.sin_yaw = math.sin(self.yaw)
        # Flatten the occupancy data into a NumPy array for efficient indexing
        # The data is provided as a flat list in row-major order, starting
        # from the cell at (0,0) in the map coordinate frame.
        self.data = np.array(msg.data, dtype=np.int8).reshape((self.height, self.width))

    def world_to_map(self, x: float, y: float) -> tuple[int, int] | None:
        """
        Convert a world coordinate (x, y) into integer map indices (mx, my).
        If the coordinate falls outside the map bounds, `None` is returned.
        """
        # Translate into map frame
        dx = x - self.origin_x
        dy = y - self.origin_y
        # Rotate by negative yaw to align with map axes
        mx_f = (dx * self.cos_yaw + dy * self.sin_yaw) / self.resolution
        my_f = (-dx * self.sin_yaw + dy * self.cos_yaw) / self.resolution
        mx = int(math.floor(mx_f))
        my = int(math.floor(my_f))
        if mx < 0 or my < 0 or mx >= self.width or my >= self.height:
            return None
        # In ROS occupancy grids the row index (y) increases from bottom to top.
        # However, the data array is stored starting from the bottom-left
        # corner when yaw is zero.  Because we directly compute `my` from the
        # origin we can use it as an index into `self.data` without further
        # transformation.
        return mx, my

    def get_pixel(self, x: float, y: float) -> int | None:
        """
        Return the occupancy value of the cell containing the world
        coordinate (x, y).  If the coordinate falls outside the map, or if
        map data is unavailable, `None` is returned.  The return value is
        between 0 (free) and 100 (occupied), or -1 (unknown).
        """
        indices = self.world_to_map(x, y)
        if indices is None:
            return None
        mx, my = indices
        # Note: in numpy indexing `self.data[row, col]` corresponds to
        # (y, x).  `my` is the row index and `mx` is the column index.
        return int(self.data[my, mx])

    def is_occupied(self, x: float, y: float) -> bool:
        """
        Determine whether the cell containing (x, y) is occupied.  Unknown
        cells are considered occupied to err on the side of caution.
        """
        val = self.get_pixel(x, y)
        if val is None:
            return True  # outside the map bounds is treated as occupied
        # In ROS occupancy grids values:
        #   0    -> free
        #   100  -> completely occupied
        #   -1   -> unknown
        return val != 0

class coordinate_converter:
    """
    A conversion utility that transforms between a global (x, y) frame and the
    Frenet frame defined along a path.  In addition to the Frenet conversion
    routines, this class can optionally subscribe to a `nav_msgs/msg/OccupancyGrid`
    to estimate the available clearance to the left and right of each path
    waypoint.  Distances are measured in metres and are computed by stepping
    perpendicular to the local path tangent until either an occupied cell or
    a maximum search distance is encountered.

    The original C++ implementation provided functions to compute these
    distances using a pre-loaded occupancy grid map.  The Python port below
    attempts to mirror that behaviour closely while adhering to Pythonic
    idioms.  Mapping support is optional; if no occupancy grid message is
    supplied the wall distance computations are skipped.
    """

    def __init__(self, node: Node):
        self.node: Node = node
        # Flags to track whether path and map information have been received
        self.path_recived: bool = False
        self.map_received: bool = False
        # Container for the global path as a list of [x, y, velocity]
        self.global_path: list[list[float]] = []
        # Containers for wall distances corresponding to each path index
        self.right_void_list: list[float] = []
        self.left_void_list: list[float] = []

        # Subscribe to the global path.  The QoS settings mirror those used
        # in the C++ implementation: reliable delivery with transient local
        # durability ensures late-joining subscribers will receive the last
        # published path.
        path_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.path_subscription = self.node.create_subscription(
            Path,
            'global_path',
            self.path_callback,
            path_qos
        )

        # Subscribe to a map topic if available.  We default to 'map' which
        # typically publishes a nav_msgs/msg/OccupancyGrid.  The QoS is set
        # similarly to the path to guarantee reception of the most recent
        # occupancy grid.
        try:
            from nav_msgs.msg import OccupancyGrid
            map_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
            self.map_subscription = self.node.create_subscription(
                OccupancyGrid,
                'map',
                self.map_callback,
                map_qos
            )
        except Exception:
            # If nav_msgs cannot be imported we simply skip map support
            self.map_subscription = None

        # Storage for the most recent map.  When a map is received the
        # `MapHandler` instance will be replaced.  Until then it remains
        # `None`.
        # A helper object for map access.  It is set once a map message is
        # received.  When no map is available this remains None.
        self.map_handler: Optional[MapHandler] = None

        # A timer to periodically check whether a path has been received.  Once
        # the path is available we cancel this timer.  This mirrors the
        # `calcAllWallDist()` call in the C++ constructor which runs after
        # the path has been loaded.
        self.path_wait_timer = self.node.create_timer(0.1, self.check_path_received)

    def check_path_received(self):
        """Called periodically until the global path has been loaded."""
        if self.path_recived:
            self.node.get_logger().info(
                f"Path received: {len(self.global_path)} points"
            )
            self.path_wait_timer.cancel()  # Stop the timer

    def get_path_length(self, start_idx=None):
        if len(self.global_path) == 0:
            self.node.get_logger().error("Cannot calculate path length: global_path is empty!")
            return 0  # Instead of returning None, return 0

        if start_idx is not None:
            return self._calc_path_distance(start_idx, len(self.global_path) - 1)
        
        return self._calc_path_distance(0, len(self.global_path) - 1)

    # ------------------------------------------------------------------
    # Map handling
    # ------------------------------------------------------------------
    def map_callback(self, msg):
        """
        Callback for receiving a nav_msgs/msg/OccupancyGrid.  Upon receipt
        the grid data and metadata are parsed and stored in a helper class
        (`MapHandler`) which provides convenience functions for converting
        between world coordinates and map indices and for querying cell
        occupancy.

        Once both a path and a map have been received we compute the wall
        distances for each path waypoint.
        """
        try:
            self.map_handler = MapHandler(msg)
            self.map_received = True
            self.node.get_logger().info(
                f"Received occupancy grid: {msg.info.width}x{msg.info.height} at {msg.info.resolution} m/px"
            )
            # If the path is already available then compute wall distances
            if self.path_recived:
                self.calc_all_wall_distances()
        except Exception as e:
            self.node.get_logger().error(
                f"Failed to process occupancy grid: {e}"
            )

    def get_global_path(self):
        return self.global_path

    # ------------------------------------------------------------------
    # Wall distance computation
    # ------------------------------------------------------------------
    def calc_all_wall_distances(self, max_search_distance: float = 2.0) -> None:
        """
        Compute the available clearance to the left and right of every
        waypoint along the global path.  Distances are measured from the
        waypoint outwards along a direction perpendicular to the local path
        tangent.  Searching stops once an occupied cell is encountered or
        when `max_search_distance` (in metres) is reached.  Results are
        stored in `self.right_void_list` and `self.left_void_list`.

        This method requires both a non‑empty global path and an available
        occupancy grid.  If either is missing the function logs a warning
        and returns without computing distances.

        Parameters
        ----------
        max_search_distance : float
            The maximum distance (in metres) to search for a wall.
        """
        if not self.global_path:
            self.node.get_logger().warn(
                "calc_all_wall_distances: global_path is empty; skipping wall distance calculation."
            )
            return
        if not self.map_handler:
            self.node.get_logger().warn(
                "calc_all_wall_distances: no map available; skipping wall distance calculation."
            )
            return

        # Clear any previous results
        self.right_void_list = [0.0] * len(self.global_path)
        self.left_void_list = [0.0] * len(self.global_path)

        for idx in range(len(self.global_path)):
            right_dist, left_dist = self._calc_wall_distance(idx, max_search_distance)
            self.right_void_list[idx] = right_dist
            self.left_void_list[idx] = left_dist
        self.node.get_logger().info("Computed wall distances for all path waypoints")

    def get_wall_distance(self, idx: int) -> tuple[float, float]:
        """
        Return the right and left clearance at the specified path index.
        If distances have not yet been computed or the index is out of
        range, zero distances are returned.  The returned tuple is
        (right_distance, left_distance).
        """
        if 0 <= idx < len(self.global_path) and self.right_void_list and self.left_void_list:
            return self.right_void_list[idx], self.left_void_list[idx]
        return (0.0, 0.0)

    def _calc_wall_distance(self, idx: int, max_search_distance: float) -> tuple[float, float]:
        """
        Compute the clearance to the right and left of the path waypoint at
        index `idx`.  The local path tangent is derived from the vector
        between the current waypoint and the next waypoint (wrapping
        around if necessary).  Distances are computed by stepping along
        the perpendicular directions in increments of approximately
        0.05 m (matching the C++ implementation), using the map's
        occupancy information to detect walls.  This method returns
        `(right_distance, left_distance)`.
        """
        n = len(self.global_path)
        # Wrap around indices to support closed loops
        curr_idx = idx % n
        next_idx = (curr_idx + 1) % n
        # Extract current and next points
        p_curr = self.global_path[curr_idx][:2]
        p_next = self.global_path[next_idx][:2]
        # Compute the path vector and its magnitude
        dx = p_next[0] - p_curr[0]
        dy = p_next[1] - p_curr[1]
        mag = math.hypot(dx, dy)
        if mag < 1e-6:
            # If the segment is degenerate, treat it as having zero length.
            # In this case use a default unit vector along the x-axis.
            ux, uy = 1.0, 0.0
        else:
            ux, uy = dx / mag, dy / mag
        # The step size matches the C++ implementation: unit vector / 20
        step = 0.05  # 1/20 metres
        # Right side: rotate (ux, uy) by 90 degrees to the right and scale by step
        step_right_x = uy * step
        step_right_y = -ux * step
        # Left side: rotate (ux, uy) by 90 degrees to the left and scale by step
        step_left_x = -uy * step
        step_left_y = ux * step

        # Compute right and left distances
        right_distance = self._find_wall_distance(
            p_curr[0], p_curr[1], step_right_x, step_right_y, max_search_distance
        )
        left_distance = self._find_wall_distance(
            p_curr[0], p_curr[1], step_left_x, step_left_y, max_search_distance
        )
        return right_distance, left_distance

    def _find_wall_distance(self, x0: float, y0: float, step_x: float, step_y: float, max_dist: float) -> float:
        """
        Step from (x0, y0) in the direction (step_x, step_y) until an
        occupied cell is encountered in the map or `max_dist` is exceeded.
        Returns the Euclidean distance travelled from the starting point to
        the first occupied cell.  If the search terminates without
        encountering a wall (i.e., `max_dist` is reached or the map
        boundary is crossed), then the return value is `max_dist`.
        """
        if not self.map_handler:
            return 0.0
        # Starting point
        curr_x, curr_y = x0, y0
        travelled = 0.0
        while travelled < max_dist:
            # Advance to the next location
            curr_x += step_x
            curr_y += step_y
            travelled += math.hypot(step_x, step_y)
            # Query occupancy
            if self.map_handler.is_occupied(curr_x, curr_y):
                break
        # Clamp to maximum distance
        if travelled > max_dist:
            travelled = max_dist
        return travelled

    def global_to_frenet_point(self, x, y):
        if len(self.global_path) == 0:
            self.node.get_logger().warn("global_to_frenet_point: No global path available.")
            return [0, 0]  # Return default values if path is missing

        closest_idx = self._get_closest_index(x, y)

        if closest_idx >= len(self.global_path) - 1:
            self.node.get_logger().warn("Closest index is out of bounds!")
            return [0, 0]

        out1 = self._calc_proj(closest_idx, closest_idx + 1, x, y)
        out2 = self._calc_proj(closest_idx - 1, closest_idx, x, y)

        s, d = 0, 0

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
            s += path_len # 음수 s도 래핑 (예: -1 -> 499)
        
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
        while self._calc_path_distance(0, idx + 1) <= s:
            idx += 1
        return idx

    def _get_closest_index(self, x, y):
        """가장 가까운 글로벌 경로 인덱스를 찾는 함수"""
        if len(self.global_path) == 0:
            self.node.get_logger().warn("Cannot find closest index: Global path is empty!")
            return 0

        idx = 0
        closest_dist = self._calc_distance([x, y], self.global_path[0][:2])
        for i in range(1, len(self.global_path)):
            dist = self._calc_distance([x, y], self.global_path[i][:2])
            if dist < closest_dist:
                idx = i
                closest_dist = dist
        return idx
    
    def get_velocity_at_s(self, s):
        if not self.global_path:
            return 0.0  # 경로가 없으면 0 반환

        path_len = self.get_path_length()
        if path_len == 0:
            return self.global_path[0][2] 

        s = s % path_len
        idx = self._get_start_path_from_frenet(s)
        return self.global_path[idx][2]

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
            # Degenerate segment; treat projection as the start point.
            proj_t = 0.0
            proj_point = pointA
        else:
            proj_t = np.dot(vectorB, vectorA) / denom
            proj_point = pointA + (proj_t * vectorA)

        d = np.linalg.norm(proj_point - pointC)
        if np.cross(vectorA, vectorB) > 0:
            d = -d
        s = proj_t * denom
        return [s, d]

    def _calc_path_distance(self, start_idx, end_idx):
        distance_counter = 0
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
        next = self.global_path[next_idx][:2]
        return self._calc_distance(cur, next)

    def _calc_distance(self, A, B):
        return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)

    def path_callback(self, msg: Path):
        if not msg.poses:
            self.node.get_logger().warn("Received empty global path!")
            return

        self.global_path = [[pose.pose.position.x, pose.pose.position.y, pose.pose.position.z] for pose in msg.poses]
        self.path_recived = True
        self.node.get_logger().info(
            f"Received global path with {len(self.global_path)} points"
        )
        # If a map has already been received, compute wall distances now
        if self.map_received:
            self.calc_all_wall_distances()


def main():
    rclpy.init()
    node = Node("coordinate_converter_test_node")
    converter = coordinate_converter(node)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
