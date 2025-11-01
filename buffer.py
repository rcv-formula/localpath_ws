def generate_ref_spline_path(self):
        if self.ego_s is None or self.ego_d is None:
            return []

        dir_sign = self._heading_dir_sign()
        s0 = float(self.ego_s)
        s1 = s0 + dir_sign * float(self.forward_length)
        S_DIST = abs(s1 - s0)  # 다항식에 사용할 총 s 거리

        if S_DIST < 1.0:  # 경로가 너무 짧으면 생성하지 않음
            self.get_logger().warn("[RefPath] Path generation distance too short.")
            return []

        # --- 1. 안전 마진 설정 (튜닝 필요) ---
        SAFETY_MARGIN = 0.4  # 벽으로부터 최소 40cm 이격
        
        idx_start = self.converter._get_closest_index(s0, self.ego_d)
        right_d_start, left_d_start = self.converter.get_wall_distance(idx_start)
        RIGHT_LIMIT_START = right_d_start - SAFETY_MARGIN
        LEFT_LIMIT_START = -left_d_start + SAFETY_MARGIN
        
        start_d = np.clip(self.ego_d, LEFT_LIMIT_START, RIGHT_LIMIT_START)

        idx_end = self.converter._get_closest_index(s1, 0.0)  
        right_d_end, left_d_end = self.converter.get_wall_distance(idx_end)
        RIGHT_LIMIT_END = right_d_end - SAFETY_MARGIN
        LEFT_LIMIT_END = -left_d_end + SAFETY_MARGIN
        
        target_d = np.clip(0.0, LEFT_LIMIT_END, RIGHT_LIMIT_END)

        qp = QuinticPolynomial(start_d, 0.0, 0.0, target_d, 0.0, 0.0, S_DIST)

        s_points = self._sample_s(s0, s1, n=20)
        if not s_points.any():
            return []
            
        s_rel = np.abs(np.array(s_points) - s0)
        d_vals = [qp.calc_point(s) for s in s_rel]

        frenet_path = []
        path_is_safe = True
        
        FINAL_CHECK_MARGIN = 0.4 

        for s, d in zip(s_points, d_vals):
            idx = self.converter._get_closest_index(s, d)
            right_d, left_d = self.converter.get_wall_distance(idx)

            if (d > right_d - FINAL_CHECK_MARGIN) or (d < -left_d + FINAL_CHECK_MARGIN):
                path_is_safe = False
                self.get_logger().warn(f"[RefPath] Generated path collides mid-way at s={s:.1f} (d={d:.2f})")
                break  # 충돌 감지 시 즉시 중단
                
            target_v = self.converter.get_velocity_at_s(s)
            frenet_path.append([s, d, target_v])

        # --- 5. 경로 반환 ---
        if path_is_safe:
            path = self.converter.frenet_to_global(frenet_path)
        else:
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
            win = 0.5 * (1.0 + np.cos(np.linspace(-(math.pi), math.pi, n)))  # ì¤‘ì•™ 1, ê°€ìž¥ìžë¦¬ 0
        else:
            win = np.ones(n)

        signed = float(side) * float(bump)
        
        d_points[i0 : i1 +1] += signed * win

        return d_points


    def static_avoidance(self, current_v, margin=0.4):

        if self.static_xy is None or self.ego_s is None or self.ego_d is None:
            return None

        s_obs, d_obs = self.converter.global_to_frenet_point(*self.static_xy)
        if s_obs is None:
            return None
        
        dir_sign = 1.0 
        s0 = float(self.ego_s)
        path_len = float(self.converter.get_path_length())
        
        if path_len <= 1e-6:
            self.get_logger().warn("[STATIC] Invalid global path length; skip avoidance.")
            return None
        
        plan_dist = max(self.forward_length, current_v * 4.0) 
        s1 = s0 + dir_sign * plan_dist
        T = np.abs(s1 - s0)

        s_obs_raw = float(s_obs)
        delta_s = (s_obs_raw - s0) % path_len
        s_obs_aligned = s0 + dir_sign * delta_s
        self.get_logger().info(f"[STATIC] Obstacle at s={s_obs_aligned:.2f}, d={d_obs:.2f}")

        s_points_list = self._sample_s(s0, s1, n=20)
        if not s_points_list.any():
            return None
        
        s_points = np.array(s_points_list)

        # d0 = float(self.ego_d)
        # qp = QuinticPolynomial(d0, 0.0, 0.0, 0.0, 0.0, 0.0, T)
        
        base_d = np.array()

        obs_idx = self.converter._get_closest_index(s_obs_aligned, d_obs)
        right_void, left_void = self.converter.get_wall_distance(obs_idx)
        
        right_clearance = float(right_void - d_obs)
        left_clearance  = float(left_void  + d_obs)
        
        self.get_logger().info(f"[STATIC] Using safety margin: {margin:.2f}m")

        can_go_right = right_clearance > margin
        can_go_left = left_clearance > margin

        i_center_s_idx = int(np.argmin(np.abs(s_points - float(s_obs_aligned))))
        d_base_at_obs = base_d[i_center_s_idx]

        side = 0.0
        bump_mag = 0.0

        if can_go_right and (right_clearance > left_clearance):
            side = +1.0
            d_target = d_obs + margin
            bump_mag = max(0, d_target - d_base_at_obs) 
        elif can_go_left:
            side = -1.0
            d_target = d_obs - margin
            bump_mag = max(0, d_base_at_obs - d_target)

        self.get_logger().info(f"[STATIC] Applying bump: side={side}, mag={bump_mag:.2f}")
        d_mod = self._apply_local_d_bump(
                s_points, base_d, s_center=s_obs_aligned, bump = bump_mag,
                half_width_idx=3, 
                side=side,
                smooth=True
            )
    
        path_xyv = []
        for s, d in zip(s_points, d_mod):
            x, y = self.converter.frenet_to_global_point(s, d)
            if x is None or y is None:
                continue
            v_ref = self.converter.get_velocity_at_s(s)
            path_xyv.append([x, y, v_ref])

        return path_xyv