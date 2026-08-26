-- Copyright 2016 The Cartographer Authors
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- ============================================================
-- localization.lua —— RAICOM2026 Cartographer 纯定位 (pure localization)
-- ------------------------------------------------------------
-- 基于 my_car.lua (建图时调好的参数), 加定位专用调优:
--   1) pure_localization_trimmer: 只保留最近 3 个子图, 不建图
--   2) motion_filter 调小: 麦轮打滑时 odom 漂移大, 更频繁做 scan matching
--      及时纠正 (建图时 0.2m/1°, 定位时 0.1m/0.5°)
--   3) real_time_correlative_scan_matcher 搜索窗口增大: 打滑瞬间位姿
--      跳变可达几十 cm/几十度, 窗口太小纠正不回来
--      (建图时 0.3m/20°, 定位时 0.5m/30°)
--
-- 用法 (cartographer_node gflags):
--   -configuration_directory config/cartographer
--   -configuration_basename localization.lua
--   -load_state_filename data/maps/competition_map.pbstream
-- ============================================================

include "my_car.lua"

-- 纯定位: 只保留最近 3 个子图, 不插入新子图 (官方 backpack_2d_localization 同款)
TRAJECTORY_BUILDER.pure_localization_trimmer = {
  max_submaps_to_keep = 3,
}

-- 2026-08-16/17 定位专用调优 (麦轮打滑 -> odom 漂移大 -> 更频繁 + 更强 scan matching):
--   注意1: 搜索窗口不能开太大 —— 1.0m/45° 时 Cartographer 处理不过来
--   (实时率掉到 76%, TF 时间戳滞后 5 秒, move_base 规划失败)。
--   注意2: 0.5m/30° + motion_filter 0.1m/0.3° 时定位乱跳 (搜索窗口大 + 匹配太频繁,
--   酒店特征不足时 scan 噪声放大成位姿跳变)。
--   回退到更稳的折中: 窗口 0.3m/20° (建图值), motion_filter 0.15m/0.5°。
TRAJECTORY_BUILDER_2D.motion_filter.max_time_seconds = 0.5
TRAJECTORY_BUILDER_2D.motion_filter.max_distance_meters = 0.15
TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = math.rad(0.5)

TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.3
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.angular_search_window = math.rad(20.)

TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight = 50.

-- 2026-08-17: ceres 求解器多线程 (建图时 num_threads=1), 定位时用 4 核加速
-- scan matching, 降低处理延迟 (配合搜索窗口回退, 恢复实时率 ~100%)。
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.ceres_solver_options.num_threads = 4

-- 2026-08-17: ceres 迭代 20->50 —— motion_filter 静止时(时间触发)每次匹配
-- 收敛到"差一点点"的局部最优 (车静止每 0.5s 匹配结果相同, 无法纠正)。
-- 增大迭代让每次匹配收敛到精确位置, 解决"静止时差一点点"。
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.ceres_solver_options.max_num_iterations = 50

-- 2026-08-17 use_imu_based 必须保持 false: Cartographer 1.0 的 2D 模式硬编码
-- CHECK(!use_imu_based) (local_trajectory_builder_2d.cc:322), 开启即 FATAL。
-- IMU 位姿外推只在 3D 实现, 2D 不可用。旋转漂移只能靠 scan matching 纠正。

-- 2026-08-16 关键: 发布 /tracked_pose (Cartographer 1.0 默认 false, 必须显式开)。
-- 任务三反坦克识别靠 /tracked_pose 拿车位姿做光轴->map 变换;
-- 不发布则 orbbec_mine_antitank 拿不到位姿 -> 锁定 0 个盘 -> no_antitank。
options.publish_tracked_pose = true

return options
