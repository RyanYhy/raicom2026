#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# test_recon_logic.py —— 报告串 + 扫描去重/瞄准 纯逻辑离线单测
# ------------------------------------------------------------
# 不依赖 rospy / 相机 / 模型, 直接用 yolov8 环境跑:
#   /home/mowen/miniconda3/envs/yolov8/bin/python tests/test_recon_logic.py
# 通过则打印 "ALL TESTS PASSED" 并以 0 退出。
# ============================================================
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from raicom2026.report import (
    make_person_report, make_mine_report,
    validate_person_line, validate_mine_line,
    ReportStore, assemble_report,
)
from raicom2026.scan_logic import (
    HOSTAGE, ENEMY, FRIEND, Det, Cluster,
    clamp, px_to_pan, px_to_tilt,
    distance_from_box_height, tilt_offset,
    cluster_person_detections, choose_aim,
)

FAILED = []


def check(name, cond):
    if cond:
        print('  [OK] %s' % name)
    else:
        print('  [FAIL] %s' % name)
        FAILED.append(name)


def test_report_strings():
    print('== 报告串 字节精确 ==')
    # 人物: 全角顿号 + 和
    s = make_person_report('A', 2, 1, 0)
    check('人物报告串精确', s == 'A区发现2个敌军、1个友军和0个人质')
    # 雷区: 半角逗号
    s = make_mine_report('A', 1, 0)
    check('雷区报告串精确(半角逗号)', s == 'A雷区未排雷数量1个,已经排雷数量0个')
    check('雷区报告串含半角逗号(非全角)', ',' in s and '，' not in s)
    # 校验 harness
    check('校验通过(人物)', validate_person_line('D区发现1个敌军、7个友军和3个人质'))
    check('校验通过(雷区)', validate_mine_line('B雷区未排雷数量0个,已经排雷数量1个'))
    check('校验拒绝(全角逗号雷区)', not validate_mine_line('A雷区未排雷数量1个，已经排雷数量0个'))
    check('校验拒绝(半角顿号人物)', not validate_person_line('A区发现2个敌军,1个友军和0个人质'))
    check('校验拒绝(缺"和")', not validate_person_line('A区发现2个敌军、1个友军0个人质'))
    check('校验拒绝(数量非数字)', not validate_person_line('A区发现x个敌军、1个友军和0个人质'))
    check('校验拒绝(非法区)', not validate_person_line('E区发现2个敌军、1个友军和0个人质'))
    check('校验拒绝(非法区 雷区)', not validate_mine_line('C雷区未排雷数量1个,已经排雷数量0个'))
    # 非法输入抛异常
    try:
        make_person_report('Z', 1, 1, 1)
        check('非法区抛异常', False)
    except ValueError:
        check('非法区抛异常', True)
    try:
        make_mine_report('A', -1, 0)
        check('负数抛异常', False)
    except ValueError:
        check('负数抛异常', True)


def test_report_store():
    print('== ReportStore 增量落盘 + assemble ==')
    d = tempfile.mkdtemp()
    p_p = os.path.join(d, 'report_person.txt')
    p_m = os.path.join(d, 'report_mine.txt')
    p_o = os.path.join(d, 'report.txt')
    st = ReportStore(p_p, p_m)
    st.add_person('A', 2, 1, 0)
    st.add_person('B', 1, 1, 0)
    st.add_mine('A', 1, 0)
    # 覆盖写同区(应只留最新)
    st.add_person('A', 2, 1, 1)
    st.add_mine('A', 0, 1)
    text = assemble_report(p_p, p_m, p_o)
    expected = [
        'A区发现2个敌军、1个友军和1个人质',
        'B区发现1个敌军、1个友军和0个人质',
        'C区发现0个敌军、0个友军和0个人质',   # 空区补 0
        'D区发现0个敌军、0个友军和0个人质',
        'A雷区未排雷数量0个,已经排雷数量1个',
        'B雷区未排雷数量0个,已经排雷数量0个',
    ]
    check('assemble 恰好6行且顺序正确',
          text.strip().split('\n') == expected)
    # 文件行数恒定
    with open(p_o, encoding='utf-8') as f:
        check('report.txt 恰好 6 行', len(f.read().strip().split('\n')) == 6)

    # 回归: report_mine.txt 缺失时 assemble 不得崩 (修复前 TypeError)
    p_m2 = os.path.join(d, 'report_mine_absent.txt')
    if os.path.exists(p_m2):
        os.remove(p_m2)
    text2 = assemble_report(p_p, p_m2, p_o)
    check('mine 文件缺失不崩溃且补0', 'A雷区未排雷数量0个,已经排雷数量0个' in text2)
    # 回归: mine 文件为空(0字节)时同样不崩
    open(p_m2, 'w').close()
    text3 = assemble_report(p_p, p_m2, p_o)
    check('mine 文件为空不崩溃且补0', 'B雷区未排雷数量0个,已经排雷数量0个' in text3)


def test_px_math():
    print('== px→度 数学 ==')
    check('px_to_pan 中心=cmd', abs(px_to_pan(90, 424, 848, 12.11) - 90) < 1e-9)
    check('px_to_pan 右偏>0', px_to_pan(90, 600, 848, 12.11) > 90)
    check('px_to_pan 数值', abs(px_to_pan(90, 600, 848, 12.11) - (90 + (600 - 424) / 12.11)) < 1e-6)
    check('px_to_tilt 下偏>0', px_to_tilt(80, 300, 480, 12.11) > 80)
    check('clamp', clamp(200, 5, 175) == 175 and clamp(-3, 5, 175) == 5)
    check('框高反推距离', abs(distance_from_box_height(0.30, 610, 122) - 1.5) < 1e-9)
    check('tilt_offset 未标定=0', tilt_offset(1.5, 0.0, 0.0) == 0.0)
    check('tilt_offset 标定 a1/d', abs(tilt_offset(1.0, 0.0, -3.8) - (-3.8)) < 1e-9)


def _det(cls, conf, pan, tilt, u=0.0, v=0.0, level=None):
    return Det(cls, conf, u, v, 100, 200, pan, tilt, level)


def test_clustering():
    print('== 聚类去重 ==')
    # 同一人两帧(pan 差 2°, tilt 差 1°) -> 1 簇
    clusters = cluster_person_detections([
        _det(ENEMY, 0.8, 90.0, 80.0), _det(ENEMY, 0.75, 92.0, 81.0)])
    check('同人两检测合并为1簇', len(clusters) == 1 and clusters[0].conf == 0.8)

    # 两个不同敌人(pan 差 15°) -> 2 簇
    clusters = cluster_person_detections([
        _det(ENEMY, 0.8, 60.0, 80.0), _det(ENEMY, 0.7, 100.0, 80.0)])
    check('两个敌人分开为2簇', len(clusters) == 2)

    # tilt 两档重复: 同人两档(pan 差 1°, tilt 差 2°) -> 1 簇
    clusters = cluster_person_detections([
        _det(ENEMY, 0.8, 90.0, 78.0), _det(ENEMY, 0.72, 91.0, 80.0)])
    check('tilt两档重复合并', len(clusters) == 1)

    # 类别冲突: enemy 0.9 与 friend 0.6 pan 差 5° (tilt 也近) -> 保留 enemy, 1 簇
    clusters = cluster_person_detections([
        _det(ENEMY, 0.9, 90.0, 80.0), _det(FRIEND, 0.6, 95.0, 80.0)])
    check('类别冲突取高置信(enemy保留)', len(clusters) == 1 and clusters[0].cls == ENEMY)

    # 友军置信更高 -> 保留友军
    clusters = cluster_person_detections([
        _det(ENEMY, 0.55, 90.0, 80.0), _det(FRIEND, 0.9, 94.0, 80.0)])
    check('类别冲突取高置信(friend保留)', len(clusters) == 1 and clusters[0].cls == FRIEND)

    # 类别冲突需 tilt 也接近: pan 差 3° 但 tilt 差 30° -> 两个真目标, 不冲突, 2 簇
    clusters = cluster_person_detections([
        _det(ENEMY, 0.9, 90.0, 100.0), _det(FRIEND, 0.8, 93.0, 70.0)])
    check('pan近但tilt差>10°异类不算冲突', len(clusters) == 2)

    # 类别冲突 tilt 也近: enemy(90,100) 与 friend(93,105) -> 冲突取高置信, 1 簇
    clusters = cluster_person_detections([
        _det(ENEMY, 0.9, 90.0, 100.0), _det(FRIEND, 0.8, 93.0, 105.0)])
    check('pan+tilt 都近的异类取高置信', len(clusters) == 1 and clusters[0].cls == ENEMY)

    # 跨档像素连续性 (s_v 符号未标定的场景):
    # 同人两档: 绝对角差 2*tilt_step(20°), 但像素 v 位移=10*12=120px -> 合并为1簇
    clusters = cluster_person_detections([
        _det(ENEMY, 0.8, 90.0, 80.0, 424.0, 240.0, 80.0),
        _det(ENEMY, 0.75, 90.0, 100.0, 424.0, 360.0, 90.0)], ppd_v=12.0)
    check('跨档像素连续性合并(符号无关)', len(clusters) == 1)

    # 同档两个真目标(tilt 差 20°), 像素规则不适用 -> 不误合并, 2 簇
    clusters = cluster_person_detections([
        _det(ENEMY, 0.8, 90.0, 80.0, 424.0, 240.0, 80.0),
        _det(ENEMY, 0.75, 90.0, 100.0, 424.0, 360.0, 80.0)], ppd_v=12.0)
    check('同档两个真目标不误合并', len(clusters) == 2)

    # 跨档但像素位移不符(非同一人) -> 不合并
    clusters = cluster_person_detections([
        _det(ENEMY, 0.8, 90.0, 80.0, 424.0, 240.0, 80.0),
        _det(ENEMY, 0.75, 90.0, 100.0, 424.0, 260.0, 90.0)], ppd_v=12.0)
    check('跨档像素位移不符不合并', len(clusters) == 2)

    # 三类各1个, pan 分得开 -> 3 簇
    clusters = cluster_person_detections([
        _det(ENEMY, 0.9, 50.0, 80.0), _det(FRIEND, 0.9, 90.0, 80.0),
        _det(HOSTAGE, 0.9, 130.0, 80.0)])
    check('三类分开为3簇', len(clusters) == 3)

    # 空 -> 空
    check('空检测->空簇', cluster_person_detections([]) == [])


def test_letterbox_roundtrip():
    print('== letterbox 回映 (非正方形输入) ==')
    from raicom2026.yolo_onnx import letterbox
    for (W, H) in [(848, 480), (640, 480)]:
        img = np.zeros((H, W, 3), np.uint8)
        padded, r, (dw, dh) = letterbox(img, 640)
        for (x0, y0) in [(0, 0), (100, 50), (W - 1, H - 1), (W // 2, H // 2)]:
            xp, yp = x0 * r + dw, y0 * r + dh
            xb, yb = (xp - dw) / r, (yp - dh) / r
            check('%dx%d 点(%d,%d) 回映误差<0.5px' % (W, H, x0, y0),
                  abs(xb - x0) < 0.5 and abs(yb - y0) < 0.5)


def test_choose_aim():
    print('== 瞄准/安全门 ==')
    W, H = 848, 480
    # 友军高置信在瞄准区 -> SKIP_DANGER
    action, _, _ = choose_aim([(FRIEND, 0.9, 424, 200)], W, H, 12, 12)
    check('友军高置信在区内 -> 放弃', action == 'SKIP_DANGER')
    # 人质高置信(距中心 3° pan 内) -> SKIP_DANGER
    action, _, _ = choose_aim([(HOSTAGE, 0.8, 460, 200)], W, H, 12, 12)
    check('人质高置信在区内 -> 放弃', action == 'SKIP_DANGER')
    # 人质高置信但距中心很远(>6°pan, 不应误伤视场边缘) -> NO_ENEMY
    action, _, _ = choose_aim([(HOSTAGE, 0.8, 700, 200)], W, H, 12, 12)
    check('人质在区外 -> 不触发安全门', action == 'NO_ENEMY')
    # 无 enemy -> NOTHING
    action, _, _ = choose_aim([], W, H, 12, 12)
    check('空视场 -> NOTHING', action == 'NOTHING')
    # enemy 居中 -> FIRE
    action, _, _ = choose_aim([(ENEMY, 0.9, 430, 235)], W, H, 12, 12)
    check('enemy 居中 -> FIRE', action == 'FIRE')
    # enemy 偏右 -> ADJUST (dp>0)
    action, dp, _ = choose_aim([(ENEMY, 0.9, 600, 235)], W, H, 12, 12)
    check('enemy 偏右 -> ADJUST 且 dp>0',
          action == 'ADJUST' and dp > 0)
    # 中置信友军 0.5 在区内 -> 也放弃 (danger 门 0.50, 不赌 -10 误伤)
    action, _, _ = choose_aim([(FRIEND, 0.5, 424, 200)], W, H, 12, 12)
    check('中置信友军0.5在区内 -> 放弃', action == 'SKIP_DANGER')
    # 低于 danger 门限的友军(0.45)不拦, 但无 enemy -> NO_ENEMY
    action, _, _ = choose_aim([(FRIEND, 0.45, 424, 200)], W, H, 12, 12)
    check('低置信友军不触发安全门', action == 'NO_ENEMY')
    # enemy 居中但友军 0.55 在射程内(±6°pan×±10°tilt) -> 放弃 (宁可丢敌分不误伤)
    action, _, _ = choose_aim([(ENEMY, 0.9, 430, 235), (FRIEND, 0.55, 470, 200)],
                              W, H, 12, 12)
    check('敌居中但友军0.55在射程内 -> 放弃', action == 'SKIP_DANGER')
    # enemy 居中且友军 0.45 在区内(低于门限) -> FIRE (低置信不拦)
    action, _, _ = choose_aim([(ENEMY, 0.9, 430, 235), (FRIEND, 0.45, 470, 200)],
                              W, H, 12, 12)
    check('敌居中且低置信友军 -> FIRE', action == 'FIRE')


if __name__ == '__main__':
    test_report_strings()
    test_report_store()
    test_px_math()
    test_letterbox_roundtrip()
    test_clustering()
    test_choose_aim()
    print()
    if FAILED:
        print('FAILED %d: %s' % (len(FAILED), ', '.join(FAILED)))
        sys.exit(1)
    print('ALL TESTS PASSED')
