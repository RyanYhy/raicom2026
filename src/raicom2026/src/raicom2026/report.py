#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# raicom2026/report.py —— 比赛报告串 生成 / 校验 / 落盘
# ------------------------------------------------------------
# 比赛只认 6 行字节级精确的报告串 (行为准则 C7):
#   人物区(4 行, A~D):
#       "A区发现x个敌军、x个友军和X个人质"
#       全角顿号 U+3001 "、" + 全角 "和"
#   雷区(2 行, A~B):
#       "A雷区未排雷数量x个,已经排雷数量x个"
#       半角逗号 ","
# 本模块提供:
#   make_person_report / make_mine_report   拼串; 拼错直接抛 ValueError(绝不产出错误串)
#   validate_person_line / validate_mine_line  字节级校验 harness
#   ReportStore   线程安全增量落盘 report_person.txt / report_mine.txt (写完即得分)
#   assemble_report   A~D 人物 + A~B 雷区 合并成 report.txt (恰好 6 行, 空区补 0)
# 纯标准库, 不依赖 rospy / onnx, 可离线单测 (tests/test_report.py)。
# ============================================================
import os
import re
import threading

PERSON_ZONES = 'ABCD'
MINE_ZONES = 'AB'

# 字节级校验: 区必须 A~D / A~B; 数量非负整数; 标点精确(全角顿号/半角逗号)
_PERSON_RE = re.compile(r'^[A-D]区发现([0-9]+)个敌军、([0-9]+)个友军和([0-9]+)个人质$')
_MINE_RE = re.compile(r'^[A-B]雷区未排雷数量([0-9]+)个,已经排雷数量([0-9]+)个$')


def make_person_report(zone, enemy, friend, hostage):
    """生成人物区报告串。非法输入直接抛 ValueError, 绝不产出错误串。"""
    enemy, friend, hostage = int(enemy), int(friend), int(hostage)
    line = '%s区发现%d个敌军、%d个友军和%d个人质' % (zone, enemy, friend, hostage)
    if _PERSON_RE.match(line) is None:
        raise ValueError('人物报告串非法: %r' % line)
    return line


def make_mine_report(zone, red, blue):
    """生成雷区报告串 (注意: 此处是半角逗号)。非法输入抛 ValueError。"""
    red, blue = int(red), int(blue)
    line = '%s雷区未排雷数量%d个,已经排雷数量%d个' % (zone, red, blue)
    if _MINE_RE.match(line) is None:
        raise ValueError('雷区报告串非法: %r' % line)
    return line


def validate_person_line(line):
    return _PERSON_RE.match(line.strip()) is not None


def validate_mine_line(line):
    return _MINE_RE.match(line.strip()) is not None


def _read_lines(path, validator):
    """读回已有合法报告行 -> {zone: line}。文件里有非法行直接抛错(宁可不读, 不产出坏报告)。"""
    d = {}
    if not os.path.exists(path):
        return d
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            ln = raw.strip()
            if not ln:
                continue
            if validator(ln):
                d[ln[0]] = ln        # 行首字母即区名
            else:
                raise ValueError('报告文件 %s 含非法行: %r' % (path, ln))
    return d


def _atomic_write(path, text):
    """原子写: 先写 .tmp 再 os.replace, 中途崩溃不会截断污染正式报告文件
    (否则后续所有区都因读回非法行而丢分)。"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(text)
    os.replace(tmp, path)


def _write_lines(path, zones, d, make_default):
    """把 {zone: line} 按区序落盘(空区补 0), 保证文件行数恒定。"""
    lines = [d.get(z) or make_default(z) for z in zones]
    _atomic_write(path, '\n'.join(lines) + '\n')


class ReportStore:
    """线程安全的报告增量落盘器。每个区写完立即落盘(先把分拿到)。
    人物与雷区分文件存, 最后由 assemble_report 合并成 report.txt。"""

    def __init__(self, person_path, mine_path):
        self.person_path = person_path
        self.mine_path = mine_path
        self._lock = threading.Lock()

    def add_person(self, zone, enemy, friend, hostage):
        line = make_person_report(zone, enemy, friend, hostage)
        with self._lock:
            os.makedirs(os.path.dirname(self.person_path) or '.', exist_ok=True)
            d = _read_lines(self.person_path, validate_person_line)
            d[zone] = line
            _write_lines(self.person_path, PERSON_ZONES, d,
                         lambda z: make_person_report(z, 0, 0, 0))

    def add_mine(self, zone, red, blue):
        line = make_mine_report(zone, red, blue)
        with self._lock:
            os.makedirs(os.path.dirname(self.mine_path) or '.', exist_ok=True)
            d = _read_lines(self.mine_path, validate_mine_line)
            d[zone] = line
            _write_lines(self.mine_path, MINE_ZONES, d,
                         lambda z: make_mine_report(z, 0, 0))


def assemble_report(person_path, mine_path, out_path):
    """合并出 report.txt: 恰好 6 行 (A~D 人物 + A~B 雷区), 空区补 0。
    返回完整文本; 任何一行非法都会抛异常。"""
    person = _read_lines(person_path, validate_person_line)
    mine = _read_lines(mine_path, validate_mine_line)
    lines = []
    for z in PERSON_ZONES:
        lines.append(person.get(z) or make_person_report(z, 0, 0, 0))
    for z in MINE_ZONES:
        lines.append(mine.get(z) or make_mine_report(z, 0, 0))
    for ln in lines:
        assert validate_person_line(ln) or validate_mine_line(ln), '报告串非法: %r' % ln
    text = '\n'.join(lines) + '\n'
    _atomic_write(out_path, text)
    return text
