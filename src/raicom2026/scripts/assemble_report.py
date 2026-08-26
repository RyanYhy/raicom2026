#!/home/mowen/miniconda3/envs/yolov8/bin/python
# -*- coding: utf-8 -*-
# ============================================================
# assemble_report.py —— 合并人物/雷区报告成 report.txt (恰好 6 行)
# ------------------------------------------------------------
# 两个检测节点分别增量写 data/reports/report_person.txt (A~D 人物)
# 和 report_mine.txt (A~B 雷区); 比赛结束前用本脚本合并成 report.txt:
#   A区发现x个敌军、x个友军和X个人质     (全角顿号 + "和")
#   B区发现x个敌军、x个友军和X个人质
#   C区发现x个敌军、x个友军和X个人质
#   D区发现x个敌军、x个友军和X个人质
#   A雷区未排雷数量x个,已经排雷数量x个   (半角逗号)
#   B雷区未排雷数量x个,已经排雷数量x个
# 空区自动补 0; 任何一行非法都会抛错(绝不产出坏报告)。
# 用法:
#   rosrun raicom2026 assemble_report.py                # 写 data/reports/report.txt
#   rosrun raicom2026 assemble_report.py --print        # 只打印不落盘
#   rosrun raicom2026 assemble_report.py --out /tmp/r.txt
# ============================================================
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from raicom2026.report import assemble_report, validate_person_line, validate_mine_line


def main():
    parser = argparse.ArgumentParser(description='合并报告串 (恰好 6 行)')
    parser.add_argument('--out', default=None, help='输出文件 (默认 data/reports/report.txt)')
    parser.add_argument('--print', action='store_true', help='只打印不写文件')
    args = parser.parse_args()

    try:
        import rospkg
        pkg = rospkg.RosPack().get_path('raicom2026')
    except Exception:
        pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report_dir = os.path.join(pkg, 'data', 'reports')
    person_path = os.path.join(report_dir, 'report_person.txt')
    mine_path = os.path.join(report_dir, 'report_mine.txt')
    out_path = args.out or os.path.join(report_dir, 'report.txt')

    text = assemble_report(person_path, mine_path, out_path)
    if args.print:
        sys.stdout.write(text)
    # 终验: 恰好 6 行且全部合法
    lines = text.strip().split('\n')
    assert len(lines) == 6, 'report.txt 必须恰好 6 行, 实际 %d' % len(lines)
    for ln in lines:
        assert validate_person_line(ln) or validate_mine_line(ln), '非法行: %r' % ln
    print('已生成 %s (6 行):\n%s' % (out_path, text))


if __name__ == '__main__':
    main()
