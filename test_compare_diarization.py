"""compare_diarization.disagreements の確認。実行: uv run python test_compare_diarization.py"""

from compare_diarization import disagreements

# 一致していれば食い違いは無い
assert disagreements([(0, 10, "A")], [(0, 10, "A")]) == []

# 途中の 2 秒だけ別の話者と判定された
assert disagreements(
    [(0, 10, "A")],
    [(0, 4, "A"), (4, 6, "B"), (6, 10, "A")],
) == [(4.0, 6.0, ("A",), ("B",))]

# 片方だけが発話と判定した区間
assert disagreements([(0, 3, "A")], []) == [(0.0, 3.0, ("A",), ())]

# min_len 未満の食い違いは聞き直す価値が薄いので捨てる
assert disagreements([(0, 10, "A")], [(0, 9.5, "A")]) == []

# 話者が入れ替わりながら続く食い違いは 1 区間にまとめ、関わった話者を全部出す
assert disagreements(
    [(0, 4, "A")],
    [(0, 2, "B"), (2, 4, "C")],
) == [(0.0, 4.0, ("A",), ("B", "C"))]

# 同時発話: 片方だけが重なりを検出した
assert disagreements(
    [(0, 4, "A"), (1, 3, "B")],
    [(0, 4, "A")],
) == [(1.0, 3.0, ("A", "B"), ("A",))]

print("ok")
