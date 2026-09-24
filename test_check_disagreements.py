"""check_disagreements.judge の確認。実行: uv run python test_check_disagreements.py"""

from check_disagreements import judge

# 両方とも発話と見ていて話者だけ違う区間は、文字起こしでは決められない
assert judge(("A",), ("B",), "そうですね", 0.1, -30) == "要聴取"

# 片方だけが発話と見た区間は、whisper が言葉を拾えたかで決める
assert judge((), ("A",), "そうですね", 0.1, -30) == "Nemotronが正しい"
assert judge(("A",), (), "そうですね", 0.1, -30) == "pyannoteが正しい"
assert judge((), ("A",), "", 0.1, -30) == "pyannoteが正しい"

# whisper が無音から作りがちな定型文は、言葉として数えない
assert judge(("A",), (), "ご視聴ありがとうございました", 0.2, -30) == "Nemotronが正しい"

# whisper 自身が「発話なし」と見ている
assert judge((), ("A",), "はい", 0.9, -30) == "pyannoteが正しい"

# ほぼ無音（-50dBFS 未満）なら、何か書き起こされても発話とみなさない
assert judge((), ("A",), "はい", 0.1, -60) == "pyannoteが正しい"

print("ok")
