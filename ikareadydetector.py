# ikareadydetector
# Copyright (c) 2018 UMEZAWA Takeshi
# Licensed under GNU GPL version 2 or later

import cv2
from time import sleep

# 1280x720 での「準備OK？」の領域
top=506
left=256
bottom=544
right=394

# ソース画像を比較関数に渡す前にかけるフィルタ
def pre_diff_filter(src):
	tmp = src
	tmp = tmp[top:bottom, left:right]
	tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
	ret, tmp = cv2.threshold(tmp, 192, 255, cv2.THRESH_BINARY)
	return tmp

# ソース画像をウィンドウに表示する前にかけるフィルタ（デバッグ向け）
def pre_show_filter(src):
	tmp = src
	return tmp

# 画像のチャンネル数を取得する
# .shape がグレースケールだと 2-tuple、カラーだと 3-tuple を返すのを吸収する
def getch(src):
	h, w, *ch = src.shape
	if len(ch) == 0:
		return 1
	else:
		return ch[0]


tmpl = cv2.imread("template.png")
tmpl = pre_diff_filter(tmpl)

cap = cv2.VideoCapture(1)
#cap = cv2.VideoCapture("D:\\cap\\amarec(20180519-2234).avi")
#cap = cv2.VideoCapture("test2.avi")

while True:
	ret, captured = cap.read()
	if not ret:
		print("cannot read from capture")
		break

	# 入力画像を 1280x720 に変換する
	# 非常にやっつけコード
	caph, capw, *_ = captured.shape
	if capw == 640 and caph == 480:
		captured = captured[60:420]
		caph = 360
	if caph == 360:
		captured = cv2.resize(captured, (1280, 720), cv2.INTER_CUBIC)

	# テンプレート画像との差異を計算して検出
	# 検出結果は画像下部に描画
	img = pre_diff_filter(captured)
	diff = cv2.absdiff(img, tmpl).sum() * 1.0 / (bottom-top) / (right-left) / getch(img)
	cv2.putText(captured, "diff=%6.2f" % diff, (0, 720), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
	detected = diff < 20
	if detected:
		cv2.putText(captured, "READY?", (640, 720), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

	# ウィンドウに描画
	cv2.imshow('Ika Private Match Ready Detector', pre_show_filter(captured))

	# q を押したら終了
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	sleep(1)

cap.release()
cv2.destroyAllWindows()
