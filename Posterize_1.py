'''
Created on 2017/09/23

@author: nekoze1004
'''
import cv2
import numpy as np
import sys
from numba import jit


# 引数の文字列はYesか
def IsYes(yn):
    if (yn == "y") | (yn == "Y") | (yn == "yes") | (yn == "Yes") | (yn == "YES"):
        return True
    else:
        return False


# 引数の文字列はNoか
def IsNo(yn):
    if (yn == "n") | (yn == "N") | (yn == "no") | (yn == "No") | (yn == "NO"):
        return True
    else:
        return False


# 引数はYesでもNoでもないか
def IsNotYN(yn):
    if IsYes(yn) | IsNo(yn):
        return False
    else:
        return True


# 画像の色を反転させる
@jit
def reverse(img):
    copyImg = np.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            copyImg[i, j] = 255 - img[i, j]
            """if img[i, j] == 0:
                copyImg[i, j] = 255
            else:
                copyImg[i, j] = 0"""
    return copyImg


@jit
def posterize(img, pos):
    if img.ndim == 2:  # モノクロ画像(二次元配列)のとき
        GaryResult = np.zeros(img.shape, np.uint8)  # モノクロ画像返り値用配列

        if pos == 1:
            # もし1が入ってきたら、普通の処理では0で割る動作があるので例外的に処理
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if img[i, j] <= 127:
                        GaryResult[i, j] = 0
                    else:
                        GaryResult[i, j] = 255
        elif pos >= 256:
            # もし256以上が入ってきたら、画像をそっくりそのまま返す
            return GaryResult
        else:
            # 2~255までを想定
            # 読み込んだ元画像の全画素を巡回して、閾値と比べて当てはまる階級の値を
            # 書き出し先の同じ場所の画素に入れていく
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    for p in range(0, pos):
                        # pos=3のとき、0-85,86-170,171-255で分けられる
                        # それぞれ0,127,255が入れられる
                        if (img[i, j] >= ((255 // pos) * p) + 1) & (img[i, j] <= (255 // pos) * (p + 1)):
                            GaryResult[i, j] = (255 // (pos - 1)) * p
                            if p + 1 == pos:
                                # 255//posが本来255ぴったりになるべき者たちが255になれるようにしている（謎の言い回し）
                                if (img[i, j] >= (255 // pos) * (p + 1)) & (img[i, j] <= 255):
                                    GaryResult[i, j] = 255
        return GaryResult
    elif img.ndim == 3:  # カラー画像(三次元配列)のとき

        # 三次元配列を分解して二次元配列にする
        B, G, R = division(img)

        # それぞれの二次元配列をポスタライズする
        PB = posterize(B, pos)
        PG = posterize(G, pos)
        PR = posterize(R, pos)

        # 上記3つの2次元配列を3次元配列にして返す
        return combination(PB, PG, PR)


# カラー画像を3つの二次元配列に分ける
def division(ColorImg):
    Bimg = ColorImg[:, :, 0]
    Gimg = ColorImg[:, :, 1]
    Rimg = ColorImg[:, :, 2]
    return (Bimg, Gimg, Rimg)


# 3つの二次元配列を結合してカラー画像にする
def combination(Bimg, Gimg, Rimg):
    ColorImg = np.zeros((Bimg.shape[0], Bimg.shape[1], 3), np.uint8)
    ColorImg[:, :, 0] = Bimg[:, :]
    ColorImg[:, :, 1] = Gimg[:, :]
    ColorImg[:, :, 2] = Rimg[:, :]
    return ColorImg


#  base画像に、mask画像の黒い部分を上書きする
@jit
def masked(base, mask):
    r = np.copy(base)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 0:
                r[i, j] = 0
    return r


# 画像を読み込む　pngとjpgに対応している
def ReadImg():
    print("読み込む画像名 NewImageフォルダ内")
    inputImg = input(">>> ")

    # inputされた名前のpngかjpg画像を読み込む
    img = cv2.imread("NewImages/" + inputImg + ".png")
    if img is None:
        img = cv2.imread("NewImages/" + inputImg + ".jpg")
        if img is None:
            print("認識できません。")
            sys.exit()  # 画像が見つからなかったのでプログラムを終了する
    return (img, inputImg)


# モノクロにするか聞く
def doMono():
    print("モノクロにしますか？(y/n)")
    mono = input(">>> ")
    while IsNotYN(mono):
        print("yかnを入力してください。")
        print("モノクロにしますか？(y/n)")
        mono = input(">>> ")

    if IsYes(mono):
        return True
    if IsNo(mono):
        return False


# 線画像を求める
def MakeLine(img, sita=50, ue=100):
    # Canny法を用いて輪郭を検出する
    CannyImg = cv2.Canny(img, sita, ue)

    # Canny法で求めた線の白黒を反転させる
    ReverseCannyImg = reverse(CannyImg)

    # 線を太くしないときに使う線画像
    sen = ReverseCannyImg

    print("線を太くしますか？ (y/n)")
    hutoi = input(">>> ")
    while IsNotYN(hutoi):
        print("yかnを入力してください。")
        print("線を太くしますか？(y/n)")
        hutoi = input(">>> ")

    if IsYes(hutoi):  # 線を太くする際の処理
        near8 = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]],
                         np.uint8)
        # 8近傍膨張処理を行う
        ErosionReverseCannyImg = cv2.erode(ReverseCannyImg, near8, iterations=1)
        # senを太くした線画像に書き換える
        sen = ErosionReverseCannyImg
    return sen


# 名前をつけて画像を保存する
def NamedFile(inputName, resultImg, CorG=0):  # CorG==0ならカラー、1ならグレー
    if CorG == 0:
        fName = inputName + "ColorPosterize.png"
    else:
        fName = inputName + "GrayPosterize.png"

    cv2.imshow("results " + fName, resultImg)  # ポスタライズの結果を表示する
    cv2.imwrite(fName, resultImg)  # ポスタライズの結果を保存する
    cv2.waitKey(0)  # なにかキーを押すと↓
    cv2.destroyAllWindows()  # 表示ウインドウが閉じる


if __name__ == "__main__":

    print("ポスタライズします\n")
    img, inputName = ReadImg()

    if doMono():
        print("モノクロでポスタライズします。")
        # imgをグレースケール化
        GrayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 画像下処理　ぼかし
        GaussGrayImg = cv2.GaussianBlur(GrayImg, (5, 5), 0)

        sen = MakeLine(GaussGrayImg)

        print("ポスタライズ階調を入力 n<20推奨")
        p = input(">>> ")
        pos = int(p)  # pはStringなのでintにする

        # ポスタライズを行う　対象画像、階調指定
        PosterizeImg = posterize(GaussGrayImg, pos)

        # ポスタライズした画像と線画像を重ねる
        result = masked(PosterizeImg, sen)

        # 画像を保存する
        NamedFile(inputName, result, 1)

        print("終了します")
        sys.exit()
    else:
        print("カラーでポスタライズします。")
        # 画像下処理　ぼかし
        GaussImg = cv2.GaussianBlur(img, (5, 5), 0)

        sen = MakeLine(GaussImg)

        print("ポスタライズ階調を入力 n<20推奨")
        p = input(">>> ")
        pos = int(p)  # pはStringなのでintにする

        # ポスタライズを行う　対象画像、階調指定
        PosterizeImg = posterize(GaussImg, pos)

        # ポスタライズした画像と線画像を重ねる
        result = masked(PosterizeImg, sen)

        # 画像を保存する
        NamedFile(inputName, result, 0)

        print("終了します")
        sys.exit()
