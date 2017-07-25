# Posterize
写真をイラストっぽくするプログラム。

## 機能
カラー画像に対して
* モノクロ化
  * ただグレースケールにしてるだけ
* 輪郭強調
  * Canny法を用いてエッジの二値画像を求めている
* 階調変化(減少)
  * 255を愚直に割ってる

を選択的に行って、イラストっぽくするプログラム


## 実行結果
元画像<br>
![元画像](http://i.imgur.com/nAbQZNo.jpg)
<br>カラー　5階調(125色) <br>
![実行結果1](http://imgur.com/GNER3L6.png)
<br>モノクロ　5階調(5色)　<br>
![実行結果2](http://i.imgur.com/qQmLxtC.png)
<br>カラー　17階調(17の3乗 色) <br>
![実行結果3](http://i.imgur.com/vQaz28j.png)
<br>モノクロ　17階調(17色)　<br>
![実行結果4](http://i.imgur.com/4yFrtkA.png)
