# 第三部 高度な数理モデル
時系列モデル，機械学習，強化学習，多体系モデルなどの実戦で多用されるモデルを紹介する．
それ以外にも，次元削減やネットワーク科学，非線形時系列解析などの複雑なシステムを分析する際に重要な解析法についても紹介する．

---
## 第7章 時系列モデル
時系列データは，他の種類のデータには特有の性質があり，分析の際に注意が必要である．

### 7.1 時系列データを構成する構造
#### さまざまな時系列
時系列といってもさまざまある．
ガス使用料や株価指数，カオスなど．
規則的な変動を周期変動という．

#### いろいろなアプローチ
時系列データのバリエーションは豊富．
それぞれに対して，アプローチの方法があり，それを紹介する．

#### トレンド＋周期成分＋ノイズ
- トレンド(trend): 比較的長い感覚で見た時の増加/減少傾向のこと
- 周期成分: 周期的な変動
- ノイズ: 説明できない誤差

#### 周波数成分
時系列データを波の集まりとして捉える．
波は，三角関数もしくは複素数の指数関数で表現することが可能
周波数は波の細かさを表現しており，特定の原因から発せられる振る舞いやノイズは特有の周波数を持っていることがある．
データにどうのような周波数が含まれているかを調べるのは有力な方針である．
波を表す基本の式
$$
Asin(\omega t + \varphi) = Ae^{i(\omega t + \varphi)}
$$
ただし，$A, \omega, \varphi$はそれぞれ，振幅，角周波数，位相である．

#### 特定の非線形構造がある場合
非線形時系列解析: 非線形なシステムの一部の特徴を利用して高度な分析を行うこと

#### 時系列が定常か
多くの時系列解析では，時系列データが確率過程から生じた現実値として解析する．
この時，データの背後にある確率過程自体が時間的に変化している場合(明確なトレンドや平均が時間で変化する)はうまく分析できないので，
それを取り除く必要がある．

#### 時間を説明変数にして普通に統計検定を行ってはいけない
多くの統計検定は，ノイズが正規分布と仮定したり，隣接した点が独立にノイズが決まると仮定しているが，
時系列データは隣接する点に関係があることがある．


---
### 7.2 観測変数を使ったモデル
#### 予測に使えるモデルたち
時系列データ分析の問題設定の一つは，「過去のデータに基づいて未来を予測すること」．
これからは，各モデルの概略を紹介する

#### ARモデル
データ点の間に何らかの時間的な関係をモデル化する．

$$
x_t = c + \phi x_{t-1} + \epsilon_t
$$
ただし，$c, \phi$は定数パラメータ．
$\epsilon$はノイズ項であり，平均が0，分散が$\sigma^2$の確率分布から毎回独立に決定されるとする．
これをホワイトノイズという．
つまり，
$x_t$は，平均が$c + \phi　x_{t-1}$，分散が $\sigma^2$のノイズ分布から生成されると，解釈することが出来る．
この関係性を仮定したモデルを自己回帰モデル(autoregressive model)，ARモデルという．

さらに，過去p時間まで遡って変数に加えたモデルをAR$(p)$と表現し，
$$
x_t = c + \phi_1x_{t-1} + \phi_2x_{t-2} + ... +  \phi_px{t-p} + \epsilon_t
$$
となる．

また，この変数を多変数のベクトルにも拡張できて，それをベクトル自己回帰(vector autoregressive: VAR)モデルという．

#### ARMAモデル
ARモデルとは異なり，ある時点でのノイズが$q$時点後まで影響を与えると仮定したモデルを，
ARMA(autoregressive moving average model)という．
$$
x_t = c + \Sigma_{i=1}^{p} \phi_ix_{t-i} + \epsilon_t + \Sigma_{i=1}^{q}\Phi_{i}\epsilon_{t-i}
$$
新たに加えた過去のノイズの項$\Sigma_{i=1}^{q}\Phi_{i}\epsilon_{t-i}$を，移動平均(moving average)という．
これで，普通の線形回帰では表現できないノイズの間の関係性を表現できる．

#### ARIMAモデル
ARモデルやARMAモデルでは，定常生が満たされていない場合は使えない．
しかし，各時刻で前後の値の差をとって(差分)，各時刻での変動分の時系列を作ると，近似的に定常と見なせることがある．
この手続きによってトレンドを除くことができる．
このように，差分をとってからARMAモデルを適用することをARIMA(autoregressive integrative moving average)モデルという．

#### SARIMAモデル
周期変動を除いて作成した時系列に対してARIMAモデルを適用する方法を，
SARIMA(seasonal autoregressive integrative moving average)モデルという．
古典的な時系列モデルだが，適切に使用すればパフォーマンスを発揮する

まとめると

- AR: 自己回帰の関係性を記述
- ARMA: 移動平均の効果を追加
- ARIMA: トレンドの効果を追加
- SARIMA: 周期(季節)変動の効果を追加


---
### 7.3 状態空間モデル
#### 状態変数を含むモデル
時系列データを分析する際に非常に強力な手法の人一つが，
状態空間モデル(state space model)．
これは非常に広い概念で，個別のモデルではない．

- 状態変数(state variable): 観測されない潜在変数のこと．
- 観測変数: 実際にデータを観測できる変数のこと．

この状態変数を組み込むことで，定常性を満たさない時系列に対しても，モデルを適用することが出来る．
状態空間モデルは，非常に汎用性の高い手法である．

#### 状態空間モデルの一般的な表現
システム方程式(system equation): 状態変数(ベクトルでも可)の時間変化を記述する方程式．
観測方程式(observation equation): ある時刻に状態変数から観測変数(ベクトルでも可)が生成される関数のこと．

状態空間モデルはこのシステム方程式と観測方程式を合わせたもの．

#### 離散時間・線形・ガウスモデル
線形ガウス型常態空間モデル(動的線形モデル dynamic linear model: DLM): 
最もスタンダードな常態空間モデル．時間が離散的に与えられている場合．

$$
\left\{
    \begin{array}{1}
        x_t = G_tx_{t-1} + w_t
        y_t = F_tx_t + v_t
    \end{array}
\right.
$$
ただし，
$x_t$が潜在変数，$y_t$が観測変数，
$G_t, F_t$は時間に依存しても良い係数行列で，
$G_t$を状態遷移行列(state-transition matrix)，
$F_t$を観測行列(observation matrix)という．
$w_t, v_t$は正規分布に従うノイズ項．

動的線形モデルを行うには，RのdlmやPythonのstatsmodelsやPyDLMがある

#### その他の場合の状態空間モデル
線形性やノイズに対する正規性，離散時間を仮定したが，そういう仮定がいらないモデルもある．

連続時間の状態空間モデルは，制御理論で深く研究されている．
特に「どのようにしてシステムの状態を所望の状態にすることができるか」に焦点を当てた方法論は，
一般のデータ分析においても，非常に有用である．


---
### 7.4 その他の時系列分析法いろいろ
#### 自己相関で時間構造を特徴づける
自己相関(autocorrelation): 着目する変数が$\tau$時点離れた点どうしてどれくらい似ているかを表した指標．
周期的な変動を見ることが出来る．

#### 異常拡散による特徴づけ
時系列において，$\tau$だけ時間が離れた時に値の変化量を$\Delta x$として，
その$\Delta x$の標準偏差$\sigma(\Delta x)$を考える．
この量は，時間が離れた時に，どれくらい結果がばらついて予測できなくなるかを示している．
各時点でのノイズが完全にランダムに独立である場合は，この量は$\tau ^0.5$に比例する．
変数の値がこのように広がっていくことを拡散(diffusion)という．

一方で，この指数が0.5から離れることがあり，これを異常拡散(anomalous diffusion)という．
この異常拡散を捉える方法として，ハースト指数(Hurst exponent)$H$があり，
$$
\sigma(\Delta x) \propto \tau^H
$$
である．この$H$によって，ノイズが過去のノイズと関係性しているか(記憶性)を特徴づけることが出来る．

#### フーリエ変換による周波数解析
フーリエ変換(Fourier transform): データにどのような波がどれくらい含まれているのかを分析する．
元の関数を時間の関数から周波数の関数に変換し，それぞれの周波数がどれくらい使用されているかを見ることで，データの特徴を把握する．
パワースペクトルという量を見ることで，特徴的な音の高さに対応する周波数の成分が多く含まれていることがわかる．

それ以外にも周波数の成分が時間的に変化するデータの場合，スペクトログラム分析やウェーブレット変換などの手法も用いられる．

#### カオス・非線形時系列解析
様々な時系列データでカオスが見られる．
ここではカオスの定義については触れないが，簡単にいえば，非周期ダイナミクスのこと．

遅れ座標(delay coordinate): 時系列データの前後いくつかの点をひとまとめにして，n次元の空間の中の一点と捉える方法．
カオス時系列に対して特定の構造が見える場合がある．

シンプレックス・プロジェクション(simplex projection): その構造を利用して，時系列データをモデルなしで予測・説明する強力な方法．

リアプノフ指数(Lyapunov exponent): 時系列がカオス的であるかどうかを評価する指標．
まず値が近い二つの異なる状態を考え，普通のシステムなら似たような状態変化を辿るが，カオスならその差が拡大していく．
この離れていく度合いを定量化するのがリアプノフ指数で，初期値が違いのに時間が経つと大きく異なる性質を初期値鋭敏性という．

#### 2つ以上の時系列から因果関係を調べる
時系列データにおいて，ある変数が別の変数に影響を与えているかを知りたい．

- 移動エントロピー(transfer entropy): 情報量の観点から，XがYに影響を与えているなら，Xの情報を使うことでYの予測精度を上げられると考える．
- Granger因果(Granger causality): VARモデルなどを利用して，予測の精度を評価する・
- CCM(convergent cross mapping): 遅れ座標によるシンプレックス・プロジェクションを2つの時系列間で行う

一般的に，データから因果関係を推定するのは非常に難しく，データの性質に応じて適切に手法を選ばないと誤った結論を簡単に導いてしまう．
「この手法さえ使っていれば間違いない」という方法は存在しないので，適用条件については深く検討する必要がある．

---
### 第7章のまとめ
- 時系列データには，トレンドや周期変動などの時間的な構造が含まれていて，通常の統計分析ではうまく分析できないものもある
- 観測変数の間の関係性を直接モデル化したものとして，ARモデル，ARMAモデル，ARIMAモデル，SARIMAモデルなどがある
- 状態空間モデルでは，状態変数を用いることで，より自由度の高いモデリングが可能となる
- 時系列データの分析法には，着目する性質に応じて，他にも周波数解析，非線形解析，因果性解析などさまざまなものがある

---
---
## 第8章 機械学習モデル
応用志向型モデリングの本丸．
機械学習モデルの基本的な思想は，目的を達成するための方法をデータから機械に学ばせるという考え方．

### 8.1 機械学習で扱われるモデル・問題の特徴
#### 機械学習の基本的な考え方
数理モデルの中でも，実応用におけるパフォーマンスを重視するのが機械学習．
複雑な現実問題を精度よく記述することが必要なため，多くのパラメータを含む複雑なモデルが必要な場合が多い．

#### 複雑な問題，複雑なモデル
高次元なデータから低次元で非線形な特徴を捉えるためには複雑なモデルが必要．

#### モデルの自由度とオーバーフィッテイング
過学習(overfitting): パラメータ推定した時に使ったデータ(訓練データ)にはよく当てはまるものの，新しく得られたデータ(テストデータ)に対しては当てはまらないこと．
モデルをデータのばらつき・誤差に合わせすぎてしまうのが原因．

汎化(generalization): モデルが過学習せずに，未知のデータにもよく当てはまること．

#### 機械学習モデルを使った分析の実施
Pythonのscikit-learnやTensor-Flow, Kerasなど．
大きな計算資源が必要になることもある．

---
### 8.2 分類・回帰問題
#### 分類と回帰
- 分類(classification): ラベルを予測する
- 回帰(regression): 値を予測する

#### 決定木
決定木(decision tree): 学習データから，データを最もうまく分類できる条件の組を学習する

利点: わかりやすい，結果の解釈が容易

欠点: 過学習しやすい

#### ランダムフォレスト
ランダムフォレスト(random forest): 決定木をたくさん生成して，それらの多数決で決定する方法(アンサンブル学習; ensmble learning)．
元のデータセットからランダムにサンプルを取り出し(ブートストラップ)，それについて決定木を作り，多数決をとる．

#### サポートベクターマシン
サポートベクターマシン(support vector machine: SVM): データをクラスごとにプロットし，それぞれのクラスを直線(平面，超平面)で分類する方法．このようにデータが直線(平面・超平面)で分離できることを，線形分離可能である，という．データに非線形な変換を施すことで，線形分離可能な問題に帰着させて適用することも可能．

#### ニューラルネットワーク
ニューラルネットワーク(neural network): 単純な計算を行う要素(ノード)をネットワーク状に組み合わせて，活性化関数(activation function)を通してノード間のやりとりを行い，最終的に値・ラベルを予測する方法．
単純な非線形関数を組み合わせて何度も適用することで極めて高い表現力を得ることが出来る．

- 入力層: データを入力する層
- 出力層: データを出力する層
- 中間層: 入力層と出力層の間の層

この中間層を増やしたものを深層学習モデル(deep learning model)という．

---
### 8.3 クラスタリング
#### クラスタリングでデータを解釈
クラスタリング: データ点の散らばり具合だけを見て，近いデータたちを同じカテゴリにまとめる手続きのこと．教師なし学習．
クラスタリングアルゴリズムの選択やクラスター数の仮定などによって結果が変化するため，常に任意性があることを忘れない．

#### k-means法
k-means法: データ点と各クラスターの中心からの距離を比較して，一番近いクラスターに分類させる．

1. クラスター数kを適当に決める
2. 各データ点を適当にランダムに各クラスターに割り当てる
3. それぞれのクラスターの中心点を求める
4. 各データ点について，一番近い中心点を探し，そのクラスターに割り当てる．
5. 割り当てが変化しなくなるまで3, 4を繰り返す

#### 混合分布モデル
k-means法では，データの生成規則に関して特に数理モデルを仮定しない．

混合分布モデル(mixture model): 各クラスターのデータを生成する確率分布をそれぞれ仮定して，そのどれかからデータが生成されている，と考える方法．
特にガウス分布でデータの確率分布を表現するものを混合ガウスモデル(Gaussian mixture model)という．

混合されている確率分布を推定すれば，そのそれぞれが各クラスターに対応する．

データの生成分布 = クラスタ1の確率分布 + クラスタ2の確率分布 + ... + クラスタkの確率分布

この右辺を混合分布モデルという．

#### 階層的クラスタリング手法
階層的クラスタリング(hierarchical clustering): クラスター間の類似度をもとに，クラスターのまとまり方を調べる方法．
類似度の計算にはさまざまな方法がある．


---
### 8.4 次元削減
#### 次元削減とは
次元削減(dimensionality reduction): 理解できない高次元データを，本質的な情報を失わずに低次元のデータで表現すること．

#### 主成分分析
主成分分析(principal component analysis; PCA): データと最もばらつく直線を見つけ，その直線を新たな座標軸とする手法．
次に，その直線と直行する直線を新たな座標軸とする．
つまりデータがばらつけばばらつくほど，情報量が多いということ．
直線的なデータのしか捉えられないため，非線形な特徴を次元圧縮することはできない．

#### 独立主成分分析
独立主成分分析(independent component analysis; ICA):
主成分分析とは異なり，必ずしも直行する直線を取る必要がない．
予め任意の成分の個数を指定して，データをその数の独立した成分で表現すること．
事前にデータがいくつの成分に分けられるかがわかっている場合有用であるが，任意性が残る．

#### 非線形な次元削減法
- カーネルPCA: データに非線形変換を行い，主成分分析を行う
- 多様体学習(manifold learning): データの非線形な構造に沿って次元圧縮する．各データ点の近傍にあるデータから多様体という構造の情報を計算して，次元に圧縮に用いる方法．Isomap，LLE，t-SNEなどのアルゴリズムがある．
- 位相的データ解析(topological data analysis): データの「かたち」に着目した次元削減法

---
### 8.5 深層学習
#### 深層学習とは
深層学習(deep learning): ディープニューラルネットワーク(deep neural network)を用いた機械学習．
学習アルゴリズムの発展，大量学習データ取得の容易化，GPUやメモリなどの性能向上によって，注目を浴びている．
モデル自体の解釈が非常に困難であり，非常に応用志向型のアプローチである．

#### 畳み込みニューラルネットワーク
畳み込みニューラルネットワーク(convolutional neural network; CNN): 畳み込み層とプーリング層という中間層を持つ．
ある対象のパターンの位置が変化しても，同じように処理することが可能になる．

#### リカレントニューラルネットワーク
リカレントニューラルネットワーク(recurrent neural neteorl; RNN): 順方向だけでなく，後ろに戻る経路をネットワークに加えた方法．
過去の中間層の値を保持したり，過去の出力を中間層に戻したりするなどの，さまざまな方法が知られている．

#### オートエンコーダ
オートエンコーダ(autoencoder): 入力と出力が同じになるように学習させたニューラルネットワーク．
入力層→中間層で少ない数の変数で表現され(エンコード)，中間層→出力層で元に戻す(デコード)ことで，中間層では必要な情報を出来るだけ失わずに次元圧縮ができたことになる．

#### 敵対的生成ネットワーク
敵対的生成ネットワーク(generative adversarial network; GAN): 生成器(generator)と識別器(discriminator)を同時学習させ，生成器では画像を生成し，識別器ではそれを見抜くことを目標にして学習させる，つまりライバル関係にある2つのモデルを学習させる．
最終的に，本物と見分けがつかないデータを生成することが可能になる．

---
### 第8章のまとめ
- 機械学習では，応用時の性能を重視し，パラメータを多く含む複雑なモデルを利用する
- 学習データにモデルを合わせすぎてしまい，その他のデータに対して性能が出なくなることを，過学習という
- 機械学習モデルが解決する課題には，代表的なものとして，分類，回帰，クラスタリング，次元削減がある
- 深層学習は学習にかかるコストは大きいが，難しい問題を解決するためのパワフルな手法である

---
---
## 第9章 強化学習モデル
強化学習とは，環境からのフィードバックに応じて最適な反応を探索するためのフレームワーク．

### 9.1 行動モデルとしての強化学習
#### 強化と学習
強化学習(reindorcement learning): トライアンドエラーを何度も繰り返しながら適切な行動を学習していく時間変化を数理モデルで表現したもの．
モデル化される意思決定主体をエージェント(agent)という．
行動の結果，正負どちらかの報酬が得られるが，その行動の結果を決める場所を環境(environment)という．

#### ギャンブル課題
カードを引いて書かれた数字の分だけお金がもらえるギャンブルを考えて，その期待値を求めることで，参加の損得を考える．
ある時刻$t$において，エージェントが予測する価値を$Q_t$，引いたカード(報酬)を$r_t$とすると，
以下の式から次の予想値を計算する
$$
Q_{t+1} = Q_t + \alpha (r_t - Q_t)
$$
$r_t - Q_t$は予想との誤差を表し，$\alpha$は誤差を考慮して，どれだけ予想値を一度に変更するかの正パラメータである．

詳細は，GambleTask.ipynbを参考にする．


#### 行動選択を含める
選択肢が複数ある場合を考える．

2腕バンディット課題: 2つのケースがあり，どちらを選択するかを決められる．

予想した価値$Q_t(A), Q_t(B)$を考えて，価値が高い方を高確率で選択しつつ，価値が低い方も一定の確率で選択するように考えると，

$$
P(a_{t+1} = A) \propto exp(\beta Q_t(A)))　\\
P(a_{t+1} = B) \propto exp(\beta Q_t(B)))
$$
ここで，$a_t, \beta$はそれぞれ，とる選択肢，現時点で$Q$の値をどれくらい信用するかを決めるパラメータである．
このような式をソフトマックス(softmax function)といい，この確率に従って毎回A, Bを選択する．

一度に得られる情報はA, Bのどちらか一方なので，価値の更新も片方のみである．
このように価値を更新していく学習の仕方を，Q学習(Q learning)という．

実際の解析はTwoBandit.ipynbに行う．


#### モデルのバリエーションと発展
アスピレーション学習(aspiration learning): ある基準(アスピレーション)を設け，ある行動をした時に得られた報酬がそれよりも大きいかどうかに応じて，その行動をとる確率を直接上下させる方法．

また追加ルールがある場合は，それに対応した変数を入れて，価値の種類を増やして，モデル化を行えばよい．

#### 行動モデルとして使う
これらのモデルは時系列モデルと捉えることもできる．
よって，実際の行動時系列データにフィッティングすることで，パラメータを推定したり，予測を行うことができる．
モデルのパラメータには直接的な意味があるので，推定されたパラメータを使って，その行動の背後にある原理に潰え推測することもよく行われる．

今回の節では，学習が進むプロセスそのものに興味がある．
機械学習の文脈における強化学習は，学習された最適な戦略に興味がある．
しかし，背後には共通した考え方，モデル構造がある．

---
### 9.2 機械学習としての強化学習
#### 機械学習としての強化学習
機械学習の文脈での強化学習は，状況に応じてそれぞれの行動の価値を正しく決めることが目標．
環境が複雑になると，Q学習では，環境の状態のバリエーションが膨大になるなどの問題点が生じる．
これを解決するために様々な工夫がされている．

#### 価値観数の性質を決める
システムの状態$s$における行動$a$の価値を$Q(s, a)$とすると，すべての状態．行動について価値$Q$がわかれば，高得点を出すための操作を行うことができる．

次に，ある行動$a$をした時にシステムが$s'$という状態にあるとして，この時に得られる報酬を$r(s, a)$とする．
最終的に得られる総得点を最大化するなら，この後に得られる報酬も考慮に入れて，次の行動を選択すべき．
つまり以下の関係式を仮定する(実際は期待値が入る)．
$$
Q(s, a) = r(s, a) + \gamma Q(s', a^*)
$$
これは状態$s$の時の行動$a$の価値 = 状態$s$の時の行動$a$で得られる報酬 + 引き起こされる状況$s'$の最大価値×割引率

#### 価値観数の更新
$$
Q(s, a) \leftarrow Q(s, a) + \alpha [r(s, a) + \gamma Q(s', a^*) - Q(s, a)]
$$

右辺の$\alpha$の中は，$Q(x, a)$のあるべき値と現在の値の差になっていて，このように差の値を使って$Q$の値を更新していく方法をTD学習(temporal-difference learning)という・

#### 深層学習を使ったQ学習
実際に全ての場合についての$Q(s, a)$を更新していくのは現実ではない．
そこでニューラルネットワークを使って，この関数を近似するDQN(deep Q network)という方法がある．

#### Q学習以外の方法
Q学習の方法を価値ベース(value base)の強化学習という．
それ以外に，どのような戦略で行動を選択していくかを学習する方策ベース(policy base)の強化学習もある．

--- 
### 第9章のまとめ
- 強化学習は，未知の環境の状況に応じて意思決定する主体(エージェント)が適切な行動を学習する様子をモデル化する
- 強化学習は，人間の学習行動をモデル化する時系列モデルとして利用される
- 強化学習は，機械学習の文脈で最適な行動を機械に学習させる方法としても利用される

---
---
## 第10章 多体系モデル・エージェントベースモデル
複雑なシステムを見せるシステムでは，個々の要素の集まりが本質的な役割を果たしている．
個々の要素がたくさん集まった時に，全体として見える振る舞いを分析するための手法が，多体系モデル・エージェントモデルである．

### 10.1 ミクロからマクロへ
多体系モデル(many-body system)，エージェントベースモデル(agent-based model):
個々の物体・エージェントの振る舞いを記述するモデルをたくさん用意し，互いに相互作用させたもの．
ミクロな現象と全体として発生するマクロな現象の間のギャップを埋めたり，ミクロな振る舞いがわかっている状態で，マクロな現象がどのようになるかを予想したい時に用いる．

#### モデルの構成要素
多体系モデルの本質は要素同士がどのように相互作用しているかを記述すること．
要素の振る舞いを記述するのに常微分方程式モデルや確率過程モデルを利用する．

代表的な相互作用の仕方

- それぞれの要素が全ての要素と，同じ強さで相互作用する
    - well-mixed populationという．数理的に解析することが容易な場合が多い
- 要素が二次元や三次元空間の「位置情報」をもっていて，近くの要素だけ相互作用する
    - モデルに空間的な情報が含まれている時，遠くの要素とは相互作用しないという仮定．要素が動くと相互作用の相手も変化する
- 要素の間に特定のつながり(ネットワーク)が定められていて，繋がっている相手とだけ相互作用する
    - ネットワークとは，要素同士の繋がり方の情報をまとめたもの．つながりの一つ一つのことをリンク，個々の要素をノードという

#### 時間・空間の離散化
多数の要素の数理モデルを同時に動かすため，理論解析が難しくなるだけでなく，シミュレーションの計算コストが莫大になることがある．
そのため，時間や空間を離散的に区切り，単純化する離散化(discretization)と行う．
離散化された時間を時間ステップ(time step)，空間の1マスをセル(cell)やサイト(site)という．

セルオートマトン(cellular automaton): 時間と空間を離散化した決定論的なシステム．

時間を離散化すると，それぞれの要素をどの順番で更新して動かすかを指定する必要がある．

- ランダムアップデート(random update): ランダムに各要素を選んで更新する
- 逐次アップデート(sequential updata): 決まった順番に更新する
- パラレル・アップデート(parallel update): 1ステップにすべての要素を同時に更新して動かす

離散化は便利な単純化であるが，現象の本質を損なう可能性もあるので，気をつける．

#### マクロな変数によってシステムの振る舞いを特徴づける
秩序変数(order parameter): システムの状態を特徴づけるマクロな変数．
パラメータの変化によって秩序変数が急激に変化して，システムが異なる状態に変化することを相転移(phase transition)という．

#### モデルの分析の仕方
1. 理論解析
    - 相互作用や個々の要素の振る舞いが完全に同じルールに従っている場合，理論解析が可能な場合がある．
2. シミュレーション解析
    - 数値実験し，その結果から理解を深めたり，現象を予測したりする．シミュレーション中に起きたことは全て測定可能なので様々な角度から結果の検討が可能
    
---
### 10.2 さまざまな集団モデル
#### 群れのモデル
複数の個体で群れをなして移動する鳥や魚の群れはどのようにして作られ，維持されているのか．

Vicsekモデル: 個体が二次元の平面状を移動していき，すべての粒子は同じ速さで動き，周りの状況に合わせて方向だけ変化させる．
$$
\theta(t + \Delta t) = <\theta(t)>_r + \epsilon
$$
$\theta(t)$はある時刻の方向，$<\theta(t)>_r$は周りの粒子の平均方向，$\epsilon$はノイズ．
各時間ステップでは，速度の向きに応じて位置を動かして更新し，全ての粒子に対して，パラレルに実施する．

#### 同期現象のモデル
同期現象(sychronization): 何かの周期的な動きのタイミングが一致すること．

蔵元モデル(Kuramoto model): 同期現象を理解するための代表的なモデル．
ある要素の振動を位相(phase)という変数の動きで表現する．
それぞれの要素(振動子; oscillator)は自身の位相の動きを持っているが，他の振動子との位相の差に応じて自身の位相を調整する．
相互作用が大きくなると，徐々に位相の動きがあってきて，最終的に全体として同期する．

#### 人間行動・意思決定のモデル
社会的ジレンマ(social dilemma): 全員が合理的な行動をとると最終的に全員が損する問題．
このような状況における人々の行動は社会科学の諸分野で研究されてきた．

囚人のジレンマゲーム(prisoner's dilemma): 「協力」か「裏切り」を選択でき，それぞれのパターンに応じて得点が与えられる．
このゲームを繰り返しプレイさせた時の様子を分析することで，人間社会における協力現象をモデル化することがよく行われる．
各プレイヤーに行動を選択させ，その行動に基づいて得点を与え，その得点をもとに行動をアップデーして次の行動をとる．

この行動のアップデートを行う方法は二つある．

1. 進化論的アプローチ
    - 自分が対戦した相手が得た得点を参照して，一番得点が高いプレイヤーの取った行動を真似する．
2. 強化学習的アプローチ
    - それぞれの行動の価値を，毎回のゲームに応じてアップデートしていく．
    
---
### 10.3 相互作用のネットワーク
#### ネットワーク構造で問題を眺める
複雑ネットワーク科学(complex network science): ネットワークそのものを調べることで，システムの特徴を説明する．

#### どれくらい他のノードとつながっているか
次数(degree): あるノードについて，繋がっているノードの数．$k$とかく．
この次数を全てのノードについて調べ，次数の出現分布$P(k)$を次数分布(degree distribution)という．
世の中のネットワークはこの次数分布がべき分布($P(k)\propto k^{-\gamma}$)になっていることがある．
このようなネットワークをスケールフリーネットワークといい，その上でのダイナミクスの伝播が早まったり，リンクの多いノード(ハブ; hub)の影響力が支配的になったりする．

#### 類は友を呼ぶ？
次数が$k$のノードと繋がっているノードたちの字数はどのくらいか？という指標が重要になることもある．

リッチクラブ(rich club): 次数の高いノード同士が互いに繋がった構造．

アソータティビティ(assortativity): 同じ次数のノード同士がどれくらい繋がりやすいかを表す指標．

#### ネットワーク上の移動のしやすさを特徴づける
最短経路長(minimum path length): あるノードから別のノードに最短経路で移動する際に通らなければならないリンクの数．

平均経路長(average path length): この最短経路長をすべてのノードのペアに対して計算し，平均したもの，
ネットワーク上の移動のしやすさを表す指標．
現実のネットワークでは，想像に反して短いことが多く，空港(全世界で4000ある)の平均経路長は約3である．

#### 「中心性」で重要なノードを特徴づける
- 媒介中心性(betweenness centrality): 平均経路長を求めた際に，それらの経路が着目しているノードの上を通過した割合のこと．
- 次数中心性(degree cetrality): 次数の大きさだけをみる
- 近接中申請(closeness centrality): 他のノードとの距離をみる

#### 「友達の友達」は友達か
クラスター係数(clustering coefficient): 三角形の関係がネットワーク中にどれくらいあるかを計算した値．
クラスター係数が高いほど，ノードがグループとしてまとまっているといえる．

コミュニティ構造(community structure): クラスター係数が高いグループ同士が，少ない数のリンクで繋がっている構造．
このようなネットワーク上では，コミュニティの中と外でダイナミクスの伝播の仕方が異なることがある．

#### ランダムなネットワークを作る
Erdős–Rényiモデル: ランダムネットワークの生成モデル．ベースラインの比較対象として利用される．

#### スケールフリーネットワークの基本モデル
Barabási–Albertモデル: スケールフリー性をもったネットワーク．
リンクをたくさん持っているノードが新たにリンクを得やすい優先的選択(preferential attachment)というルールがある．

#### コンフィギュレーションモデル
コンフィギュレーションモデル(configuration model): 指定した次数分布を持ったランダムなネットワークを生成する．

---
### 第10章のまとめ
- 個々の要素の振る舞いから，それらが相互作用して全体としてどのような振る舞いを示すかを調べるモデルを，多体系モデル・エージェントベースモデルという．
- 個々の要素のモデルには，微分方程式モデルや確率モデル，強化学習モデルなど問題に応じて適切なものを用いることができる
- 相互作用の仕方を決めるネットワークの構造を調べることで，全体のダイナミクスについての示唆を得ることができる

---
---
## 第三部のまとめ
様々なモデルについて基礎的な内容を説明した．
何かが時間変化する様子をモデル化する，という一つの課題に対して，複数のアプローチが可能である．
そうした選択肢の中から実際にどうやってモデルを選ぶのか，それをどのように活用するのかは第四部で解説．






































