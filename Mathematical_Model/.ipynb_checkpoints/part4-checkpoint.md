# 第四部 数理モデルを作る
実際に数理モデルを利用する時に必要な，目的に応じたモデルの利用法の違い，アプローチの選定，パラメータの推定，モデルの評価について説明する．
さらに，データ取得時の注意点やモデル構築のノウハウ，数理モデルによって導かれた結論が何を意味するのかも説明し，数理モデルがしっかり力を発揮するための理解しておきたい内容を紹介する．

---
## 第11章 モデルを決めるための要素
今分析したい対象に対して，実際にどのような数理モデルを選べば良いか？
データの性質や問題に応じて，どのような点に注意してモデルを選択すべきかについて説明する．

### 11.1 数理モデルの性質
#### 数理モデルの目的
理解志向型モデリングか応用志向型モデリングかを決めて，目的を期待することが重要．

#### モデリングは試行錯誤
大きく分けて以下のステップからなる．

0. データを可視化する
1. 問題・目的を定義する
2. どのモデルを使用するか決める
3. パラメータを推定する
4. モデルの性能を評価する

これを何度も繰り返して試行錯誤し，場合によっては問題設定から見直すことも必要である．
場合によっては，数理モデルを使わなくても，ルールベースアルゴリズムで解決できることもある．

#### 決定論的モデル vs 確率モデル・統計モデル
数理モデルは，決定論的モデルと確率モデルに大別できる．
決定論的モデルは，確率の概念が入っていないモデルで，データの訂正的な振る舞いや平均的な振る舞いの理解を深めたい，ノイズが無視できる場合は有効である．
通常のデータは様々な要因によってばらついているため，確率モデルを利用すべきである．

#### 利用可能なモデルの検討
既存モデルをよく吟味した上で目標が達成できない場合，問題点を特定して新しいモデルを作成する．

---
### 11.2 理解志向型モデリングのポイント
#### 理解しやすいモデルとは
理解志向型モデリングにおける最終的な目標は，現象・データ生成ルールの理解．
以下の条件を満たすと理解しやすいモデルと言える．

- パラメータの数が少ない
- 使用している関数が簡単
- モデルの各要素(数理構造・変数・パラメータ)が直感的に理解できる
- 数理的に解析できる

パラメータの数が多かったり，関数が複雑だとたまたま整合しただけなのか，本質的に何かを捉えているのかを判断するのが困難になる．
モデルに含まれている全ての要素を言葉で説明する必要がある．
**数理モデルから得られた説明の強さは，数理モデルにおける論理が一番弱い部部と同じ強さもしくはそれ以下**である．

#### 簡単なモデルなら何でもいいわけではない
「データをどれくらい説明できるか」と「モデルの複雑さ」はトレードオフの関係．
データへの当てはまりが同程度なら，基本的に理解しやすいモデルを選択する．

#### 理解したい深さとモデリング手法
現象のメカニズムには階層があり，どのレベルでの理解が求められるかによって使用するべきモデルが異なる．
理解したい・説明したいレベルで数理的な記述をモデルに含める．

#### 数理モデルと演繹
数理モデルによって，変数たちの関係性やダイナミクスについてがよく説明できた時，まず推論できるのは，
その変数たちはそのように動いている．ということ．
さらに，そのモデルが正しいという過程のもで，演繹的に様々なことを示したり，予測することができ，
それらが論理的に正しい手順で行われれば，演繹によって得られた結果の信頼性はモデルの信頼性と一致する．

#### モデルで指定したメカニズムのレベルよりも根源的なことは説明できない
「なぜ仮定したダイナミクスが生じているのか」という，数理モデルのメカニズムが生じるメカニズムを説明することはできない．
それを言いたいなら，さらにその階層でモデリングを行う必要がある．

例えば，対象となるデータの統計分布を統計モデルで表現した場合，データがそのモデルに従っていること，何が起こるかを計算することは出来るが，
なぜその分布が出てきたのかについては何も言えない．
対象の振る舞いを直接モデリングした結果，同じ統計分布が出てくることを説明できれば，その仮定した振る舞いからその分布が出てくるメカニズムを理解することができる．

**どのレベルでデータとモデルを合わせるか**は非常に重要な視点である．

---
### 11.3 応用志向型モデリングのポイント
#### 問題を定義する
達成したい目標を数値で表現する．

例えば，「◯◯の分類問題で，入力データは△△で出力データは××のラベル，そのラベルの予測の正解率を指標とする」

しかしながら，一見もっともらしい評価指標も実際にモデルを動かすと実用上求められているものとは異なった，ということがしばしば起きる．
また運用上のコストについても考える．
継続的にデータを取得して，モデルをアップデートする必要があるのか，など．

#### 性能を重視したモデル選び
モデルの決定は性能の良し悪しで評価する．性能を評価する指標は様々あるが，「モデル選択」が重要である．

#### データの性質
どのようなデータを利用することができるかも重要である．
数理モデルは現実世界の現象を直接再現するのではなく，あくまでも与えられたデータの生成ルールを再現するので，
データが偏っていたり，誤差や欠損値が多く含まれていると，モデルの性能にそのまま反映される．
またデータの変数が不足して，手がかりとなる情報が足りないこともよくある．

---
### 第11章のまとめ
- 数理モデルを決めるために，まず目的と使用できるデータを吟味する
- 対象となるデータの性質も，使用するモデルを決める重要な要素である
- 理解志向型モデリングでは，達成したい理解のレベルに応じてモデルを決める
- 応用志向型モデリングでは，真に達成すべき目標を正しく評価指標に落とし込むことが必要

---
---
## 第12章 モデルを設計する
数理モデルを使った問題設定ができたとして，次にどの変数を含めるか，どのような要素を数理構造の中に配置するか，どこにパラメータを用意するかを問題に応じて決定する必要がある．

### 12.1 変数の選択
#### 含めるべき変数・そうでない変数
数理モデルの性能が変わらないなら，変数は少ないほどよい．

次元の呪い(the curse of dimensionality): 変数が多いほど，モデルの解釈性が下がったり，パラメータ推定のコストや過学習の危険性が増大すること．

しかし，モデルの性能向上のために必要な変数をいれなければならないので，どの変数を入れるかが非常に重要なポイントになる．

#### 変数の解釈性
理解志向型モデリングの場合，現実に何に対応するか説明できない変数は，出来るだけ排除する．
そのような変数があると，モデルを使って現象のメカニズムを説明する際に，論理的な演繹が出来なくなるから．

#### 無関係な変数は外す
対象のデータ生成規則に関係のない変数はモデルに含めない(ID番号など，リーケージ(IDに実験条件の情報が載ってしまうこと)につながる．)．
また理解志向型モデリングの場合，本質的に同じ情報を表している変数は次元削減するか，代表的なもの以外を除外する方がよい．

一方で，明らかに重要な他と独立した変数は，分析の結果として除いても影響がなかったとしても，一度はモデルに含めた方が良い場合がある．
それにより，「関係があると思ってモデルに含めたけど，結果として関係なかった」と結論づけることが可能．

#### 特徴量エンジニアリング
特徴量エンジニアリング(feature engineering): モデルの性能が良くなるように既にある変数を組み合わせて新たな変数を作り出すこと．
理解志向型モデリングの場合は，モデルの解釈性が下がったり，統計検定におけるp-hackingに繋がるため推奨されない．

#### 離散値変数・連続値変数
離散値変数を使ったモデルの特徴

- 値の表現の幅が離散であるため，表現に不正確性が生じる
- パターンの数が数えられるので状態の数が減り，扱いやすくなることがある
- 変数に関する微積分が行えないため，論理的な解析・パラメータの推定が難しくなる

連続値変数を使ったモデルの特徴

- 離散化による誤差なしで値を表現できる
- モデルのとりうる状態の数が数えられなくなり，扱いにくくなることがある
- 変数に関する微積分が行えるため，一般的に理論的な解析・パラメータ推定がしやすい

### 12.2 データの取得・実験計画
#### 着目する変数の影響をコントロールしながらデータを取得する
数理モデルの性能はデータの質に大きく左右される．

実験計画法(design of experiments): 対象について様々な要因が考えられる状況で，どのようにデータ取得をデザインするか．
ありうる条件の組み合わせの内，どれを何回どのような順番，まとめ方で実施するかを検討する．
そして分散分析(analysis of variance; ANOVA)という統計的な手法を用いて各要因が与える影響を評価する．

統計解析でない数理モデル分析を行う際にも有用である．

#### フィッシャーの三原則
フィッシャーの三原則: 着目している要因以外から生じるデータの偏りをコントロールする．

- 反復(replication): 同じ条件で観測を繰り返す．平均値としてより信頼できる値が求まり，さらに測定誤差の大きさを見積もることができる．
- 無作為化(randomization): 観測行う場所や順番などをランダムに決めること．着目している要因以外の条件を出来るだけ均一にする．
- 局所管理(local control): 無視できない要因の影響をコントロールすること．

#### フィッシャーの三原則はデータの偏りに気を付けるためのヒント
分散分析を行わない場合でも，取得したデータが結果に影響を与えるかもしれない要因がしっかりコントロールされているかどうかを検討することは重要である．
フィッシャーの三原則が満たされていない場合は，どれが満たされていないのか，それによってどういった偏りが生じうるかをチェックすることでより精度の高い分析が可能になる．

### 12.3 数理構造・パラメータの選択
#### 目的に応じた数理構造の選択 
応用志向型モデリングの場合は，基本的な分析であれば，チャートに従って機械学習モデルを選択することができる．
一方で理解志向型モデリングの場合，問題に応じてそもそもどの種類のモデルを使用するかを決める必要がある．

#### 目的変数のばらつきが無視できない場合
まず目的変数の振る舞いにおいて，確率的なばらつきが本質的か無視できるかを考える

- ばらつきが本質的な場合
数理モデルは目的変数の確率的な振る舞いを再現することを目指す．
ばらつきが大きくない場合は，値を確率的に一定の精度で予測することは可能だが，ばらつきが大きい場合は個々の値の予測は難しく，背後にある確率分布の形からメカニズムを推測する．
また，なぜその分布が出てくるのかを調べたい(メカニズムを記述して説明したい)場合は確率モデルを利用する．

#### ばらつきを考えなくて良い場合
目的変数を説明変数で表した関数を求めることが目標．
関数が生じているメカニズムを知りたい場合は，決定論的な数理構造(常微分方程式やセルオートマトン)などを用いて，変数の振る舞いを記述する．
データが行う方程式の形がすでに分かっている場合はカーブフィッティングを行う．

#### パラメータの値の範囲
変数と数理構造を決めるとパラメータが必要な場所が自然と決定する．
パラメータの値に意味がある場合，その範囲に気を配る必要がある(正負など)．

### 12.4 間違ったモデリングをしないために
#### 既存の体系との整合性・比較
新しいモデルを作ったり，既存のモデルを拡張する場合，従うべき既存の体系・法則と整合するようにする．
現実と乖離した振る舞いがモデルに含まれていると，現象を説明する論理が破綻してしまう．
既存の体型で説明できないものをモデル化する時には，提案モデルで既存体系の何が破られているのかを明確化する必要がある．

#### ハンマーしか持っていない人にはすべて釘に見える
If all you have is a hammer, everything looks like a nail.

問題を見た時に，自分が使える限られたモデリング手法の問題として，無意識的に解釈してしまう．
数多のモデリング手法の中から，最も性能が良いものを選択して使用するべきである．

そもそもモデリング手法を知らないと調べようがないので，本書でモデリング手法を大まかに示している．

#### データは適切に前処理しておく
取得したデータは適切に前処理して数理モデルに適用する．
前処理の仕方は，結論やモデルの性能に大きく影響を与える

- 外れ値(outlier)の処理: 外れ値とは，他の値とは大きく異なる値のこと．論理的に確実な方法はない．外れ値によって誤った結論が導かれることがあるので，注意する
    - 判断基準がある場合は，それを使用する．
    - 統計検定を用いて外れ値を特定する．
    - 外れ値があっても大きく影響されない分析手法を用いる
    - 外れ値を入れた分析と除いた分析の両方の結果を報告する

- 欠損値(missing value)の処理: データにおいてそもそも値が抜けてしまっている値のこと．欠損発生の意味をまず調べる．
    - データからその点を取り除く
    - 適当な値(平均値，中央値)でうめる．あまり推奨されない？

---
### 第12章のまとめ
- 理解志向型モデリングでは，必要な変数を吟味して使用する
- 応用志向型モデリングでは，少しでも使える情報は使う
- 理解志向型モデリングの数理構造は説明したいデータのばらつきが本質的か・無視できるか，また説明メカニズムのレベルで選ぶ
- モデルと現実・既存体系との整合性を確保しつつ，一番適切なアプローチを選択する
- 外れ値や欠損値，その他のデータの質が数理モデリングの質を決める

---
---
## 第13章 パラメータを推定する
数理モデルが出来たら，データによく当てはまるようにパラメータを推定する．
パラメータの決め方に，モデルや問題設定によって様々な方法がある

### 13.1 目的に応じたパラメータ推定
#### 動かせるパラメータと動かせないパラメータ
パラメータの値が実験などですでに求まっている場合に，違う値を入れても現実と乖離したモデルになる(あえて違う値をいれることで本質を知ることができるが)．
ボトムアップ的な理解志向型モデリングの場合は，全てのパラメータが大体わかっていて自動的に決められるのが理想．

#### パラメータの点推定
点推定(point estimation): パラメータの値の組を一つに決めること．

#### 変数の振る舞いを定量的にデータと合わせたい場合
目的関数(objective function): モデルから生成される値と実際のデータとの差(誤差)を計算するための指標．
平均二乗誤差(mean squared error; MSE)や対数尤度(log likelihood)などさまざまある．

#### 単に誤差の大きさを平均する
平均二乗誤差や外れ値から受ける影響を弱めた平均絶対誤差(mean absolute error; MAE)などがある．
また，誤差が小さいデータには二乗誤差，大きいデータには絶対誤差を計算して和をとるHuber損失関数(Huber loss function)や，
値が一定以内に収まっていれば誤差を0とすることで過学習を防ぐ$\epsilon-$許容損失関数($\epsilon-$insensitive loss function)などがある．

#### 対数尤度
モデルが確率的な要素を含んでいて，あるデータが得られる確率を直接記述できる場合，対数尤度によってモデルの当てはまり具合を評価できる．
尤度とは，仮定されたモデルに，全ての観測値がそのモデルから出現する確率を表していて，尤度が大きいほどモデルがよくデータを表現していることになる．
尤度を数式で書くと
$$
L = p(X|\theta)
$$
であり，$X$はすべての観測地，$\theta$はパラメータである．
実際には，計算しやすくするために対数をとって行う．

このように，尤度を目的関数としてパラメータを推定する方法を最尤推定(maximum likelihood estimation; MLE)という．

しかし，尤度を最大化するパラメータが真のパラメータの良い近似を与えることは自明ではなく，漸近正規性が担保されないモデルでは理論的な保証が失われる．

#### 確率分布間の「差」を最小化する指標
カルバック・ライブラー情報量(Kullback-Leibler divergence): 分布間の差を定量化する指標．

#### 交差エントロピー
交差エントロピー(cross entropy): 情報量の観点から二つの分布の近さを定量化する

---
### 13.2 パラメータ推定における目的関数の最小化
#### 目的関数を最小化するには
適切に設定した目的関数を最小化するパラメータをどうやって求めるのか

#### 解析的に解く
データとモデルが与えられた時の目的関数$L$は，パラメータ$\theta$の関数になるので，
$$
\dfrac{\partial L}{\partial \theta} = 0
$$
となるパラメータ$\theta$を求めれば良い．

#### パラメータをスウィープする
解析的に求められない場合に，パラメータに数値を入れて目的関数を計算する方法がある．

パラメータスウィープ(parameter sweep): パラメータの値を試す方法．
調整するパラメータの数が多ければ使えないが，正しく大域最適解を探せる可能性がある．
またある程度，最適解の目星をつけた後に，二分法や最急降下法でパラメータを求めることもある．

#### 最急降下法
最急降下法(gradient-descent method): 目的関数が具体的に数式で計算でき，そのパラメータによる微分が計算できる場合に用いる．
$$
\theta \leftarrow \theta - \alpha \dfrac{\partial L}{\partial \theta}
$$
を使って，パラメータの値を更新し，変化しなくなるまで行う．

#### 局所解陥らないために
初期値をランダムに設定したり，確率的勾配降下法(stochastic gradient descent)などを用いる．

確率的勾配効果法は，最急降下法でパラメータを更新する際に，データを全て使わずに毎回一部のデータをランダムに用いる方法．
局所解から抜け出しやすくなる他，計算コストを軽減できることもある．

一方で，大域最適解を求めることはほぼ不可能なので，実用上は性能が良い局所解で良い場面もある．

#### 過学習を防ぐ
過学習を防ぐために，データをほどほどに信用して合せすぎないようにする方法がある．

正則化(regularization): 目的関数にパラメータの「値の大きさ: ノルム」である$||\theta||$を足した$L(\theta) + \lambda ||\theta||$を最小化する．
このノルムの定義にはいくつか方法がある．

- L2ノルム正則化: モデルに含まれる全てのパラメータの値を二乗して足し算する．$||\theta|| = \Sigma_{i} {\theta_{i}}^2$
- L1ノルム正則化: パラメータの値の絶対値を足し合わせたもの．$||\theta|| = \Sigma_{i} |\theta_{i}|$

L1ノルム正則化の方が，値が小さいパラメータに対する罰則が強いので，値が0になるパラメータの数が増える．
つまり，パラメータの数を減らしてモデルの推定を行うことになる．
これをスパースモデル(sparse model)という．

正則化がうまくいくかは，データによるので試行錯誤が必要である．

#### 目的関数最小化の実施
モデルに応じた計算ライブラリがあり，簡単に目的関数の最適化によるパラメータ推定が可能である．
モデルが複雑な場合は，MCMCによる手法を行う．

微分方程式モデルや，複雑は確率モデル，多体系・エージェントベースモデルにおいては，このようなライブラリが存在しないが，
これらは理解志向型モデリングとして採用されることが多く，目的関数最小化によるパラメータ推定を行うニーズがほとんどない．
(理解志向型モデリングでは，パラメータは最初から自動的に決まることが理想だから)

最後に，
**定量的に十分な予測力をもたないモデルにおいて，パラメータの値を細かくきっちり決める行為には意味がない．**


---
### 13.3 ベイズ推定・ベイズモデリング
#### パラメータの分布を考えるのがベイズ推定
ベイズ推定(Bayesian inference): データからパラメータの確率分布を推定する方法の一つ．

#### パラメータの確率分布？
事前分布$\phi (\theta)$，パラメータ$\theta$が与えられた時の$X$の確率密度関数を$p(X|\theta)$とする．
データ$X^n$が与えられた時のパラメータの条件付き密度関数は，
$$
p(\theta|X^n) = (1/Z) \phi(\theta)p(X^n|\theta)
$$
で表され，この$p(\theta|X^n)$を事後分布(posterior distribution)という．
ただし$Z$は正規化定数であり，詳細は省く．
この事後分布から以下のように予測分布$p(x|X^n)$を計算することをベイズ推定という．
$$
p(x|X^n) = E_{p(\theta|X^n)}[p(x|\theta)] = \int{p(x|\theta)p(\theta|X^n)}d\theta
$$

#### 推定された分布を特徴づける
パラメータの値を一つに決めたい時は，事後確率が最大になるMAP estimatorや，事後分布による期待値EAP estimator，中央値(MED estimator)などを計算して点推定値とする．
また分布の標準偏差(事後標準偏差)を求めれば，パラメータがどれくらいばらついているかを特徴づけられる．

#### マルコフ連鎖モンテカルロ法
事後分布を求める際に，解析的に計算を行うことは困難なことが多いので，マルコフ連鎖モンテカルロ法(Markov chain Monte Carlo; MCMC)を用いることが多い．
簡単な原理としては

数値的に求めたい確率分布を$q(\theta)$とすると，まず確率変数の従う確率モデルを考える．
このモデルをシミュレートして動かすと，最終的に$(\theta_1, \theta_2, ..., \theta_t)$が得られ，この出現確率の分布が求めたい確率分布$q(\tehta)$と一致することができる

#### メトロポリス法
MCMCにおいて，具体的に確率過程を求める方法の一つがメトロポリス法．

1. $\theta$の初期値をランダムに決める
2. 現時点での値$\theta_t$からランダムに値を変化させた値$\theta_t'$を用意する
3. 関数の値の比$q(\theta_t')/q(\theta_t)$を計算する
4. この値が1より大きければ$\theta_t'$を採用し，小さい場合は，確率$q(\theta_t')/q(\theta_t)$で$\theta_t'$を採用し，残りの確率で元の値を維持する．
5. 2-4を繰り返す

十分長くシミュレーションすれば，初期条件によらない定常分布が得られる．

---
### 第13章のまとめ
- モデルを定量的にデータと細かく一致させたい場合には，平均二乗誤差や対数尤度などを目的関数として，それを最小化するパラメータの値を求める
- 目的関数の最小化には，幅広い問題に使える最急降下法などの手法を用いる
- パラメータを1つの値ではなく分布と考えるベイズ推定の考え方も非常に有用で，推定されるパラメータ分布の様々な情報を利用することができる

---
---
## 第14章 モデルを評価する
数理モデルを作成する際に，試行錯誤をするが，モデルを選択するための指標が必要になる．

### 14.1 「良いモデル」とは
#### 目的に応じたモデルの評価
データによく当てはまっていればそれだけで良いわけではない．
モデルの良さを評価するための考え方・指標も目的に応じて変わる．

#### メカニズム理解を目的としたモデルの評価
- モデルの解釈性: モデルの各要素(変数・数理構造・パラメータ)がすべて説明可能で，既存体系と矛盾していないか．まt対象のデータ生成規則を再現する最小限の構成になっているか
- 当てはまりの良さ: モデルがデータと許容される範囲で当てはまり，整合的か．

#### 統計的推論を行うためのモデルの評価
解釈性とデータへの当てはまりのバランスが大事だが，よりデータへの当てはまりの重要度が高まる．
モデルの複雑さとデータへの当てはまりの良さのバランスを数値的に評価する方法論がある．

#### 応用志向型モデリングにおけるモデルの評価
解釈性はそこまで重視されず，未知のデータへの当てはまり(予測性能)が良いかが重要になる．

---
### 14.2 分類精度の指標
#### 当てはまりの良さ・性能を測る
適合度(goodness of fit): 当てはまりの良さのこと．目的関数や決定係数などもそうである．

#### 正解率・再現率・特異度・適合率・F値
混同行列(confusion matrix)

|  | 実際は陰性 | 実際は陽性 |
| --------  | ---------- | -------- |
| 陰性と予測 | 真陰性(TN)| 偽陰性(FN) |
| 陽性と予測 | 偽陽性(FP) | 真陽性(TP) |

- 正解率(accuracy): 実際に予測された値が正解する割合．$\dfrac{TN+TP}{TN+FN+FP+TP}$
- 再現率(recall): 陽性をの人を正しく陽性だと当てた割合．$\dfrac{TP}{FN+TP}$
- 特異度(specificity): 陰性の人を正しく陰性と当てた割合．$\dfrac{TN}{TN+FP}$
- 精度(precision): 陽性と予測した人の中で実際に陽性であった割合．$\dfrac{TP}{(FP+TP)}$
- F値(F score): 精度と再現率の調和平均．

目的に応じて，それぞれの指標を使い分ける．

#### ROC曲線とAUC
実際のモデルでは，どのクラスに属するかを0-1の間の数値で予測する．この時の基準を閾値(threshold)といい，
理想的なモデルではこの閾値を正しくとれば全てに対して正しく分類できるし，反対に，性能の低いモデルでは閾値をどんな帯にしても一定割合で誤った分類をしてしまう．

ROC曲線(receiver operating characteristic curve): 閾値を変化させることによって，モデルが予測する分類スコアがどれくらい2つのクラスで分離しているかを評価する指標．
縦軸に再現率，横軸に偽陽性率(=1-特異度)をとり，プロットしていく．
そのグラフの下部面積をAUC(area under the curve)といい，モデルの評価指標として利用できる．
1に近づくほど良いモデルである．

---
### 14.3 情報量基準
#### モデルが複雑ならば適合度は上がる
モデルを複雑にすれば，過学習の危険性も高まる．
よって，モデルを適切な複雑さにとどめつつ，その中で出来るだけよくデータを説明するモデルを選ぶ必要がある．

情報量基準(information criterion)という指標が一般的に用いられる．
情報量基準は同じ規則で生成された未知のデータをどれだけ説明できるかというアイデアに基づいている．

#### 赤池情報量基準AIC
赤池情報量基準(AIC)
$$
AIC = -2\ln{L} + 2k
$$
ただし，$L$はモデルの最大尤度(パラメータ推定して尤度を最大化したもの)，$k$はモデルに含まれている自由に動かせるパラメータの数．

#### ベイズ情報量基準BIC
$$
BIC = -2\ln{L} + 2k\ln{n}
$$
ただし，$L$はモデルの最大尤度(パラメータ推定して尤度を最大化したもの)，$k$はモデルに含まれている自由に動かせるパラメータの数，$n$はデータの観測数．

#### その他の情報量基準
- 最小記述長(MDL)
- 逸脱度情報量基準(DIC)
- WAIC, WBIC

一般的に，このような指標に基づいたモデル選択でも，「真のモデルは分からない状態で限られたデータからモデルを評価する必要がある」という困難から逃れられない．
作ったモデルが未知のデータに対してよく機能することを実際に確かめることができれば，それが一番確実である．
テストデータを用意できない場合には，情報量基準が役に立つ．

### 14.4 ヌルモデルとの比較・尤度比検定
#### モデルに入れた要素に意味があるか
あるモデルが別のモデルに含まれている(ネストされている; nested)場合に，これらを比較することを考える．
この分析はより適合度の良いモデルを探したり，対象となるシステムにおいてある要素が重要であるかどうかを検証できる．
二つのモデル，提案したいモデルとヌルモデル(null model; 主張に必要な要素を除いたモデル)を比較する．

提案モデルの方がデータをよく説明することを示せば，この要素は重要なファクターであるという主張がいえる．

#### 尤度比検定
尤度比検定(likelihood): データがヌルモデルから生成されていると仮定し，ランダムにデータを生成して，ヌルモデルと提案モデルの適合度の差をみる．
この手続きを何度も行い，適合度の差の分布を作り，その分布の中で実際のデータに当てはめた際に得られる適合度の差が発生する確率を計算する．
それが事前に決めた有意水準よりも小さければ，有意に「良いモデル」であると主張できる．

#### 統計検定とヌルモデル
モデル間の比較の文脈でヌルモデルの利用することは，統計検定の文脈で自然と行われている．
データからある量を計算したところ，ヌルモデルでは説明できない異常な値が出ているという主張を行う時，このヌルモデルに対応するモデルには何を使っても良い．

### 14.5 交差検証
#### 実際にモデルの性能を未知データで試す
推定されたモデルの未知のデータに対する説明能力を計測するために，データをモデル推定に用いる訓練データと性能をテストするテストデータに分ける．
単純に訓練データとテストデータの2つに分けて性能評価を行うことをホールドアウト検証(hold-out validation)という．
しかし，訓練データが減ってしまうデメリットがある．
そこで，交差検証(cross-validation)という方法がよく使用される．

K-分割交差検証(K-fold cross-validation): 全体のデータをK個のブロックに分割し，K-1個で学習し，残り1個でテストする．
これをK通り全てで行い，性能の平均値を取って最終的なモデルの性能とする．

leave-one-out交差検証(LOOCV): さらに分割の個数を最大限まで増やし，テスト用のデータを1サンプルだけ残して交差検証する方法．

#### リーケージには気を付ける
リーケージ(leakage): 訓練データにテストデータの何らかの情報が残っている場合(本来は見えないテストデータの情報が訓練時に見えている)，モデルの性能が不当に高くなること．
時系列データでは特に気をつける．
データに対する前処理を全体に行う場合も情報がリークすることがあるので，分割した後に行う．
基本的な考え方として，今手元に持っていないデータを将来取得することができたとして，それに対する性能評価を後で行う時に踏む手続きと同じになっているかをチェックする．

#### モデルの信憑性と未知のデータ
数理モデルの推定では，あくまでも仮定したモデルの中で訓練データに一番近いものを選んでいるだけ．
あるデータを説明することができるモデルが，さらに別の新しいデータをたまたまよく説明できることは考えにくい，という考えの元交差検証などが行われている．


---
### 第14章のまとめ
- 「いいモデル」とは，目的を達成するのに役立つモデルのことである
- 目的に応じて，評価する観点・指標が異なる
- テスト用のデータが準備できない時には，情報量基準が便利
- テスト用のデータを準備できる時は，ホールドアウト・交差検証を行う

---
---
## 第四部のまとめ
実際に数理モデルを構築するための各ステップについて，考えるべきポイント，具体的な方法論について解説した．
一般論としての内容を個別の文脈と照らし合わせて反芻するとより深い理解につながる．

