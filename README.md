# lecture-3A-E5

## Branches

-   feature/single
    -   シングルエージェントでの学習
-   feature/multi
    -   マルチエージェントでの学習

## Directory

-   run.py
    -   シミュレーションの実行と結果の表示
    -   feature/single と feature/multi で関数などの実行が異なる
-   environment.py
    -   シミュレーション環境の定義
    -   feature/multi では，複数のエージェントを内包するクラスを定義
-   dqn.py
    -   DQN の定義

## Run

```bash
$ pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116
$ python run.py [rewards]
```

ただし，`[rewards]` は以下のいずれか（複数可）を指定する．

-   `0` : 停止中の車両数の減少に伴う報酬
-   `1` : 信号機の頻繁な切り替わりに伴う報酬
-   `2` : 信号における「圧力」に伴う報酬

