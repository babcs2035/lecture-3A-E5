# lecture-3A-E5

## directory

-   run.py
    -   シミュレーションの実行と結果の表示
-   environment.py
    -   シミュレーション環境の定義
-   dqn.py
    -   DQN の定義

## Run

```bash
$ pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116
$ srun -p v -t 60:00 --gres=gpu:4 --pty python run.py # 60 分間，GPU 2 基を確保
```
