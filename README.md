# 🐀 MABe-Challenge---Social-Action-Recognition-in-Mice
このリポジトリは **MABe-Challenge---Social-Action-Recognition-in-Mice コンペティション** のための開発環境です。  
Neural Network モデルの学習、特徴量設計、前処理、可視化、実験管理までを一貫して実行できる構成になっています。

---

# 📁 ディレクトリ構成

```text
├── README.md
├── main.py
├── configs/                 # Hydra 設定ファイル
├── apps/                    # 可視化・本番推論などの補助ツール
│   └── play_movement.py
├── data/
│   ├── raw/                 # 生データ（Kaggle input）
│   ├── interim/             # 中間生成物（EDA/デバッグ）
│   └── processed/           # 学習用前処理済みデータ
├── experiments/             # 実験結果（ログ/モデル/OOF など）
│   ├── 102_1dcnngru_exp296/
│   ├── 102_1dcnngru_exp297/
│   └── 102_1dcnngru_exp298/
├── notebooks/               # EDA/可視化 Notebook
├── src/
│   ├── data/                # Dataset, Loader, 前処理ロジック
│   ├── models/              # NNモデル（CNN/GRU/Transformer/Mamba など）
│   ├── scripts/             # 学習・推論スクリプト
│   ├── utils/               # 共通ユーティリティ関数
│   └── visualization/       # プロットや可視化系関数
├── reports/                 # 実験レポート・分析ログ
├── score_progress.md        # CV/LB 推移ログ
└── useful_kaggle_links.md   # 参考リンク集
```
---

# 🔍 各ディレクトリの詳細

## ◆ `data/`
  コンペ提供データと前処理後のデータを整理したディレクトリ。

- `raw/`  
  Kaggle から取得した **生データ**。

- `interim/`  
  中間前処理結果。EDA・デバッグ用。

- `processed/`  
  学習用に最終整形したデータ。

---

## ◆ `notebooks/`
EDA・可視化・実験検証のための Notebook。

**番号ルール（旧テンプレ継承）**  
- `0xx`: データ前処理  
- `1xx`: シンプル ML（決定木/線形モデル）  
- `2xx`: Neural Network（CNN/GRU）  
- `3xx`: NN + Transformer/Mamba など  
- `9xx`: アンサンブル  

---

## ◆ `src/scripts/`
学習・推論スクリプトを配置。

- `0xx/`: データ前処理  
- `1xx/`: 伝統的 ML  
- `2xx/`: CNN + GRU 系  
- `3xx/`: Transformer / Mamba 系  
- `9xx/`: アンサンブル  

### `configs/`  
Hydra の設定ファイル。  
スクリプトごとに `configs/<script_name>/<yaml>` を配置。

---

## ◆ `experiments/`
- 各実験のログやモデルを保存する。

**フォルダ形式**  
- experiments/<script番号>_<モデル名>_exp<連番>/
    例:experiments/102_1dcnngru_exp296/
---

## ◆ `apps/`
- 小規模アプリケーションや可視化ツール。
    `play_movement.py` → プレイ単位の選手軌跡可視化 など

---

## ◆ `src/models/`
- 本プロジェクトで使用するモジュールを整理。

- CNN, GRU  
- Transformer  
- Mamba  
- Feature Fusion  
- Attention Pooling  
など

---

# 🚀 実行方法

## ■ Hydra を使った実行
```sh
python main.py experiment=102_1dcnngru_exp296
または、個別スクリプトを直接実行：
python src/scripts/102_1dcnngru/train.py
```
# ⚙️ 設定ファイル（Hydra）

メイン設定ファイル：
configs/config.yaml

defaults を切り替えることで実験の制御が可能：
defaults:
  - 102_1dcnngru: exp296

実行時に override する例：
python main.py 102_1dcnngru=exp297 trainer.epochs=50
Hydra のドキュメント:
https://hydra.cc/docs/intro/


# 🆕 新規スクリプト作成

- テンプレート生成ツール：
```sh
python generate_template.py --name 103_newmodel
```
- 生成されるもの：
	•	script本体
	•	config yaml
	•	実験フォルダ構成

⸻

# 🛠 開発ルール

■ Docstring

- Google Style に準拠
https://google.github.io/styleguide/pyguide.html

■ Type Hint
	•	from __future__ import annotations
	•	List, Dict は使用せず list, dict を使用

■ パス
- OS 非依存にするため pathlib.Path を推奨。

■ コードフォーマット
- black￼ を使用。
```sh
pip install black
black src/
```

# 📈 実験管理（experiments/）

- 各フォルダには以下を保存：
	•	Hydra 設定（.hydra/）
	•	学習済モデル（model.pth）
	•	OOF prediction
	•	CV metrics
	•	ログ（local / wandb）
	•	推論結果


⸻

# 🔗 参考リンク

- useful_kaggle_links.md
    → トップ解法、論文、Github、Notebook などを整理。

⸻

# 🗂 Appendix: Score Progress

- score_progress.md
    → CV / LB の時系列推移、実験改善メモなどを管理。