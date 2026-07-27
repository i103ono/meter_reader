# meter_reader

画像に写ったデジタル計器とアナログ計器から、表示値と単位を読み取るための実験用ツールです。OpenAI Visionによる計器設定の推定、EasyOCRによる文字認識、OpenCVによる文字盤・指針検出を組み合わせています。

> [!IMPORTANT]
> 本ツールは研究・試作段階です。読み取り値の正しさを保証するものではありません。設備の操作、安全管理、料金算定などに利用する場合は、必ず人が元画像と実機を確認してください。

## 主な機能

- 画像内の計器種別（デジタル／アナログ）、ROI、単位、目盛範囲のAI推定
- デジタル計器のAI読取またはEasyOCRによるローカル読取
- アナログ計器の円・指針検出と、指針角度から値への変換
- 複数画像の一括処理とCSV出力
- 目視で作成した正解ラベルによる評価
- グレースケール化、二値化、ROI切り出しなどの確認用ツール

## 処理方式

| 実行方法 | デジタル計器 | アナログ計器 |
|---|---|---|
| `meter_agent.py` | OpenAI Visionの推定値 | ローカル画像処理。失敗時はAI推定値へフォールバック |
| `batch_meter_agent.py` | OpenAI Visionの推定値 | AIが推定した設定とローカル画像処理 |
| `batch_read.py --config-dir` | 設定JSONに保存されたAI推定値 | 設定JSONとローカル画像処理 |
| `read_meter.py --type ...` | EasyOCR | ローカル画像処理 |

AIを使う処理では画像がOpenAI APIへ送信されます。ローカル処理だけを行いたい場合は、`read_meter.py`または`batch_read.py --type ...`を使用してください。

## 必要環境

- Python 3.11
- OpenCV、NumPy、Pillow、Matplotlib
- EasyOCR（ローカルでデジタル計器を読む場合）
- OpenAI APIキー（AI推定を使う場合）

## セットアップ

Conda環境を作成して有効化します。

```bash
conda env create -f environment.yml
conda activate meter_reader
```

AI推定を使う場合は、`.env.example`を参考にリポジトリ直下へ`.env`を作成します。

```dotenv
OPENAI_API_KEY=your_api_key_here
```

`.env`、`data/`、`configs/`、`outputs/`はGitの管理対象外です。APIキーをソースコードや設定JSONへ記載しないでください。EasyOCRは初回実行時にモデルをダウンロードする場合があります。

## クイックスタート

### 1枚をAIで判定して読み取る

入力画像には、原則として計器を1台だけ写してください。

```bash
python src/meter_agent.py data/images/meter_0001.jpg
```

計器種別、ROI、単位、目盛範囲などの設定と、最終的な読取値が表示されます。計器が0台または複数台と判定された場合は、誤読を避けるためエラーになります。

### 複数画像をAIで一括処理する

`data/images/`に画像を置き、次を実行します。

```bash
python src/batch_meter_agent.py
```

最初の数枚だけ試す場合は`--limit`を指定します。

```bash
python src/batch_meter_agent.py --limit 5
```

主な出力は次のとおりです。

- `configs/<画像名>.json`：画像ごとのAI推定設定
- `outputs/results.csv`：読取結果またはエラー
- `outputs/*_analog_debug.png`：アナログ検出結果の確認画像

APIを再実行せず、既存の設定JSONから再処理する場合は次を使用します。

```bash
python src/batch_read.py --config-dir configs
```

### APIを使わずに読み取る

デジタル計器をEasyOCRで読み取ります。

```bash
python src/read_meter.py data/images/meter_0005.jpg \
  --type digital --digits-only
```

アナログ計器では、目盛の最小値、最大値、単位を指定します。

```bash
python src/read_meter.py data/images/meter_0001.jpg \
  --type analog --min-value 0 --max-value 70 --unit MPa
```

設定JSONを使うこともできます。CLI引数と設定JSONの両方に値がある場合は、CLI引数が優先されます。

```bash
python src/read_meter.py data/images/meter_0001.jpg \
  --config configs/sample_meter_config.json
```

## 設定JSON

AI推定または手動調整した設定を`configs/*.json`へ保存できます。

```json
{
  "meter_type": "analog",
  "roi": [100, 200, 800, 800],
  "unit": "MPa",
  "min_value": 0,
  "max_value": 1.0,
  "displayed_max_value": 10,
  "scale_multiplier": 0.1,
  "min_angle": 225,
  "max_angle": 345,
  "direction": "counterclockwise"
}
```

主なフィールドは次のとおりです。

| フィールド | 内容 |
|---|---|
| `meter_type` | 必須。`digital`または`analog` |
| `roi` | 元画像上の`[x, y, width, height]`。不要な場合は`null` |
| `value` | AIが読み取った値。デジタルの設定JSON読取で使用 |
| `raw_text`、`confidence` | AIの認識文字列と信頼度 |
| `unit` | `m^3`、`m^3/h`、`MPa`などの単位 |
| `min_value`、`max_value` | アナログ計器の目盛範囲 |
| `displayed_max_value` | 文字盤に印刷された最大目盛 |
| `scale_multiplier` | `×0.1`などの倍率 |
| `min_angle`、`max_angle` | 最小値と最大値に対応する角度（度） |
| `direction` | `clockwise`または`counterclockwise` |

1枚の設定候補をAIで作成するには次を実行します。

```bash
python src/suggest_meter_config.py data/images/meter_0005.jpg \
  --output configs/meter_0005.json
```

複数画像分を作成する場合は次を実行します。失敗内容は`outputs/config_suggestion_errors.csv`へ記録されます。

```bash
python src/batch_suggest_configs.py --limit 5
```

設定したROIを画像として確認できます。

```bash
python src/apply_meter_config.py \
  data/images/meter_0001.jpg configs/meter_0001.json
```

## 別の画像データへ適用する際の注意事項

### 撮影条件

- 1画像につき計器1台を、文字盤または表示部が十分大きくなるように撮影してください。
- 正面に近い角度で撮影し、傾き、手ぶれ、ピンぼけ、逆光、ガラス面の反射を避けてください。
- アナログ計器では、外周、全目盛、指針の先端、最小・最大値、単位、倍率表示がすべて見えるようにしてください。
- デジタル計器では、小数点、先頭のゼロ、負号、桁区切り、単位、`×10`などの倍率表示を切らないでください。
- EXIFによる回転は読み込み時に補正されますが、ROIは補正後の画像座標として確認してください。

### 計器固有の確認

- AI推定値をそのまま正解とみなさず、生成された設定JSONと元画像を照合してください。
- `×0.1`、`×10`、科学表記などの倍率を必ず確認してください。倍率の見落としは桁違いの誤差になります。
- 二重目盛、非線形目盛、色分けされた複数指針、液面計、回転式数字盤などは正しく扱えない場合があります。
- アナログ計器では、指針の反対側にあるカウンターウェイト、影、目盛線、反射を指針として検出することがあります。デバッグ画像で検出線を確認してください。
- デジタル計器では、`0/O`、`1/I`、`5/S`、小数点、単位を誤認識する可能性があります。
- 設定JSONは画像の解像度と向きに依存します。同じ計器でも画像をリサイズ、回転、切り出しした場合はROIを再設定してください。

### 現在の既知の制約

- `batch_read.py --config-dir`のデジタル処理はOCRを再実行せず、設定JSON内の`value`を結果として使用します。
- バッチのアナログ処理は元画像に対して円・指針を検出し、設定JSONの`roi`を自動適用しません。必要な場合は、事前に`apply_meter_config.py`で切り出して個別に確認してください。
- バッチのアナログ処理では、現在`min_angle=225`、`max_angle=345`、`counterclockwise`が使用されます。異なる角度範囲の計器は、`read_meter.py --config`による個別確認が必要です。
- 円形でない計器、非線形目盛、複数回転する指針、複数計器を含む画像には対応していません。
- AIの出力はモデル更新や撮影条件により変動する可能性があります。再現性が必要な実験では、モデル名、設定JSON、入力画像、実行日時を保存してください。

### 情報管理と費用

- AI機能では画像を外部APIへ送信します。機密情報、個人情報、設備情報を含む画像は、所属組織の規程とAPIの利用条件を確認してから使用してください。
- 一括処理は画像ごとにAPI呼び出しを行うため、処理時間と利用料金が発生します。最初は`--limit`で少数画像を試してください。
- APIエラー、通信失敗、利用制限が発生した場合は、CSVの`status`と`error`列を確認してください。

## 評価

目視で確認した正解値を`data/evaluation_labels.csv`へ記録します。

```csv
filename,type,value,unit,min_value,max_value
meter_0001.jpg,analog,0,MPa,0,70
meter_0005.jpg,digital,40731,m^3,,
```

デジタル行には`filename`、`type`、`value`、`unit`、アナログ行には全列が必要です。値には範囲ではなく単一の数値を記入してください。

```bash
python src/evaluate.py
```

既定では`data/evaluation_labels.csv`と`outputs/results.csv`を比較し、次を表示します。

- 数値出力率
- 平均・最大絶対誤差
- 単位一致率と計器種別一致率
- アナログ計器の目盛範囲で正規化した誤差
- アナログ計器で正規化誤差が5%以内となった割合

数値出力率は「値を出力できた割合」であり、「正しい値を出力した割合」ではありません。また、単位や目盛範囲が異なる計器間の絶対誤差を単純平均すると解釈しにくいため、計器種別ごとの結果や正規化誤差も併記してください。

## 補助ツール

通常は上記の統合コマンドを使用し、問題の切り分けやパラメータ調整が必要な場合に以下を使用します。

| 目的 | コマンド例 |
|---|---|
| 画像一覧とサイズの確認 | `python src/list_images.py` |
| ファイル名の連番化（確認のみ） | `python scripts/rename_images.py` |
| ファイル名の連番化（適用） | `python scripts/rename_images.py --apply` |
| グレースケール化・二値化 | `python src/preprocess.py data/images/meter_0001.jpg --threshold 140 --resize 0.5 --debug` |
| EasyOCRの単体確認 | `python src/ocr_test.py data/images/meter_0005.jpg --digits-only` |
| ROIの数値指定による切り出し | `python src/crop_roi.py data/images/meter_0005.jpg --roi 100 200 300 120 --output outputs/roi.png` |
| ROIの対話選択 | `python src/crop_roi.py data/images/meter_0005.jpg --interactive --output outputs/roi.png` |
| OCR文字列の数値・単位変換 | `python src/parse_reading.py "l23.O kWh"` |
| 円形文字盤の検出 | `python src/detect_dial.py data/images/meter_0001.jpg --min-radius 20` |
| 指針角度の検出 | `python src/detect_needle.py data/images/meter_0001.jpg --center-x 1867 --center-y 2114 --radius 1043` |
| 指針角度から値への変換 | `python src/analog_calibration.py --angle 223.2 --min-value 0 --max-value 70 --min-angle 225 --max-angle 345 --direction counterclockwise` |

各コマンドの全オプションは`--help`で確認できます。

```bash
python src/read_meter.py --help
python src/batch_meter_agent.py --help
```

## ディレクトリ構成

```text
meter_reader/
├── configs/       # 画像ごとの設定JSON（Git管理外）
├── data/images/   # 入力画像（Git管理外）
├── outputs/       # 結果CSV、切り出し、デバッグ画像（Git管理外）
├── scripts/       # 画像名変更などの補助スクリプト
├── src/           # 読取・評価プログラム
├── .env.example   # APIキー設定例
└── environment.yml
```
