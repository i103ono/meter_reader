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

Condaまたはuvのどちらかを使用できます。同じ作業ディレクトリで両方の環境を同時に有効化せず、利用する環境を一つに決めてください。

### uvを使う場合

uvが未導入の場合は、[uv公式のインストール手順](https://docs.astral.sh/uv/getting-started/installation/)に従ってインストールします。macOSではHomebrewも利用できます。

```bash
brew install uv
```

リポジトリ直下で依存関係を同期します。`.python-version`に従ってPython 3.11と`.venv`が用意されます。

```bash
uv sync
```

コマンドは`uv run`経由で実行できます。仮想環境を手動で有効化する必要はありません。

```bash
uv run python src/meter_agent.py data/images/meter_0001.jpg
uv run python src/evaluate.py
```

または、従来どおり`.venv`を有効化して`python`を実行します。

```bash
source .venv/bin/activate
python src/meter_agent.py data/images/meter_0001.jpg
```

`pyproject.toml`を変更した場合は、再度`uv sync`を実行してください。`uv.lock`がある場合は、CIなどで`uv sync --locked`を使用すると、ロックファイルを更新せずに環境を再現できます。

### Condaを使う場合

Conda環境を作成して有効化します。

```bash
conda env create -f environment.yml
conda activate meter_reader
```

### APIキーの設定

AI推定を使う場合は、`.env.example`を参考にリポジトリ直下へ`.env`を作成します。

```dotenv
OPENAI_API_KEY=your_api_key_here
```

`.env`、`data/`、`configs/`、`outputs/`はGitの管理対象外です。APIキーをソースコードや設定JSONへ記載しないでください。EasyOCRは初回実行時にモデルをダウンロードする場合があります。

## クイックスタート

### 仕様v0.2・段階1：APIなしで画像を記録する

段階1用CLIは `src/cli.py` です。JPEG/PNGの検証、EXIF向き補正、ROI切り出し、
観測ID・実行ID・日時の記録、結果JSONの保存を行います。
**MockReaderは画素から値を推論せず、値・単位・目盛情報を `null` にします。**
`mode=mock`、`status=needs_review`、`auto_accept=false` を保存し、精度や測定結果には使用しません。
APIキー、外部通信、OCRモデルのダウンロード・実行は不要です。

リポジトリ直下で実行します。Python 3.11とPydanticを含む依存関係はロックファイルから導入します。

```bash
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
uv run --locked python src/cli.py --help
```

書き込み先が制限されたクラウド環境では、先に次を設定してください。

```bash
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_PYTHON_INSTALL_DIR=/workspace/.local/share/uv/python
export XDG_CACHE_HOME=/workspace/.cache
```

実画像を持っていない場合も、次のコマンドをそのまま実行できます。
人工画像を一意のディレクトリへ作成し、APIなしで結果JSONまで保存します。

```bash
uv run --locked python - <<'PY'
from pathlib import Path
import subprocess
import sys
import tempfile
from PIL import Image
Path("outputs").mkdir(exist_ok=True)
folder = Path(tempfile.mkdtemp(prefix="stage1-demo-", dir="outputs"))
with Image.new("RGB", (120, 80), "white") as image:
    image.save(folder / "synthetic.png")
subprocess.run([sys.executable, "src/cli.py", str(folder / "synthetic.png"),
                "--question", "動作確認専用。値を読んだ結果ではありません"], check=True)
PY
```

自分の画像を指定する場合：

```bash
uv run --locked python src/cli.py data/images/example.jpg \
  --roi 100 200 900 1000 --target-type analog \
  --meter-id pressure-A --observation-id obs-A-001 \
  --captured-at 2026-10-27T10:00:00+09:00 \
  --question "MPa側の目盛と指針を対象にする"
```

新CLIの `--roi` は **EXIF向き補正後の元画像上の `[x1,y1,x2,y2]`** で、
右端・下端は含みません。範囲外、空領域、不正型はエラーとなり、補正しません。
既存の `crop_roi.py --roi` や設定JSONは引き続き `[x,y,width,height]` です。
旧形式から接続する場合は新CLIの `--legacy-roi X Y W H`、
または `image_io.xywh_to_xyxy()` / `xyxy_to_xywh()` を明示的に使ってください。
`--roi` と `--legacy-roi` の併用はできません。

ROI省略時は補正後の画像全体を保存します。日時省略時は、タイムゾーンが分かる
EXIFのDateTimeOriginalだけを使用し、それ以外は `captured_at=null` にします。
タイムゾーンなしEXIFを解釈する場合は `--exif-timezone Asia/Tokyo` のように明示してください。
DSTで曖昧／存在しない日時は不明として残します。手入力の撮影日時を優先し、
タイムゾーンなし手入力は拒否します。保存する撮影・処理日時はUTCで、
撮影日時の元タイムゾーン、EXIF文字列、仮定したタイムゾーンと根拠を別に記録します。
処理時刻を撮影時刻へ代入しません。

結果は既定で `outputs/runs/run_<UUID>/` に保存し、標準出力へ `result.json` のパスを返します。
`--output-dir` で保存ルートを変更できます。各実行には、元バイト列の `source.jpg/png`、
向き補正済み `normalized.png`、未補正ROIの `roi.png`、`result.json` が残ります。
元画像パス・SHA-256・向き・補正前後サイズ・ROI・質問文・実行モードを結果へ記録します。
同じ画像を再読取する場合は `--observation-id` を維持してください。
`run_id` と保存先は毎回別になり、既存ファイルがあれば上書きせず失敗します。
観測ID省略時には新しい観測IDを発行します。単発CLIの `meter_id` は省略可能で、
将来の履歴用には `MeterObservationInput` 型で必須にできます。

### 開発用20枚の入力と正解の分離

MeasureBench real_worldの選定済み20枚は、このクラウド環境の
`data/measurebench/images/` に配置済みです。全20枚のSHA-256・画像サイズが添付の正解メタデータと一致し、
MockReaderで全件の入力・JSON保存と再実行を確認しました。数値の精度評価は行っていません。
添付の `inputs.jsonl`、`labels.jsonl`、`labels.csv`、`provenance.json` は原本のまま保持しています。
原本の入力は `filename/question_id/question` 形式のため、
入力とprovenanceだけからCLI用の `inputs.stage1.jsonl` を別途作成しました。
正解の数値・単位・evaluatorはこの変換やReader入力へ含めません。

別のセットを用意する場合は、次の入力専用JSONL契約を使用してください。
JSONLの画像相対パスは **JSONLファイルのあるディレクトリ基準** で解決します。
各行は次の形式です（パス・ID・質問・revision等は実際のメタデータを使用し、不明なら省略／null）。

```json
{"image_path":"images/example.jpg","observation_id":"obs-example","roi":null,"target_type":"auto","question":"元のデータセットの質問文","dataset":{"name":"MeasureBench","split":"real_world","revision":null,"seed":null,"question_id":null,"category":"pressure_gauge","source":null}}
```

```bash
uv run --locked python src/cli.py --inputs data/measurebench/inputs.stage1.jsonl
```

各行から別の結果を保存し、標準エラーへ `inputs: 20, saved: 20, errors: 0` のような
件数を表示します。一部画像が不正でも残りを確認し、エラーがあれば終了コードは1です。
不正JSON・不明フィールドは行番号付きで拒否します。JSONLとCLIの個別設定は混在させません。

| 種類 | 配置済み枚数 | 段階1と後続段階で確認すること |
|---|---:|---|
| 圧力計 | 10 | 入力・ROI・質問を保存。二重目盛は対象の単位・目盛を特定し、曖昧なら要確認 |
| 電力量計 | 6 | 対象表示・桁・小数点・単位の質問を保持。カテゴリだけで計器形式を決めない |
| 流量計 | 1 | 積算値と瞬時流量など、どの表示を読むか保持 |
| パルスオキシメーター | 1 | SpO2と脈拍等の複数表示から対象を質問で指定 |
| 電子ノギス | 2 | 表示・符号・mm/inch等の対象を保持し、単位を推測しない |

カテゴリや質問から値・単位を作りません。Readerへ渡す型は
`image_path`（保存したROI）、`target_type`、`question` のみです。
`labels.jsonl` / `labels.csv`、元の `evaluator` / `evaluator_kwargs`、
区間下限・上限、単位候補・出典は評価用ファイル側に元のまま保持してください。
CLIはそれらを開かず、`labels` や `evaluator` を入力行へ混ぜると拒否します。
区間正解を中央値へ変換したり、画像から公式正解を作ったりしません。
`dataset.source` は入力の出典参照で、正解データ本体を含めません。
20枚は開発・入力確認用であり、企業画像による最終性能評価や数値正解率とは分けます。

詳細な既存機能との差分、検証範囲、段階2の前提は
[段階1の実装記録](docs/task01_stage1.md) を参照してください。

### 仕様v0.2・段階2：OpenAI接続と未補正ROIの読取

段階2の接続実装を追加しました。今回の検証範囲はユーザー指定により
**実装と通信モックまで** です。実API、VLMによる実画像の読取、精度評価は未実施です。
既定の `--mode mock` は引き続きAPIを呼びません。

希望モデルの「GPT-6 Luna」は、公式OpenAI SDKの
[モデル型定義](https://github.com/openai/openai-python/blob/main/src/openai/types/shared/chat_model.py)
で `gpt-6-luna` を確認し、
非秘密の設定例を [examples/openai_gpt6_luna.json](examples/openai_gpt6_luna.json) に用意しました。
この一覧はモデルIDの根拠であり、画像入力・構造化出力の実動作を検証したものではありません。
別モデルは `--model <APIモデルID>` または設定ファイルの `model` で指定します。
モデル指定を省略した推論は、設定不備として拒否し、別モデルへ自動で切り替えません。

次はAPIキー不要・外部AI通信なしで実行できる確認コマンドです。

```bash
uv run --locked --offline python -m unittest discover -s tests -v
uv run --locked --offline python src/cli.py --help
uv run --locked --offline python src/cli.py \
  --inputs data/measurebench/inputs.stage1.jsonl --mode mock
```

段階2までの自動テスト54件（段階1の28件＋段階2の26件）では、架空のHTTP応答で
アナログ・デジタルの値・単位・不明状態、構造化出力、モデル設定の切替、
認証／通信エラー、再試行、予算、記録、CLIの終了コードを確認します。
テストの架空値・トークン数は測定結果やプロバイダの実使用量ではありません。
実画像20枚のHTTP模擬接続確認も実施し、20保存・0技術エラー、外部ソケット呼出し0件、
画像と正解ファイルの不変性を確認しました。模擬応答は全件「値・単位不明」です。

実APIの利用は次回の範囲です。利用時には、モデルの画像・JSON Schema対応を確認し、
HTTPSの接続先 `api.openai.com` と認証・料金・費用上限を安全な環境設定で用意してください。
設定ファイルにはキーの値を保存せず、環境変数名 `METER_READER_API_KEY` だけを指定します。
ローカルで既存の `OPENAI_API_KEY` を使用する場合は、
`--api-key-env OPENAI_API_KEY` と明示できます。新CLIは.envを自動読込しません。

認証・費用設定を終えた後の実行形式（今回の検証では実行していません）：

```bash
uv run --locked python src/cli.py \
  --inputs data/measurebench/inputs.stage1.jsonl \
  --mode inference --provider-config examples/openai_gpt6_luna.json --limit 2
```

設定例の料金と米ドル上限は `null` です。金額上限を使用するには、
`pricing` に `model`、現在の `input_usd_per_million` / `output_usd_per_million`、
モデルの全コンテキストに基づく入力トークン上限 `max_input_tokens_per_call`、
確認元 `source` を設定し、`max_cost_usd` を指定します。
各送信前に入力上限＋最大出力分を予約し、使用量が分かった場合だけ精算します。
バッチ全体で同じ上限を共有し、使用量不明の失敗・再試行は予約を保持します。
これは設定された料金・上限に基づく見積りであり、未確認の料金や請求額を推測しません。
料金未設定時の `estimated_cost_usd` は `null` で、金額上限は有効になりません。

実行結果には `stage=2`、`mode=inference`、モデルID、プロンプト版、
画像入力設定、呼出し回数・使用量・所要時間、元の出力文字列を記録します。
APIキー、Authorizationヘッダー、HTTPエラー本文は保存しません。
タイムアウトは最大30秒、Readerの期限は最大120秒、送信は1回＋一時障害の再試行1回までです。
期限前後で残り時間を確認し、通信待機にはその残り時間以下のタイムアウトを渡します。
構造不正・拒否・出力途切れは自動で修復再送しません。

値・単位とも得られなければ `unreadable`、一部不明や候補値があれば `needs_review`、
API障害や構造不正なら `error` とします。値・単位が読めても未校正の自動採用は行いません。
エラーJSONを保存できた場合もCLIは終了コード1を返し、保存件数とエラー件数を別々に表示します。
詳しくは [段階2の実装記録](docs/task02_stage2.md) を参照してください。

### 仕様v0.2・段階3：固定補正の生成と再生成

`--recipes` で最大2種類の固定レシピを選びます。全候補を同じ未補正ROIから
独立して生成し、PNGと処理条件・ハッシュ・画質指標・個別の読取結果を保存します。
候補の自動選択、Planner、精度による自動採用は行いません。
指定しなければ段階1・2の従来動作のままです。

| レシピID | 処理順 |
|---|---|
| resize_2x | Bicubicで縦横2倍 |
| gamma_mild | 輝度Yのガンマ補正、指数0.75 |
| clahe_mild | 輝度YのCLAHE |
| gamma_clahe | ガンマ → CLAHE |
| resize_gamma | ガンマ → 縦横2倍 |
| resize_clahe | CLAHE → 縦横2倍 |
| resize_gamma_clahe | ガンマ → CLAHE → 縦横2倍 |

ガンマは `round(255 * (Y / 255) ** 0.75)` で、逆数ではありません。
輝度処理にはOpenCVのRGB↔YCrCbを使い、透過PNGのalphaを保持します。
CLAHEはclipLimit=2.0、希望タイル数8×8から始め、各辺のタイル数を
`min(8, max(1, 辺の画素数 // 32))` に抑えます。実際のタイル数を結果へ記録します。
拡大は縦横比を維持し、出力が6400万画素を超える場合は縮小せずエラーにします。

手元の取得スクリプト既定の配置先を使う例です。APIキー不要・実API呼出し0回で、
元ROIと2候補を生成し、保存した記録から同じ画像を再生成します。
画像の場所が異なる場合は入力パスを変更してください。

```bash
stage3_result_path=$(uv run --locked --offline python src/cli.py \
  data/dev_external/measurebench/images/pressure_gauge_104_045c9dd6.jpg \
  --mode mock --recipes resize_gamma clahe_mild)

uv run --locked --offline python src/replay_corrections.py "$stage3_result_path"
```

ROI未指定なら画像全体の補正です。文字盤だけを対象にする場合は、
EXIF向き補正後の座標で `--roi X1 Y1 X2 Y2` を明示し、単位と全目盛も含めてください。
`--auto-roi` を明示しない場合、自動ROI検出は行いません。均一な白黒画像等では補正後も同じ見た目になることがあり、
補正による変化は精度改善の証明ではありません。

各runの保存内容：

```text
source.jpg / source.png
normalized.png
roi.png
candidates/c1_resize_gamma.png
candidates/c2_clahe_mild.png
result.json
```

`result.json` の `candidates` は未補正のc0と補正候補c1/c2を含みます。
処理版、各処理の実パラメータ、ライブラリ版、元ROIのハッシュ、出力のファイル・画素ハッシュを保存します。
全候補の親はc0です。最上位の `reading` / `inference` は未補正ROIの結果を維持し、
`selected_candidate_id=null`、`auto_accept=false` とします。
画質指標は輝度分布等の参考情報であり、信頼度や採用スコアではありません。

実APIで候補も読む場合は、段階2の認証・料金設定後に `--mode inference` と
`--provider-config examples/openai_gpt6_luna.json` 等を指定します。
未補正ROIと候補を同じ質問・設定で別々に読み、前の候補値や正解を渡しません。
候補2件なら通常3送信、再試行を含め最大6送信です。
観測全体で期限を共有し、金額枠は引き続きバッチ全体で共有します。
候補生成・読取の失敗も結果に残し、CLIは非ゼロで終了します。
今回は実APIを呼ばず、画像処理・モック・模擬HTTPだけで検証しました。

再生成コマンドはROIと保存パラメータを検証し、API・モデル読取を行いません。
元画像ファイルや元の補正候補が移動していても、run内のroi.pngとresult.jsonがあれば再生成できます。
元のrunは変更せず、`outputs/replays/replay_<UUID>/` に画像とreplay.jsonを保存します。
画素不一致は終了コード1、実行環境の差やPNGバイト列の一致は別項目で報告します。
詳細と検証結果は [段階3の実装記録](docs/task03_stage3.md) を参照してください。

### 段階3追加：GPTによるROI提案と目視確認

`--auto-roi` を指定すると、設定したVLMへ対象領域だけを問い合わせてから、
Pythonが座標を検証し、元解像度から切り出します。Readerは別のリクエストで読みます。
既存の手動ROI・画像全体の処理は引き続き使用できます。Plannerの実装は段階4です。
旧 `meter_agent.py` のROI・数値同時推定は呼び出しません。

まずROIだけを取得して、枠を目視確認する手順を勧めます。
以下は**実APIを呼ぶコマンド**です。クラウドでの検証では実行していません。
新CLIは.envを自動読込しないため、手元のAPIキーを環境変数へ読み込みます。

```bash
uv sync --locked
set -a
source .env
set +a

roi_result_path=$(uv run --locked python src/cli.py \
  data/dev_external/measurebench/images/pressure_gauge_104_045c9dd6.jpg \
  --mode inference \
  --provider-config examples/openai_gpt6_luna.json \
  --api-key-env OPENAI_API_KEY \
  --auto-roi --roi-only \
  --target-type analog \
  --observation-id measurebench_pressure_gauge_104 \
  --question "Read the tire pressure gauge using the psi scale.")

roi_run_dir=$(dirname "$roi_result_path")
echo "$roi_run_dir"
# macOSで提案枠を表示します。
open "$roi_run_dir/roi_preview.png"
```

`roi_preview.png` の緑枠に、文字盤全体・針先・目盛・単位が収まるか確認します。
`roi.png` が実際の切り出し画像です。枠・注釈はReaderの画像に混ぜません。
対象が曖昧／見つからない場合は `roi.png` を生成せず、`needs_review` と理由を保存します。
技術エラーは結果を保存して終了コード1になります。確認前にJSONの `roi_proposal.status`
も確認してください。`--roi-only` は測定値を読みません。

確認後、保存した提案を使って切り出し・固定補正・読取を行います。
ROI用のAPIを再び呼ばず、質問文・計器種別・観測IDは提案から引き継ぎます。

```bash
roi_read_result_path=$(uv run --locked python src/cli.py \
  data/dev_external/measurebench/images/pressure_gauge_104_045c9dd6.jpg \
  --mode inference \
  --provider-config examples/openai_gpt6_luna.json \
  --api-key-env OPENAI_API_KEY \
  --roi-proposal "$roi_run_dir/roi_proposal.json" \
  --recipes resize_gamma clahe_mild)

uv run --locked --offline python src/replay_corrections.py "$roi_read_result_path"
```

枠を修正する場合は、上の再利用コマンドに `--roi X1 Y1 X2 Y2` を追加します。
EXIF補正後の元画像座標・右下を含まない区間です。元の提案と過去runは保持し、
新runの `roi` に修正座標、`roi_source=manual_override` を保存します。
曖昧だった提案にも手動座標を指定できます。手動指定は読取値の確認済み扱いにはしません。
提案の再利用では入力ハッシュ・補正後画素・画像サイズ・質問・計器種別・座標変換を検証し、
別画像や別対象への誤適用を拒否します。入力のファイル名変更だけでは拒否しません。

一度に提案から読取まで実行する場合は、最初のコマンドから `--roi-only` を外し、
必要なら `--recipes resize_gamma clahe_mild` を追加してください。
この場合はROI提案1回＋未補正／2補正の読取3回が通常の構成で、
再試行を含め**観測run全体で最大6送信・120秒**を共有します。
設定の `max_api_calls` は各リクエストの最大試行数、`max_observation_api_calls` は
ROIと全Reader共通の最大送信数（1〜6、既定6）です。上限到達時は未実施の処理も記録します。
料金枠は同一CLIバッチ内で共有し、未設定の料金を0として記録しません。
上の目視確認を挟む2コマンドは別runで、それぞれに回数・期限・費用枠が適用されます。
再利用結果の `roi_proposal.inference` は過去の提案時の通信記録であり、
最上位の `api_calls` は今回の送信だけを数えます。

追加の保存内容は `roi_locator_input.png`（GPTへ渡した画像）、`roi_proposal.json`
（提案座標・応答・モデル・プロンプト版・使用量）、`roi_preview.png`（確認用の枠）です。
GPTへの入力はEXIF補正後の全体画像を長辺1600画素以下へLanczos縮小したものです。
提案座標を検証後、各辺の実倍率で元画像へ戻し、左上は切捨て・右下は切上げます。
元画像を変更せず、全補正候補には同じ切り出しROIを使います。
この段階ではROI提案も測定値も自動承認しません。

```bash
# APIキー不要・実API通信なしの自動テスト
uv run --locked --offline python -m unittest discover -s tests -v

# 保存済み提案があれば、再利用と補正だけをAPIなしで確認できます。
uv run --locked --offline python src/cli.py \
  data/dev_external/measurebench/images/pressure_gauge_104_045c9dd6.jpg \
  --mode mock --roi-proposal "$roi_run_dir/roi_proposal.json" \
  --recipes resize_gamma clahe_mild
```

詳細と検証範囲は [ROI提案の実装記録](docs/task03a_vlm_roi.md) を参照してください。

### 仕様v0.2・段階4：Plannerによる補正選択

`--agent` で、未補正ROIのReader → Planner → 補正 → 独立Readerを実行します。
Plannerは登録済みの未試行レシピ1つ、終了、再撮影推奨のいずれかを選び、補正は最大2候補です。
`--recipes` との同時指定はできません。指定しなければ従来の固定補正・ROI処理を維持します。

APIキーなし・外部通信なしで流れを確認できます。MockPlannerのレシピ選択は固定された
動作確認用で、画像を判断した結果ではありません。MockReaderの値・単位も不明のままです。

```bash
uv run --locked --offline python src/cli.py \
  data/dev_external/measurebench/images/pressure_gauge_104_045c9dd6.jpg \
  --mode mock --agent

uv run --locked --offline python -m unittest discover -s tests -v
```

次は実APIを使う例です。先のROI確認で設定した `roi_run_dir` を使い、
保存済み `roi_proposal.json` を再利用します。新しいROIのAPI呼出しは行いません。
新CLIは.envを自動読込しないため、キーを環境へ読み込んでから実行します。

```bash
set -a
source .env
set +a

agent_result_path=$(uv run --locked python src/cli.py \
  data/dev_external/measurebench/images/pressure_gauge_104_045c9dd6.jpg \
  --mode inference --agent \
  --provider-config examples/openai_gpt6_luna.json \
  --api-key-env OPENAI_API_KEY \
  --roi-proposal "$roi_run_dir/roi_proposal.json")

echo "$agent_result_path"
```

手動の `--roi`、`--inputs` の入力専用JSONL、`--auto-roi` も併用できます。
ROI提案・Reader・Planner・最終比較のすべてで**最大6送信・120秒**と料金枠を共有します。
ROIを新規提案すると残りの送信枠が減るため、確認済み提案の再利用を勧めます。
最終比較が必要でも予算がなければ実行せず、`needs_review` と未実施の理由を保存します。

Reader同士の値・単位・目盛に矛盾があれば、追加の候補や多数決で解消せず停止します。
補正候補だけが読めた場合は、残予算があれば元ROIとその候補を並べた最終比較を1回行います。
最終比較は既存候補IDの支持・読取可否・矛盾だけを返し、新しい数値の出力を拒否します。

アナログの許容差は未設定が既定です。食い違う値を同一とみなしたい場合だけ、
`--comparison-policy <JSONファイル>` へ、対象単位に合わせた設定を指定します。
例えば `{"analog_absolute_tolerance":"0.02","unit":"MPa"}` です。
入力JSONLでは `acceptance_policy.comparison` に同じオブジェクトを置けます。
許容差は同じ単位のアナログ値にだけ適用し、Pa↔MPa等の換算はしません。
これは候補間の一致判定だけであり、自動採用基準の校正を済ませた扱いにはしません。

`result.json` の主な確認箇所：

| フィールド | 内容 |
|---|---|
| stage | 4 |
| candidates | 未補正c0と最大2候補。画像・処理・独立した読取・通信記録 |
| agent.events | Planner指示、補正の完了／失敗／未実施、最終比較と各通信記録 |
| agent.stop_reason | 終了・再撮影・候補上限・不一致・API／予算上限等の停止理由 |
| agent.comparisons | 候補間の値・単位・目盛等の矛盾 |
| selected_candidate_id | 要確認の表示候補ID、またはnull。運用上の採用値ではない |
| next_action | human_review または recapture |
| api_calls | 今回の全役割の実際の送信数。過去の保存ROI提案分は含めない |

最上位の `reading` / `inference` は従来どおり未補正c0の記録を維持します。
表示候補の抽出値は `selected_candidate_id` に対応する `candidates[].reading` を参照してください。
`auto_accept=false`、`effective_value=null`、`effective_unit=null` のままです。
同じVLMによる一致や最終比較だけで、正解や精度改善を証明したことにはしません。

技術エラー・無効な指示があればCLIは結果を保存して終了コード1、
通常の停止・再撮影推奨・矛盾・予算上限による要確認は終了コード0です。
終了コード0でも読取成功とは限らないため、`status` と `reason_codes` を確認してください。
補正候補が正常に生成されていれば、段階3と同じ `src/replay_corrections.py` で
API・Plannerなしに再生成できます。候補なし、または未生成候補を含むrunの再生成は拒否します。

今回の検証は140自動テストと公開開発画像20枚の通信モックです。
実APIでのPlannerの判断・読取精度・費用は未検証です。
詳細は [段階4の実装記録](docs/task04_stage4.md) を参照してください。

### 追加レシピ：対応点を用いた傾き補正

`perspective_rectify` を追加しました。文字盤平面上の対応点からOpenCVで射影変換します。
明るさ補正と組み合わせず、元ROIから独立した1候補として生成します。
最大2候補・合計6送信・120秒の制限は維持します。Readerのプロンプトと自動採用方針は変更しません。

`--agent --allow-perspective` でPlannerの選択肢へ追加します。
Plannerがこのレシピを選んだ場合だけ、別のVLM要求で物理的な対応点を提案させます。
レシピの選択を強制する引数ではありません。保存ROIを再利用する実APIの実行例です。
`roi_run_dir` は前述のROI確認で保存したrunのディレクトリです。

```bash
set -a
source .env
set +a

uv run --locked python src/cli.py \
  data/dev_external/measurebench/images/pressure_gauge_104_045c9dd6.jpg \
  --mode inference --agent --allow-perspective \
  --provider-config examples/openai_gpt6_luna.json \
  --api-key-env OPENAI_API_KEY \
  --roi-proposal "$roi_run_dir/roi_proposal.json"
```

自動提案の対象は円形のアナログ文字盤で、文字盤上の物理的な「上・右・下・左」を
特定できる場合です。**見かけの楕円の端や、円を囲む四角の角を対応点にしません。**
物理的な向きの根拠がない、手で隠れている、印刷面とケースの縁を区別できない場合は、
曖昧として補正を見送ります。今回のタイヤ圧力計でも補正を見送る可能性があります。
自動提案は縮小せず元ROIの座標を使うため、各辺2048画素以下に限定します。
明示的なdigital指定では、未知の縦横比を推測して矩形補正せず、手動対応点を必要とします。

**対応点の数値検証に通っても、物理的に正しい点を選べた保証にはなりません。**
`candidates/cN_perspective_points.png` の点と、その候補の補正画像を目視確認してください。
補正前後の値が食い違えば要確認のままです。

APIなしで人工画像の傾き補正・再生成を試せます。ここで生成する画像は人工の図形で、
実際の計器や測定値ではありません。既知の四隅と正方形の縦横比を使用します。

```bash
demo_dir=$(uv run --locked --offline python examples/demo_perspective.py)
demo_result_path=$(uv run --locked --offline python src/cli.py \
  "$demo_dir/oblique.png" --mode mock \
  --recipes perspective_rectify \
  --perspective-config "$demo_dir/perspective.json")

uv run --locked --offline python src/replay_corrections.py "$demo_result_path"
```

`front.png` が元の図形、`oblique.png` が人工的に傾けた入力です。
補正画像は結果runの `candidates/c1_perspective_rectify.png` に保存されます。

実画像で手動対応点を使う場合は、次の形式のJSONを作り `--perspective-config` へ指定します。
以下の数字は人工デモ専用です。実画像へ流用しないでください。

```json
{
  "reference": "rectangle",
  "source_points": [[60,22],[340,55],[304,260],[25,222]],
  "aspect_ratio": 1.0,
  "evidence": "Artificial known square drawing"
}
```

- 座標は **EXIF補正・ROI切り出し後の画像内の画素中心**。元画像座標ではありません。
  範囲は `0 <= x <= ROI幅-1`、`0 <= y <= ROI高さ-1` で、ROIの右下を含まない範囲指定とは区別します。
- `rectangle` の点順は左上・右上・右下・左下。`aspect_ratio` は既知の実物の幅÷高さです。
  点の見かけの距離から縦横比を推測しません。
- `dial_cardinals` は物理的な円の上・右・下・左で、`aspect_ratio` はnullまたは省略します。
  円のどの位置か特定できない場合、楕円の端を代入しないでください。
- 不正な型・非有限値・範囲外・逆順・交差・重複・小さすぎる領域・不安定な変換・
  元ROIの外側が必要な出力を拒否します。無効な点を自動補正しません。

手動設定は単一画像に対応し、`--agent` または `--recipes perspective_rectify` と併用します。
`--agent` では、設定があるだけでレシピを選択肢へ追加し、対応点用のAPIを省略します。
固定レシピで実行する場合は必ず手動設定が必要です。
画像・ROI・向きが変わった場合は、対応点の再確認が必要です。

記録は `candidates[].geometry`（対応点の出典、根拠、状態、通信記録）と
`candidates[].steps[].parameters`（対応点、出力サイズ、変換行列、補間法）を確認します。
対応点の曖昧さは `geometry_ambiguous`／`geometry_not_found` として停止し、
未生成の候補を読取結果へ混ぜません。数値不正・API障害は技術エラーとして残します。
この候補の `api_calls` は、対応点用APIとReaderの送信数の合計です。

保存行列は対応点から検証したうえで再生成に使用します。再生成ではVLMを呼びません。
元画像・元ROIを変更せず、RGB/RGBAに対応します。RGBAはalphaを乗じて補間し、色にじみを抑えます。
出力長辺は元ROIの長辺以下かつ最大2048画素とし、別途2倍拡大を組み合わせません。
詳細と検証範囲は [傾き補正の実装記録](docs/task04a_perspective.md) を参照してください。

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
├── pyproject.toml # uv用の依存関係
├── uv.lock        # uvが生成する依存関係のロックファイル
├── .env.example   # APIキー設定例
└── environment.yml
```
