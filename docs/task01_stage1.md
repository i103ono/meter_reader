# Task 01：仕様v0.2・段階1の実装記録

作業ブランチ：feature/image-agent-web。対象は第9節の段階1のみ。
API接続、補正レシピ、Planner、DB、前日比較、Web、Excelは実装対象外。

## 既存コードの確認と流用

- 作業開始時は work ブランチで追跡・未追跡の変更なし。リモートに存在する
  feature/image-agent-web を取得し、追跡ブランチへ切り替えた。
  クラウド初期設定のfetch対象がmasterだけだったため、対象ブランチのfetch設定を追加した。
  master/mainの変更・マージ、既存ブランチのリセット・上書きは行っていない。
- 適用対象のAGENTS.mdは存在しない。README、仕様v0.2、pyproject.toml、
  uv.lock、.gitignoreと指定された既存モジュールを確認した。
- image_io.load_image は既存CLIが共用するPillowデコード処理。内部デコードを共通化し、
  新入力も同じ処理を利用する。既存関数は向き補正せず、呼出側がImageOps.exif_transposeを行う。
- crop_roi.py と read_meter.crop_config_roi はEXIF補正後のxywh契約。
  既存CLIの引数と出力は維持し、xyxyの新契約を別関数として追加した。
- read_meter.py はデジタルEasyOCR／アナログ円・針検出のCLI。
  meter_agent.py と batch_meter_agent.py はOpenAI呼出しに接続している。
  新仕様の欠損・目盛・観測・実行・採用状態を備えていないため、そのまま新Readerにはしない。
- test_meter_agent.py は実画像依存の確認CLIであり、一般的な自動テストスイートではない。
  --mock のデジタル値は既存の固定架空値で、アナログはローカル処理。
  新しいMockReaderは固定値も返さず、値・単位・目盛を欠損にする。
- evaluate.py は単一の正解数値をCSVで比較する既存評価用CLI。
  MeasureBenchの区間正解／evaluatorを扱う新評価器ではなく、今回は接続・変換しない。
- data/、outputs/、.env等は既存.gitignoreで除外済み。
  初期実装時は入力画像、inputs.jsonl、labels.jsonl/csv、provenance.jsonが存在しなかったが、
  追加添付によりdata/measurebench/へ配置し、実画像20枚を検証済み。
  MeasureBench取得スクリプトは未追加。既存outputsの人工デバッグ画像は開発用実画像に数えない。

## 変更ファイルと理由

| ファイル | 変更理由 |
|---|---|
| src/schemas.py | Pydanticで入力・Reader出力・目盛・最終結果を検証。欠損、矛盾、有限数、日時、将来必須のmeter_idを表現 |
| src/image_io.py | 既存デコードを共用し、形式検証・入力ハッシュ・EXIF正規化・厳密ROI・xywh変換・排他PNG保存を追加 |
| src/timestamps.py | 撮影日時の根拠・タイムゾーンを保存。UTC変換し、不明・不正・DSTの曖昧さを処理日時と分離 |
| src/reader.py | 交換可能なProtocol、正解情報を含まないReaderInput、外部通信なしのMockReader |
| src/recording.py | 観測と実行を分離し、元バイト列・正規化画像・ROI・結果JSONを新規ディレクトリへ記録 |
| src/cli.py | 1枚／入力専用JSONLの入口。質問・provenance参照を保持し、結果パス・入力件数・エラーを返す |
| tests/test_stage1.py | 実画像・API・OCRモデルに依存しない契約・保存・CLI回帰テスト |
| pyproject.toml / uv.lock | Pydantic依存を追加して同期。他の依存の版は変更しない |
| README.md / 本記録 | uv実行手順、ROI・日時・保存の契約、20枚の種類別扱い、未検証範囲を説明 |

新CLIのROIはEXIF正規化後のxyxy、右端・下端を含まない。
旧CLI／設定JSONはxywhのままで、暗黙の形式判定はしない。
元画像のコピーは元バイト列、正規化・ROIはRGB/RGBA PNGで保存し、透過情報を維持する。
保存はoutputs/runs/run_<UUID>/以下、ファイル作成は排他で行う。
入力の観測IDやmeter_idをパスとして使用しない。
再読取では同じobservation_idを指定し、run_idは新たに発行する。
DBを持たない段階のため、別実行間の観測IDと実画像の対応の照合は後続段階で扱う。

型は取得できた値0と欠損nullを区別し、readableなのにnullなどの矛盾を拒否する。
目盛のmanual/registry出典は目盛型で表現できるが、Readerが返す出典としては拒否する。
採用ポリシーは後段用に記録するだけで、段階1では自動採用しない。
MockReaderの実行成功は入出力処理の確認に限り、読取状態successへ分類しない。

## 検証と実画像の位置付け

READMEのuvコマンドを利用する。自動テストではJPEG/PNG、破損・偽装形式、
EXIF回転・反転とROI座標、元画像のハッシュ・元バイト列保存、ROI形式変換・不正型・範囲外・空領域、
撮影日時不明・入力日時・EXIF日時・UTC・DST、欠損と真の0、同名画像と再実行の保存先、
Reader交換・正解分離、キーなし／ソケットとHTTP呼出し禁止、CLIと旧crop_roi.pyの互換性を確認する。

人工画像による20行のJSONLも処理し、圧力計10・電力量計6・流量計1・パルスオキシメーター1・
電子ノギス2というカテゴリ情報と質問・観測IDを保持できることを確認する。
この人工テストはMeasureBenchの実画像入力確認でも精度評価でもない。

実画像の入力確認件数は **20/20**。追加添付のimages.zipから、全20枚のSHA-256と
画像サイズを正解メタデータと照合したうえでdata/measurebench/images/へ配置した。
添付4ファイルの原本はバイト単位で維持し、入力とprovenanceのみから
CLI用inputs.stage1.jsonlを別途作成した。固定revisionの外部再取得は接続制限403のため未実施だが、
今回の入力確認はZIP内の画像で完了した。
外部通信を禁止しAPIキーを除いた状態で、全20枚を2回実行して各回20保存・0エラーを確認した。
観測ID・元質問・provenanceを保持し、実行ID・保存先は毎回別で、
元画像・正解ファイル・初回の結果が再実行で変更されないことも確認した。
全件mode=mock、needs_review、値・単位・目盛情報は欠損のままで、自動採用しない。
カテゴリごとの対象表示・二重目盛は実画像と元質問で決め、カテゴリ名から推測しない。
企業画像は最終評価用に残し、この20枚を最終テストセットにはしない。

labels.jsonl/csvはReader入力と分離したまま保存し、公式evaluator/evaluator_kwargs、
区間下限・上限、単位候補、出典を変更しない。
この段階では正解値生成・区間中央値化・数値正解率計算を実施しない。
既存のtest_meter_agent.py --mock（既定画像）とevaluate.py（既定CSV）は、
必要データがないため既定データでの有効な検証はできない。
既存CLIの--help、旧crop_roi.pyの数値ROI、人工データによる既存CLIの確認と区別する。

今回の実行結果：

| 確認 | 結果 |
|---|---|
| uv sync --locked / uv pip check | 成功。ロックのOpenCV 5をそのまま使う環境でも段階1は動作 |
| uv run --locked python -m unittest discover -s tests -v | 28件成功。失敗・スキップなし |
| 人工20行JSONL | 20入力・20保存・0エラー。カテゴリ内訳・質問・IDを保持 |
| 添付の実画像20枚 | SHA-256・画像サイズ20/20一致。全件入力・JSON保存成功、再実行も20/20成功。通信禁止・APIキーなし |
| READMEの人工画像サンプル | 1入力・1保存・0エラー。結果JSONと元画像・正規化画像・ROIを保存 |
| read_meter / meter_agent / test_meter_agent / batch_meter_agent / evaluate --help | 全て成功 |
| 既存test_meter_agent --mock＋人工JPEG1枚 | 成功。既存の固定架空値を返す経路の互換性確認のみ |
| 既存test_meter_agent --mock（既定画像） | 終了1、No input images found。画像不在を確認し、有効な読取テストは未実施 |
| 既存evaluate（既定CSV） | 終了1、data/evaluation_labels.csv不在。評価は未実施 |
| MeasureBench実画像／企業画像の精度 | 未実施。実画像入力20/20と正解データ保存を確認。API呼出し・モデル推論・正解率算出なし |

追加検証の詳細はクラウド内の
`outputs/measurebench-verified-l22c0o5f/validation_summary.json` と
`data/measurebench/import_report.json` に保存した。

uv.lockは既存のOpenCV 5を保持している。既存アナログ処理には
HoughLinesPの戻り値形状に関する互換性問題があり、クラウド初期設定では環境側を4系へ固定していた。
段階1はOpenCV／EasyOCRを呼ばないため、この互換性回避なしでも動作する。
既存アナログ処理の本修正は本タスクに含めない。

## 段階2に入る前の設定

- 読取VLMのプロバイダ・モデルID、画像入力形式・解像度設定、構造化出力、利用可能な地域・APIを決める。
  Reader Protocolの実アダプターとして実装し、モデル名・API固有の処理を記録処理へ埋め込まない。
- 選択プロバイダの認証とHTTPS接続先を環境設定で安全に用意し、画像の外部送信を許可する範囲を確認する。
  既存OpenAI CLIのキーを新インターフェースが自動的に利用できると仮定しない。
  クラウドの予約済みOPENAI_名を直接シークレット対象にできない場合は、許可された別名をプロセスへ明示的に対応付ける。
- APIの画像／トークン単価を確認し、1画像・総実行の金額上限、呼出数・時間上限、
  利用量の記録方法と再試行条件を決める。料金不明を0円にしない。
- 元の20枚・質問・provenanceを配置し、対象表示を確定する。区間正解を維持する評価器と
  自動採用基準の検証は別段階で行い、未検証のVLM候補は引き続きneeds_reviewとする。
