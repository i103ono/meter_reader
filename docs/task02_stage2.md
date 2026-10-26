# Task 02：VLM接続と未補正ROIの読取

作業ブランチ：feature/image-agent-web。
ユーザー指定の範囲は「実装と通信モック検証まで」、希望モデルはGPT-6 Luna。
実APIによる計測や精度評価、補正レシピ、Planner、DB、Web、前日比較は今回の対象外。

## モデルとプロバイダ

公式OpenAI Python SDKのChatModel一覧で `gpt-6-luna` を確認した。
ResponsesModelはChatModelを参照している。これらはモデルIDの公開資料で、
利用者の認証権限、画像入力、strict JSON Schemaの実動作を保証するものではない。
今回の確認は公開ソースの読取に限り、OpenAI APIには接続していない。

- https://github.com/openai/openai-python/blob/main/src/openai/types/shared/chat_model.py
- https://github.com/openai/openai-python/blob/main/src/openai/types/shared/responses_model.py
- https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_format_text_json_schema_config_param.py
- https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_create_params.py

設定例はexamples/openai_gpt6_luna.json。別モデルはCLIまたは設定ファイルで明示する。
認証値は環境変数から送信時に取り出し、設定ファイルや結果へ保存しない。
APIキーの構造を決めつけず、プロキシ用プレースホルダーもそのまま扱う。
今回の範囲では認証・ネットワーク・料金設定の変更は必要なく、下書きにも追加していない。

## 実装した接続経路

1. 段階1の入力検証・EXIF補正・ROI保存を共用する。
2. VLMReaderが保存した未補正ROIをPNGのdata URLにし、対象の質問・種別ヒントと独立したリクエストを構築する。
3. OpenAIProviderがResponses APIへ送信する。store=false、truncation=disabled、
   出力トークン上限、strict JSON Schemaを設定する。HTTPリダイレクトは追跡しない。
4. ReaderOutputと同じ型・必須null・目盛出典制約をローカルでも検証する。
   不正なJSONを正規表現等で修復したり、欠損項目を測定値で埋めたりしない。
5. 元の出力文字列、設定・モデル・使用量・試行履歴と検証済み候補を結果へ保存する。
   既存の観測ID、入力ハッシュ、質問とprovenanceは維持する。

Readerへ渡す情報は引き続きimage_path/target_type/questionだけ。
正解CSV、labels.jsonl、evaluator、区間正解、過去の値、採用ポリシーをモデルへ渡さない。
画像内の文や質問文を処理手順の変更命令として扱わないよう、上位プロンプトにも明記した。
二重目盛・複数表示で対象が曖昧なら勝手に選ばず、欠損やissue_codesとして残す。

単位の正規化はm³→m^3、m³/h→m^3/h、kW·h/kW h→kWhという明示表だけ。
数値と元の数字表記を保持し、Pa→MPa等の倍率変更はしない。
目盛情報が不明でも、読めた値や単位は保持する。

## エラー、時間、料金

- 一時的な通信障害・429・一部5xxだけ最大1回再試行する。認証・権限エラーは再試行しない。
  API呼出し回数には失敗・再試行も含める。
- Retry-Afterの秒数・HTTP日時を扱う。5秒を超える待機や残期限に収まらない待機は再送せずエラーとして残す。
- 1回の通信タイムアウトは最大30秒かつReaderの残期限以下。Reader期限は最大120秒。
  前後で期限を確認する。ソケットタイムアウトを使うため、OSスケジューリングまで含めた
  プロセス強制停止や、サーバー側の課金キャンセルを保証する仕組みではない。
- 未校正の結果をsuccessへ分類しない。値・単位が両方不明ならunreadable、
  部分欠損や候補ならneeds_review、通信・形式・拒否・途切れならerror。
- エラーになった実行も結果JSONを保存し、CLIは非ゼロで終了する。
  API呼出し前の認証欠損・予算超過は0送信として記録する。
- 使用量が返らない呼出しのトークン数や料金はnull。設定した単価がある場合だけDecimalで見積る。
  金額上限を使う場合は、正しいモデル・最新単価・全コンテキストの入力上限を設定する必要がある。
  設定と実際の単価・上限が異なる場合の請求額は保証しない。設定例に料金の推測値は入れていない。
- 上限はCLIバッチで共有し、送信前に最悪値を予約する。使用量不明の失敗分は解放しない。
  個々の実行の推定料金と、バッチで保持する予約額を区別して記録する。
- キー・認証ヘッダー・HTTPエラー本文を結果や例外メッセージへ出さない。
  不正な設定にキーを書いた場合も、CLIの検証エラーは値を表示しない。

## 変更ファイル

| ファイル | 内容 |
|---|---|
| src/provider.py | OpenAI ResponsesのHTTP境界、設定、再試行、タイムアウト、使用量、共有費用枠 |
| src/vlm_reader.py | 交換可能なReader実装、プロンプト・構造化出力、単位表、読取の検証・記録 |
| src/schemas.py | 段階2、unreadable、API試行と使用量・設定の追跡情報を追加 |
| src/recording.py | 既存画像保存と接続し、技術エラーと読取不能を分け、実際の試行数を記録 |
| src/reader.py | 既存ReaderErrorへ機械判定用codeを追加。ReaderInput/返り値は互換 |
| src/cli.py | inferenceモード、設定・モデル・キー変数名・件数制限、エラー結果を含む終了コード |
| examples/openai_gpt6_luna.json | 希望モデルの非秘密設定例。認証値・推測料金は含めない |
| tests/test_stage2.py | 架空のHTTP応答による26件の自動テスト |
| README.md / 本記録 | 実行方法、検証の意味、料金・認証・実APIの未検証範囲 |

標準ライブラリと段階1の依存だけを使い、新しい依存追加・ロック変更は不要だった。
旧meter_agent.py等のAPI経路は変更していない。新CLIの既定mockモードも維持している。
RunResultのinferenceは追加の任意項目で、段階1の結果も引き続き読み込める。

## 検証結果

```bash
uv run --locked --offline python -m unittest discover -s tests -v
uv run --locked --offline python src/cli.py --help
```

- 54件成功（既存28件＋段階2の26件）。失敗・スキップなし。
- 架空のアナログ・デジタルHTTP応答から、値・単位・目盛・部分欠損・読取不能を保存できる。
  これはプロバイダの実出力・実精度・実利用量ではない。
- 実画像20枚をHTTP模擬接続へ通し、20入力・20保存・0技術エラー。
  模擬応答は全件「値・単位不明」で、モデルの推論を装う値は作っていない。
- この接続テストではソケット生成を禁止し、実API送信は0件。
  テスト専用モデル名・認証値・利用量を使用し、結果は模擬検証と明記したレポート配下に置いた。
- 元画像・入力・正解・provenanceのハッシュ不変、質問と観測IDの保持、正解の非送信を確認した。
- 料金情報不明、認証エラー、無効JSON、拒否、途切れ、429と再試行上限、予算不足・共有、
  モデル切替、GPT-6 Luna設定例、認証情報の非出力を確認した。

クラウドの検証レポート：
`outputs/stage2-simulated-http-mj7bf5bl/validation_summary.json`。
実APIによるアナログ・デジタルの読取という段階2本来の実機完了条件は、今回の希望範囲に含まれない。
次回、認証・許可接続先・現在のモデル仕様・料金上限を確認したうえで少数画像から接続を検証する。
企業画像による最終評価や、公式区間正解による精度評価は引き続き後続の範囲とする。
