# 段階4：Plannerと最大2候補の再試行

作業ブランチは `feature/image-agent-web`。仕様v0.2第4.3節、第5〜6節、第9節の段階4を実装した。
ユーザーの方針を継続し、クラウドで実APIは呼んでいない。実APIでの確認はローカルで行う。

## 処理と停止条件

新CLIの `--agent` で段階4を有効にする。既存の `--recipes` と排他で、
引数なし・手動ROI・固定補正・ROI専用確認の動作を維持する。

1. 既存のEXIF・ROI検証、必要ならROI提案／保存提案の再利用を行う。
2. 未補正c0をReaderで読み、元ROI・画質指標を保持する。
3. Plannerへ元ROI、質問・対象種別、画質指標、今回の候補履歴、未試行レシピ、残り枠を渡す。
4. `apply` の登録済みIDをPythonが検証する。元ROIから独立した補正を生成し、別のReader要求で読む。
5. 候補間の矛盾があれば停止する。読取可能な補正候補があれば最終確認へ進む。
   どちらもなければ、最大2回までPlanner／補正／Readerを繰り返す。
6. `stop`・`recommend_recapture`、不正指示・API障害・予算上限でも終了する。
7. 元ROIが使える場合はc0を要確認の表示候補として優先する。
   補正候補だけが読める場合は残予算内で最終比較を1回行い、既存候補の支持だけを記録する。

Planner出力は `action`、`recipe_id`、`reason`、`reason_codes` の厳密な契約とした。
登録外・既試行・不整合・項目不足の指示は実行しない。出力修正依頼も自動送信しない。
2候補の上限はループと実行境界の両方で検証する。任意コードやモデル指定のパラメータを実行しない。

Readerには候補の画像・質問・対象種別だけを渡す。
Plannerには今回の候補履歴を渡すが、過去の観測、前日値、正解、evaluator、採用方針内の任意データを渡さない。
データセットの質問は保持し、画像内文字や履歴の文章を指示として扱わないよう役割プロンプトを分離する。
画質指標は参考情報として扱い、読取成功確率や採用点数へ変換しない。

## 比較と採用状態

値は `Decimal(str(value))` で比較する。デジタルのraw表記を保存し、
数字表記と構造化値の不整合、小数点・符号の変化を確認する。
アナログは許容差未設定なら異なる値を一致扱いしない。
明示した `analog_absolute_tolerance` と `unit` が同じ単位のアナログ同士にだけ有効になる。
許容差はCLIの `--comparison-policy` または入力の `acceptance_policy.comparison` へ指定する。
候補の一致確認専用であり、自動採用を有効にする設定ではない。

値・単位・計器種別・既知の目盛情報の不一致、対象／目盛の曖昧さを記録する。
単位倍率を文字列置換で揃えず、欠損値を0や既知値で補完しない。
一部不明の読取と、明示的に矛盾する複数の読取を区別する。
多数決やモデルの自己申告確信度で選択しない。

最終比較には元ROI・補正候補の2画像と、その2つの既存記録だけを渡す。
出力は `supported_candidate_id`、`readable`、`contradictions`、`reason` に限定する。
別候補ID、新規の値・単位、支持と矛盾の同時出力等を拒否する。
矛盾がある場合は最終比較に判断を委ねず停止する。最終比較が必要なのに予算がなければ選択しない。

`selected_candidate_id` は要確認の表示候補であり、採用済みの測定値ではない。
校正済み採用基準をまだ実装していないため、全件 `auto_accept=false`、
`effective_value=null`、`effective_unit=null` とする。`success` へは分類しない。
最上位の `reading` / `inference` は未補正c0の記録を維持し、補正候補の表示値で上書きしない。

全読取で値・単位が得られず正常に終了した場合は `unreadable`、
部分読取・矛盾・後段障害・予算不足・mockは `needs_review`、
最初のReaderも失敗して有効な観察がない場合は `error` とする。
再撮影推奨は `next_action=recapture` に記録し、人の確認状態を付けない。

## 共通予算と記録

ROI・Reader・Planner・最終比較は同じProviderの総回数・期限・料金枠を共有する。
別Providerの組合せは拒否する。通信再試行も最大6送信へ含め、
120秒以内の共有期限、1回最大30秒、論理リクエストの一時障害再試行最大1回を維持する。
画像処理とAPI送信の前に残り枠を確認する。料金枠も処理前に確認し、予約・精算は送信時に行う。
同じCLIバッチ内では費用予約を共有し、使用量不明の失敗は予約を保持する。

通常の最大構成はReader1＋Planner／Reader各2＋最終比較1で6送信。
新規ROI提案・通信再試行で枠を使うと、残りの工程は実行せず履歴へ残す。
確認済みROI提案の再利用は今回の通信枠を消費しない。
モデル・料金設定例は既存のものを使い、未設定の料金はnullのままにする。

`result.json` に `stage=4`、`agent` を追加した。
`agent.events` はPlannerの指示、補正の完了／失敗／未実施、最終比較とそのtraceを含む。
Readerのtraceは各 `candidates[].inference`、ROIのtraceは `roi_proposal.inference` に残る。
`agent.comparisons` は候補間の矛盾、`agent.stop_reason` は停止理由を持つ。
最上位 `api_calls` は今回の全役割の実送信数で、過去のROI通信を加算しない。
APIキー・ヘッダー・HTTPエラー本文は保存しない。

既存runを上書きせず、元画像・ROI・候補のハッシュを処理中に確認する。
固定補正とPlannerの候補生成を同じExecutorへ分離し、同一レシピなら同じ処理を使う。
段階4の生成済み候補も、既存のオフライン再生成CLIで検証できる。
元のPlannerを再実行するのではなく、記録されたレシピ・パラメータから画像を再生成する。

CLI終了コードは、技術エラー・無効な指示なら1、
通常停止・再撮影推奨・読取不能・矛盾・予算上限で保存できた場合は0とした。
終了コード0を読取の正しさとして扱わず、保存された状態と理由を確認する。
従来の段階1〜3の終了コードの扱いは維持した。

## 実行手順

[READMEの段階4手順](../README.md#仕様v02段階4plannerによる補正選択) を参照。

```bash
uv sync --locked
uv run --locked --offline python -m unittest discover -s tests -v
uv run --locked --offline python src/cli.py \
  data/dev_external/measurebench/images/pressure_gauge_104_045c9dd6.jpg --mode mock --agent
```

実APIでは既存設定・環境変数名を使い、保存したROI提案を指定して `--agent` を追加する。
`--recipes` は付けない。画像パス・ROI提案ファイルは各自の保存先に合わせる。
クラウドでは実APIのコマンドを実行していない。

## 検証

既存106件＋段階4の34件、計140件の自動テストが成功した。
登録外・重複・不正JSON・不整合なPlanner出力、終了／再撮影、独立したReader、
候補の矛盾、十進数・符号・raw表記・単位・許容差、最終比較の制限、API障害、
上限6・期限・費用枠・未実施記録、元画像／候補の不変性、CLI、再生成を確認した。
HTTP応答・料金・読取値は架空のテスト用で、正解ファイルを使っていない。

公開開発画像20枚では、全件「値・単位不明」の模擬Readerと固定された模擬Planner指示で接続確認をした。
圧力計10、電力量計6、流量計1、パルスオキシメーター1、電子ノギス2の入力・保存を確認した。

- Planner経路20run保存。各run5模擬送信、2候補生成。
- MockPlanner／MockReaderのCLIでも20run保存。実API通信0、架空の測定値を作らない。
- ユーザー添付の実GPTによるROI提案を再利用し、245×199の元ROIから追加1runを保存。
  ここでのPlannerとReaderも模擬応答である。
- 合計105模擬HTTP送信、実API呼出し0、外部ソケット呼出し0。
- 21runの42補正画像をオフライン再生成し、画素・PNGファイルハッシュとも全件一致。
- 元画像20枚・入力・正解・provenance・保存したROI提案・再生成元runの不変性を確認。

検証記録は `outputs/stage4-simulated-http-m6e9r57u/validation_summary.json`。
画像・正解・出力はGit管理外である。実APIでのPlanner判断・最終比較、数値の正解率、
補正の改善効果、実使用量・料金は未検証で、ローカルでの動作確認と段階5の評価へ残す。
20枚は開発用データであり、工場画像の最終評価データとしては使わない。

## 変更ファイル

| ファイル | 理由 |
|---|---|
| src/agent.py | 最大2候補の状態・停止制御、最終比較、全役割の回数集計 |
| src/planner.py | Planner／最終比較の独立したVLM接続、厳密な応答検証、通信しないMock |
| src/validator.py | 十進数・単位・raw表記・目盛・許容差の保守的な比較 |
| src/candidate_io.py | 固定補正と適応補正が共用する独立候補生成・ハッシュ検証 |
| src/recording.py | 段階4を既存のROI・観測・保存処理へ接続 |
| src/schemas.py | Planner・最終比較・イベント・比較結果の型、段階4・再撮影状態・表示候補ID |
| src/vlm_request.py | 最終比較で2画像を同じ構造化リクエストへ渡す |
| src/provider.py | 費用枠の非破壊チェックを共通化し、画像処理前にも検査 |
| src/cli.py | --agent／--comparison-policy、引数・設定・終了コード検証 |
| src/replay_corrections.py | 段階4の保存済み補正にもオフライン再生成を適用 |
| tests/test_stage4.py | 架空HTTPによる34件の段階4テスト |
| README.md、docs/task04_stage4.md | 実行手順・互換性・予算・検証と未実施範囲 |

依存追加はなく、`pyproject.toml` と `uv.lock` は変更していない。
段階5の評価・履歴DB、段階6のWeb、段階7の前日比較・Excel出力は追加していない。
