# アナログReaderの工程別記録と整合性検査

## 参考にした研究と適用範囲

Reitsma, Keller, Blomqvist, Siegwart, **Under Pressure: Learning-Based Analog Gauge Reading In The Wild** (2024):

- 論文: https://arxiv.org/abs/2404.08785
- 著者の公式実装: https://github.com/ethz-asl/analog_gauge_reader
- 中間結果の保存: https://github.com/ethz-asl/analog_gauge_reader/blob/main/pipeline.py
- 工程別評価: https://github.com/ethz-asl/analog_gauge_reader/blob/main/evaluation/full_evaluation.py

2026-10-27に公式README、pipeline.py、evaluation/full_evaluation.py、evaluation/constants.pyを確認した。
この環境ではarXiv本文への通信が拒否されたため、実装方針の根拠は閲覧できた著者コードによる。
同コードは計器の検出、目盛りキーポイント、OCR、針セグメンテーション等を保存し、
工程別の正解アノテーションと比較して、計器のIoU、目盛りキーポイントの位置指標、
数値領域の検出率、針マスクのIoU等を評価している。

本変更は「最終値だけでなく工程ごとの観測と中間表現を保存する」という設計を取り入れた。
同研究の学習済みモデルやコードを導入・再現したものではない。
現在のMeasureBench用labelsには最終値と単位の正解はあるが、針先・目盛り点等の正解位置がない。
したがって本変更の工程状態はVLMの自己申告、矛盾検査は中間表現間の整合性であり、
工程の実際の正誤や失敗原因の確定ではない。
工程別精度を測るには、別途、評価専用の位置・文字・対象尺度のアノテーションが必要である。
正解ラベルは引き続きオフライン評価でのみ使用する。

## 読取の変更

Readerプロンプトを `meter-reader-v03-analog-stages-1` へ更新した。
対象、幾何、目盛り、針、補間の順に短い観測を記録し、同じ画像の中間表現を用いて値を出す。
モデルがanalogと判定した場合、`reading.analog_trace` に次を保存する。

| フィールド | 内容 |
|---|---|
| target | 対象の計器・尺度を特定できたか |
| geometry | 中心・撮影角度等の観測 |
| scale | 同じ尺度の目盛り文字・単位を確認できたか |
| needle | 細い指示端と反対側の太い部分を区別できたか |
| interpolation | 指示端と目盛りの対応、補間の根拠 |
| center / tip / counterweight | 中心、細い指示端、反対側の点。見えなければnull |
| anchors | 同じ尺度から選んだ2つの近接ラベルの数値・原文・対応する目盛り点 |
| fraction | anchor[0]からanchor[1]への位置。線形補間の根拠がなければnull |
| scale_unit / interpolation_mode / view | 単位、補間方法、正面・斜め・不明の区別 |

各工程のstatusは `visible / uncertain / unreadable / not_assessed`。
`visible` はモデルが根拠を見たとの申告で、正解の保証ではない。
点は、そのReaderに実際に送った候補画像の `(x/width, y/height)`。
元画像・ROI・拡大後・射影補正後の座標を混ぜない。
デジタル・unknownではanalog_trace=null。古い保存結果もnull既定で読み込める。

針の色や暗さを決めつけず、細い指示端を辿るよう明示した。
別尺度の目盛り、文字の小数点、目盛り線と負号、カウンターウェイトを区別する。
非線形尺度には線形補間を強制しない。斜め画像では画像上の角度を物理角度として使わず、
見えている局所目盛りを数える方法等と、その不確実性を記録する。
補正済み画像も送るため、旧プロンプトの「uncorrected image」という固定表現も修正した。

## Pythonによる検査

API呼出しを追加せず、以下の内部矛盾を検出する。

- 針先と中心の一致、指示端とカウンターウェイトが同じ方向に配置されている。
- 単位の食い違い。暗黙にPa↔MPa等を換算しない。
- 数字として解釈できるラベル原文とanchor.valueの桁・符号が食い違う。
- 線形補間を申告したのに、2目盛りまたは比率がない、2目盛り値が同じ。
- `a + fraction * (b-a)` と予測値が食い違う。
  可視の最小目盛りが同一単位で記録されていれば、その半分の丸めを許す。
  未記録なら浮動小数点誤差程度のみ許す。これは正解判定の許容区間ではない。
- 正面かつ幾何を可視と申告した場合だけ、指示端が近接する2目盛りの局所円弧を外れているか。
  正規化座標を実画素の縦横比に戻し、角度の折返しを処理する。
  90度以内の局所円弧に限り、端の座標誤差として2度を許す。斜め・不明では実行しない。

矛盾があれば値を残してvalue_statusをuncertainに下げる。自動で計算値に置き換えない。
対象・目盛り・針・補間の未確定も同様。幾何の不確実性だけで全斜め画像を棄却しない。
新応答でアナログのtraceがnullならanalog_trace_missingとして不確実にする。
従来どおり自動採用は無効。Dでは不確実な候補を採用候補にしない。
誤った針先と誤った目盛りが自己整合するケースは、この検査では検出できない。

## 評価ファイルの見方

通常どおりevaluation.pyを実行すると、既存ファイルに加えて `diagnostics.csv` を出力する。

- `cases.csv`：従来の評価対象を維持し、agent_stop_reasonとrun_reason_codesを追加。
- `diagnostics.csv`：全候補を1行ずつ出力。A/Bはc0、Cはc0/c1、Dは生成した候補すべて。
  Dが棄権しても、採用されなかった値・単位・レシピ・射影補正状態を確認できる。
- `evaluation.json.conditions.<条件>[].candidate_diagnostics`：上記の詳細と全analog_trace。
- `summaries.<条件>.agent_stop_reasons`：D等の停止理由ごとの件数。
- `summaries.<条件>.analog_diagnostics`：アナログ候補の記録有無、申告された問題工程・内部矛盾の件数。
  分母は最終評価ケース数ではなく**全候補数**。一候補が複数項目に計上される。

diagnostics.csvでは、次の順で確認するとよい。

1. `condition / question_id / candidate_id / recipe_id`：どの画像・候補か。
2. `is_evaluation_candidate`：従来のA〜Dの正解率に使った候補か。D棄権時はすべてfalse。
3. `value / unit / value_status / unit_status`：何を出したか。
4. `agent_stop_reason`：候補不一致、予算上限、再撮影等、なぜ停止したか。
5. `target_stage_status`〜`interpolation_stage_status` と対応する `*_stage_evidence`：モデルの工程別観測。
6. `consistency_findings`：Pythonが検出した矛盾。
   例：analog_interpolation_value_conflict、analog_tip_outside_anchor_arc。
7. `analog_trace`：針先・目盛りの座標や原文。実画像との目視照合に使う。

`first_reported_problem_stage` は最初に可視以外と**申告**された工程で、実際の最初の誤りではない。
古い結果をオフライン再評価すると、存在しない中間観測は補完せずtrace_available=falseとする。
今回追加した記録を得るには新プロンプトで再推論する必要がある。
正解率の区間・単位照合、棄権の分母、評価候補の選択ルールは変更しない。

## ローカルでの比較

既存の `outputs/batch20/batch.json` の確認済みROIを再利用する。
benchmarkは新しいexperimentへ出力するため、旧実験への `--resume` は使わない。
プロンプトが変わった実験の混在・再開は既存のプロフィール検証で拒否される。
以下のbenchmarkは実APIを使用する。evaluationはオフラインである。

```bash
set -a
source .env
set +a

experiment_path=$(uv run --locked python src/benchmark.py \
  --prepared-batch outputs/batch20/batch.json \
  --mode inference \
  --provider-config examples/openai_gpt6_luna.json \
  --api-key-env OPENAI_API_KEY \
  --fixed-recipe resize_2x --allow-perspective \
  --database outputs/history/batch20.sqlite3)

evaluation_path=$(uv run --locked --offline python src/evaluation.py \
  --experiment "$experiment_path" \
  --labels data/dev_external/measurebench/labels.jsonl)

open "$(dirname "$evaluation_path")/diagnostics.csv"
open "$(dirname "$evaluation_path")/summary.csv"
```

本変更の検証は人工中間表現・通信モックによるもの。
実画像での改善幅、実APIの出力長・応答時間・費用はローカル再実験で確認する必要がある。
