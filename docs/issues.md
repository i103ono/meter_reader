# meter_reader Issue Roadmap

## Issue 1: 画像読み込み処理を共通化する
目的: 今後の前処理・OCR・メーター読み取りで使い回すため、画像読み込み処理を共通化する。

完了条件:
- src/image_io.py が作成される
- load_image(path) で画像を読み込める
- 読み込み失敗時に分かりやすい例外を出す
- src/list_images.py が必要に応じて image_io.py を利用する
- 既存の list_images.py の動作が壊れていない

## Issue 2: 前処理スクリプトを作成する
目的: OCRや針検出に使いやすい画像を作るため、前処理スクリプトを作成する。

完了条件:
- src/preprocess.py が作成される
- 画像1枚を入力として受け取れる
- グレースケール画像を outputs/ に保存できる
- 二値化画像を outputs/ に保存できる
- READMEに実行例が追加される

## Issue 3: 前処理パラメータを調整可能にする
目的: 画像ごとの明るさや条件差に対応するため、前処理パラメータをCLI引数で調整可能にする。

完了条件:
- threshold値を指定できる
- resize倍率を指定できる
- debugモードで中間画像を保存できる
- 既存のpreprocess.pyの基本動作が維持される

## Issue 4: OCRライブラリを試験導入する
目的: デジタルメーターの数字・単位読み取りのため、OCRライブラリを試験導入する。

完了条件:
- src/ocr_test.py が作成される
- 画像1枚をOCRにかけられる
- 認識結果をターミナルに表示できる
- 失敗時も落ちずにエラーを表示する
- READMEに実行例が追加される

## Issue 5: OCR結果から数値と単位を抽出する
目的: OCRの生テキストから、数値と単位を構造化して取り出す。

完了条件:
- src/parse_reading.py が作成される
- 小数を含む数値を抽出できる
- 単位を抽出できる
- 最低限の誤認識補正ができる
- 単体で動作確認できる

## Issue 6: デジタルメーター読み取りを統合する
目的: 画像読み込み、前処理、OCR、数値・単位抽出を組み合わせて、デジタルメーターを読み取る。

完了条件:
- src/read_digital_meter.py が作成される
- 入力画像1枚から raw_text, value, unit を表示できる
- confidence が取れる場合は表示する
- 読み取り失敗時に分かりやすいエラーを出す

## Issue 7: アナログメーターのROI切り出しを実装する
目的: アナログメーターの文字盤部分を切り出して後続処理をしやすくする。

完了条件:
- src/detect_meter_region.py が作成される
- ROIをCLI引数で指定できる
- 切り出し画像を outputs/ に保存できる
- READMEに実行例が追加される

## Issue 8: 円形メーターを検出する
目的: アナログメーターの中心と半径を検出する。

完了条件:
- src/detect_dial.py が作成される
- 円検出で center_x, center_y, radius を出力できる
- 検出結果を描画したdebug画像を outputs/ に保存できる

## Issue 9: 針を検出して角度を算出する
目的: アナログメーターの針を検出し、角度を算出する。

完了条件:
- src/detect_needle.py が作成される
- 針候補線を検出できる
- 針角度を degree で出力できる
- debug画像を outputs/ に保存できる

## Issue 10: 角度から値へ変換する
目的: 針角度をメーターの数値に変換する。

完了条件:
- src/analog_calibration.py が作成される
- min_value, max_value, min_angle, max_angle から値を計算できる
- CLIまたは関数として動作確認できる

## Issue 11: アナログメーター読み取りを統合する
目的: 円検出、針検出、角度変換を組み合わせてアナログメーターを読み取る。

完了条件:
- src/read_analog_meter.py が作成される
- 画像1枚から value と unit を表示できる
- debug画像を保存できる
- パラメータをCLI引数で指定できる

## Issue 12: デジタル・アナログ統合CLIを作る
目的: 1つのCLIから digital / analog を指定して読み取れるようにする。

完了条件:
- src/read_meter.py が作成される
- --type digital / --type analog に対応する
- 出力形式を統一する
- 既存の個別スクリプトの動作を壊さない

## Issue 13: 複数画像の一括処理を作る
目的: data/images/ 内の画像を一括処理し、結果をCSVに保存する。

完了条件:
- src/batch_read.py が作成される
- data/images/ を順番に処理できる
- outputs/results.csv に保存できる
- 失敗時もファイル名とエラーを記録できる

## Issue 14: 評価用CSVと評価スクリプトを作る
目的: 読み取り結果の精度を確認する。

完了条件:
- data/labels.csv の形式をREADMEに記載する
- src/evaluate.py が作成される
- outputs/results.csv と labels.csv を比較できる
- 数値誤差、単位一致率、成功率を出力できる

## Issue 15: READMEを整備する
目的: 環境構築から実行までの手順を分かりやすくまとめる。

完了条件:
- 環境構築手順がある
- 画像の置き場所が説明されている
- 各スクリプトの実行例がある
- 出力ファイルの説明がある
