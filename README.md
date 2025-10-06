# Chrome Whisper Transcriber (.exe版)

## Patch Note  
- ver1 Patched
  - Output folders（Audio files/Markdown）の指定を可能にしました。
  - 録音の再開ができなかった問題を修正し、（録音再開/録音停止）ボタンを追加しました。  
- ver1.1 Patched-
  - 録音した音声ファイルを（保存/削除）選択できる機能を追加しました。
  - 音声ファイル単体を読み込ませ、文字起こし・要約できる機能を追加しました。  
- ver1.2 Patched
  - Output folders/録音対象デバイスの設定を保存するようにしました。  
- ver1.3 Patched
  - Ollamaの使用モデル/推論モードをGUI上から選択できるようにしました。
  - Ollamaサーバーの自動起動機能を追加しました。
  - 利便性向上のため、exeファイル化しました。  
- ver1.3.1 Patched
  - ollamaのモデルダウンロードを容易にするため、ダウンロード用スクリプト（`download_ollama_model.exe`）を実装、同梱しました。
  - exe化した際、faster-whisperがexeファイルに含まれておらず、正常に動作しない問題を修正しました。

## 概要
- Windows向けのスタンドアロンアプリ。ChromeなどのPC音声をループバック録音し、faster-whisperで文字起こし、Ollamaで要約します。
- 配布物は `ChromeWhisperTranscriber.exe`（GUI）と `transcribe_files.exe`（バッチ処理）, `download_ollama_model.exe`の3点です。
- ファイル保存場所などの設定は保持され、`ChromeWhisperTranscriber/settings.json`に保存されます。  
  ただし、無音区切り/録音ファイル保存のチェックボックス設定は保持されません。

## 必要環境
- Ollama CLIのダウンロード、インストールが必要です。[こちら](https://ollama.com/download)からダウンロードしてください。
- Windows 10/11 (64bit)、WASAPI対応のサウンドデバイス。
- GPUを使う場合はCUDA環境をセットアップ。CPUのみでも動作します。

## 起動と操作
1. （初回のみ）`download_ollama_model.exe`を起動し、ollamaをインストールします。
2. `ChromeWhisperTranscriber.exe` を起動し、録音対象デバイスを選択します。前回の選択と出力先は `settings.json` に保存され自動復元されます。
3. `録音開始` で収録。既定で5秒以上の無音を検知すると自動で区切り、Markdown出力と要約を行います。  
    録音したファイルを残したくない場合、`録音した音声ファイルを保存する`チェックを外してください。
4. `区切って書き起こし` は手動区切り、`停止` で録音停止、`録音再開` で最後のデバイス設定を再利用します。
5. `ファイルで要約` を押すと既存の音声ファイルをキューに追加できます。音声保存先・Markdown保存先は画面下部のフォルダ選択から変更可能です。

## 出力
- 初回起動時、書き込み可能な場所に `ChromeWhisperTranscriber\output` フォルダを自動作成し、音声 (`CWT_AUDIO_DIR`) とMarkdown (`CWT_SUMMARY_DIR`) を保存します。
- Markdownには科目・タイトル・要約ポイント・全文が含まれ、`_index.md` に処理履歴が追記されます。
- 録音した音声を残したくない場合は画面下部のチェックを外してください。
