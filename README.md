# kenkyu

日本語の対話生成と、発話の自己開示判定を試す研究・学習用リポジトリです。対話用スクリプトはリポジトリ直下にあり、応答方針は `prompts/` に置いています。分類モデルの `my_custom_bert/` はローカルで用意するファイルで、Gitの追跡対象ではありません。

## 対話プログラム

| ファイル | 内容 | 主な参照先・実行上の注意 |
| --- | --- | --- |
| [`chat_keicyo_revised.py`](chat_keicyo_revised.py) | 傾聴を重視する対話の修正版。直近の発話をBERTで自己開示判定し、その結果と会話履歴をOpenAI APIへ渡します。 | `prompts/keicyo_binary.md`、ローカルで用意する `my_custom_bert/`。最初に試す場合はこちら。 |
| [`chat_keicyo.py`](chat_keicyo.py) | 傾聴を重視する元の実験用スクリプト。BERTの判定結果を応答プロンプトに追加します。 | `prompts/keicyo.md`、`my_custom_bert/`。クラス0と1の解釈が確認用コードと一致せず、5段階の指示に対して判定結果は2クラスです。 |
| [`chat_kyokan.py`](chat_kyokan.py) | 共感を重視する対話。BERTによる自己開示の有無も応答に反映します。 | `prompts/kyokan.md`、`my_custom_bert/`。起動時にこれらを読み込みます。 |
| [`chat_normal.py`](chat_normal.py) | 通常の対話を試すスクリプト。 | `prompts/chatprompt.md`。現在は起動時に `outputs/` のJSONを読みますが、このリポジトリに `outputs/` はなく、文字コード指定にも誤記があります。現状のままでは起動できません。 |
| [`elyza_chat.py`](elyza_chat.py) | ELYZAのローカルLLMを使う対話の実験用スクリプト。 | `llama_cpp` と別途用意するモデルが必要です。データ読み込みの呼び出しなどが未完成で、現状のままでは対話を実行できません。 |

ファイル名は `chat_keicyo.py` です（`chat_kecyo.py` ではありません）。`chat_keicyo_revised.py` は元のスクリプトを置き換えず、別ファイルとして保存しています。

## ディレクトリ構成

以下は `git ls-files` で確認した、Gitの追跡対象だけの構成です。ローカル専用のモデルや `.env` は含めていません。

```text
kenkyu/
├── .gitignore
├── README.md
├── chat_keicyo.py
├── chat_keicyo_revised.py
├── chat_kyokan.py
├── chat_normal.py
├── elyza_chat.py
├── ex4_3.py
├── ex5.py
├── example.py
├── fin.py
├── intro.py
├── jsontest.py
├── test.py
├── prompts/
│   ├── chatprompt.md
│   ├── keicyo.md
│   ├── keicyo_binary.md
│   ├── kyokan.md
│   ├── labelprompt.md
│   ├── prompt.md
│   └── seed_plan_prompt.txt
├── label/
│   ├── elyza_label.py
│   ├── json_add.py
│   ├── json_csv.py
│   ├── label.py
│   └── label101-200.py
├── test/
│   ├── bertcheck.py
│   ├── bertkyu.py
│   ├── elyza_test.py
│   ├── fscore.py
│   ├── gpt_test.py
│   └── labeltest.py
├── ch07/
│   ├── ex71.py
│   └── ex7_2.py
├── ch8/
│   └── ch8.py
└── practice/
    └── test.py
```

`my_custom_bert/` は `.gitignore` で除外されています。傾聴・共感チャットの実行時にはローカルに配置してください。`outputs/` も追跡対象ではなく、一部の実験用スクリプトはそこに生成したJSONファイルがあることを前提にしています。

## 傾聴チャットの起動

Pythonと、少なくとも `openai`、`python-dotenv`、`transformers`、`torch` が必要です。依存パッケージの固定ファイルは現在ありません。必要に応じて利用するPython環境にインストールしてください。

OpenAI APIキーを環境変数 `OPENAI_API_KEY` に設定します。リポジトリ直下の `.env` を使う場合の書式は次のとおりです。キーをGitに登録しないでください。

```dotenv
OPENAI_API_KEY=your_api_key_here
```

修正版は次のコマンドで起動します。

```bash
python chat_keicyo_revised.py
```

`User:` に日本語で入力すると返答が表示されます。`exit` で終了します。修正版はスクリプトの配置場所を基準にプロンプトとBERTモデルを探します。元の `chat_keicyo.py` と `chat_kyokan.py` は相対パスを使うため、リポジトリ直下から実行してください。

共感対話を試す場合は、同じ環境で次を実行します。

```bash
python chat_kyokan.py
```

## `chat_keicyo_revised.py` の関数

### `classify_disclosure(text, tokenizer, model)`

入力した発話 `text` の自己開示を判定する関数です。起動時に読み込んだトークナイザーとBERTモデルを受け取り、次の順で処理します。

1. 発話をBERT用のトークンに変換します。モデルの最大入力長を超える場合は、分類に使う部分だけを切り詰めます。
2. 学習済みモデルで推論し、2クラスの出力を確率に変換します。
3. クラス1の確率が `THRESHOLD`（現在は0.7）以上なら「自己開示有り」、クラス0が0.7以上なら「自己開示無し」、どちらも満たさなければ「自己開示不明」という文字列を返します。

クラス1を「有り」とする対応は既存の確認用コードに合わせたものです。学習時のラベル定義が分かる場合は照合してください。

### `chatbot()`

対話全体を進める関数です。まず `.env` または環境変数からAPIキーを読み、応答用プロンプト、BERTモデル、トークナイザー、OpenAIクライアントを準備します。モデルが2クラスでない場合は会話を開始しません。

会話中は入力を受け取るたびに `classify_disclosure()` を呼び、判定結果をプロンプトへ加えます。過去の会話と今回の入力をAPIへ送り、返答を表示します。返答が得られた場合だけ今回のやり取りを履歴に保存し、履歴は直近10往復に保ちます。APIエラーや空の返答があった場合は、その入力を履歴に追加せず次の入力を待ちます。

空白だけの入力は無視します。`exit`、Ctrl+D、Ctrl+C で会話を終了します。この関数自体は返答文字列を返さず、ターミナルへの入出力を行います。

ファイル末尾の `if __name__ == "__main__":` は関数ではなく、`python chat_keicyo_revised.py` で直接実行したときに `chatbot()` を起動するための条件です。

## 開発環境とデータセット

元の開発環境は Python 3.13.5、NumPy 2.4.6、Ubuntu 24.04.1 LTS です。

モデルの学習と評価には、[日本語日常対話コーパス（Japanese-Daily-Dialogue）](https://github.com/jqk09a/japanese-daily-dialogue) を利用しています。データセットはライセンスの都合でこのリポジトリには含めていません。必要な場合は公式リポジトリを参照してください。
