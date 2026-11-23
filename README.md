# kenkyu

日本語の対話生成と、発話の自己開示判定を試す研究・学習用リポジトリです。対話用スクリプトは `dialogues/`、入門用スクリプトは `intro/` にあり、応答方針は `prompts/` に置いています。分類モデルの `my_custom_bert/` はローカルで用意するファイルで、Gitの追跡対象ではありません。

## 対話プログラム

| ファイル | 内容 | 主な参照先・実行上の注意 |
| --- | --- | --- |
| [`chat_keicyo_revised.py`](dialogues/chat_keicyo_revised.py) | 傾聴を重視する対話の修正版。直近の発話をBERTで自己開示判定し、その結果と会話履歴をOpenAI APIへ渡します。 | `prompts/keicyo_binary.md`、ローカルで用意する `my_custom_bert/`。移動後の参照パス調整が必要です（下記参照）。 |
| [`chat_keicyo.py`](dialogues/chat_keicyo.py) | 傾聴を重視する元の実験用スクリプト。BERTの判定結果を応答プロンプトに追加します。 | `prompts/keicyo.md`、`my_custom_bert/`。クラス0と1の解釈が確認用コードと一致せず、5段階の指示に対して判定結果は2クラスです。 |
| [`chat_kyokan.py`](dialogues/chat_kyokan.py) | 共感を重視する対話。BERTによる自己開示の有無も応答に反映します。 | `prompts/kyokan.md`、`my_custom_bert/`。起動時にこれらを読み込みます。 |
| [`chat_normal.py`](dialogues/chat_normal.py) | 通常の対話を試すスクリプト。 | `prompts/chatprompt.md`。起動時にGit管理外の `outputs/` のJSONを読みます。文字コード指定が `uft-8` になっているため、修正が必要です。 |
| [`elyza_chat.py`](dialogues/elyza_chat.py) | ELYZAのローカルLLMを使う対話の実験用スクリプト。 | `llama_cpp` と別途用意するモデルが必要です。データ読み込みの呼び出しなどが未完成で、現状のままでは対話を実行できません。 |

ファイル名は `chat_keicyo.py` です（`chat_kecyo.py` ではありません）。`chat_keicyo_revised.py` は元のスクリプトを置き換えず、別ファイルとして保存しています。

## ディレクトリ構成

Git追跡中のファイルを現在の配置に合わせて示しています。移動先がまだ未追跡のファイルも含め、移動前の存在しないパスと `.gitignore` で除外されたファイル・ディレクトリは含めていません。

```text
kenkyu/
├── .gitignore
├── README.md
├── requirements.txt
├── dialogues/
│   ├── chat_keicyo.py
│   ├── chat_keicyo_revised.py
│   ├── chat_kyokan.py
│   ├── chat_normal.py
│   ├── elyza_chat.py
│   └── exp/
│       └── jsontest.py
├── intro/
│   ├── ex4_3.py
│   ├── ex5.py
│   ├── example.py
│   ├── fin.py
│   ├── intro.py
│   └── test.py
├── label/
│   ├── Llama_label.py
│   ├── Llama_label_tf.py
│   ├── elyza_label.py
│   ├── json_add.py
│   ├── json_csv.py
│   ├── label.py
│   └── label101-200.py
├── practice/
│   ├── ch07/
│   │   ├── ex71.py
│   │   └── ex7_2.py
│   ├── ch8/
│   │   └── ch8.py
│   └── test.py
├── prompts/
│   ├── chatprompt.md
│   ├── keicyo.md
│   ├── keicyo_binary.md
│   ├── kyokan.md
│   ├── prompt.md
│   ├── seed_plan_prompt.txt
│   └── self-disclosure.md
└── test/
    ├── bertcheck.py
    ├── bertkyu.py
    ├── elyza_test.py
    ├── fscore.py
    ├── gpt_test.py
    └── labeltest.py
```

`my_custom_bert/` は `.gitignore` で除外されています。傾聴・共感チャットの実行時にはローカルに配置してください。`outputs/` も追跡対象ではなく、一部の実験用スクリプトはそこに生成したJSONファイルがあることを前提にしています。

## 自己開示ラベル付け（`elyza_label.py`）

[`label/elyza_label.py`](label/elyza_label.py) は、日本語訳したESConvの各発話に、量子化版 Llama-3-ELYZA-JP-8B（`llama-cpp-python`）で自己開示のラベルを付けるスクリプトです。

### 判定する項目

まず自己開示の有無（`self_disclosure`）を判定し、続けて次の4属性を判定します。各発話には判定理由（`reason`）も付きます。

| キー | 内容 | 値 |
| --- | --- | --- |
| `actual_damage` | 実害の有無 | `"0"` 無し / `"1"` 有り |
| `introspection` | 内面性の有無 | `"0"` 表面的 / `"1"` 内面的 |
| `negative_aspects` | 否定的な側面 | `"0"` 無し / `"1"` 有り |
| `psychological_resistance` | 開示への心理的抵抗 | `"0"` 無し / `"1"` 有り |

自己開示が無い発話は、4属性をすべて `"0"` にするようプロンプトで指示しています。

### 必要なファイル

| パス | 内容 | Git管理 |
| --- | --- | --- |
| `prompts/self-disclosure.md` | 判定用プロンプト。`{{GATE}}` と `{{DEFINITIONS}}` に定義が埋め込まれます。 | 追跡対象 |
| `definitions.yaml` | 自己開示（`gate`）と各属性（`attributes`）の定義・値・例文。 | `.gitignore` で除外。ローカルに配置してください |
| `corpus/esconv-ja/ESconvja.json` | 日本語訳ESConv。各対話に `dialogue_id` と `dialog`、各発話に `turn_num`・`speaker`・`content` が必要です。 | 除外 |

判定する属性の種類と値は `definitions.yaml` から読み込み、出力のJSON Schemaも同じファイルから作ります。属性を追加・削除するときは `definitions.yaml` だけを編集すればよく、コードとプロンプトの変更は不要です。

### 実行方法

```bash
python label/elyza_label.py                       # 既定のコーパスを使う
python label/elyza_label.py --corpus path/to.json # コーパスを指定する
```

現在は動作確認のため、先頭の1対話（`data[:1]`）だけを処理します。結果は `outputs/elyza_label_output_test.json` に次の形式で保存されます。

```json
[
    {
        "dialogue_id": 1,
        "labels": [
            {"turn_num": 1, "self_disclosure": "0", "actual_damage": "0", "introspection": "0", "negative_aspects": "0", "psychological_resistance": "0", "reason": "..."}
        ]
    }
]
```

### 環境変数

| 変数 | 既定値 | 内容 |
| --- | --- | --- |
| `LLAMA_MODEL_PATH` | なし | GGUFモデルのパス。未指定ならHugging Faceのキャッシュを探し、無ければダウンロードします |
| `LLAMA_N_CTX` | `4096` | コンテキスト長 |
| `LLAMA_N_BATCH` | `512` | バッチサイズ |
| `LLAMA_N_TOKENS` | `2048` | 1回の出力トークン数の上限 |
| `LLAMA_CHUNK_SIZE` | `8` | 1回のモデル呼び出しで判定する発話数 |

対話は最長120発話あり、1回で渡すとコンテキスト長を超えるため、`LLAMA_CHUNK_SIZE` 発話ずつに分けて判定します。分けたまとまり同士の文脈はモデルに渡していません。「コンテキストが足りません」や「出力がトークン上限で途切れました」というエラーが出た場合は、`LLAMA_CHUNK_SIZE` を小さくするか `LLAMA_N_CTX` を大きくしてください。

### 変更履歴

- 2026-11-23
  - グループゼミでの議論を受け、属性「開示内容の深さ（`depth`）」を廃止しました。判定する属性は5つから4つになりました。
  - `{{GATE}}` と `{{DEFINITIONS}}` に `definitions.yaml` の内容を埋め込む処理を実装しました。
  - 出力のJSON Schemaを `definitions.yaml` から作るようにしました（`required` に誤って `"string"` と書かれていた不具合も解消）。
  - 長い対話を `LLAMA_CHUNK_SIZE` 発話ずつに分けて判定し、出力トークンの上限を入力の長さから計算するようにしました。
  - コーパスの参照先を `corpus/esconv-ja/ESconvja.json` に変更し、`dialog` と `dialogue_id` を読むよう修正しました。
  - 保存形式を、対話ごとに `dialogue_id` と `labels` を持つ形に変更しました。
  - プロンプト内の自己開示を5段階とする記述を削除し、属性の数を書かない表現に変更しました。

`label/Llama_label_tf.py` は以前のスクリプトで、`depth` を含む属性をコード内に直接書いているため、この変更には対応していません。

## 傾聴チャットの起動

リポジトリ直下で、使用する仮想環境を有効にしてから依存ライブラリをインストールします。`requirements.txt` には対話・ラベル付け・評価用の依存ライブラリをまとめており、PyTorchはCPU版を指定しています。

```bash
python -m pip install -r requirements.txt
```

OpenAI APIキーを環境変数 `OPENAI_API_KEY` に設定します。`.env` を使う場合の書式は次のとおりです。修正版は現在 `dialogues/.env` を明示的に読み込むため、リポジトリ直下の `.env` を使うには読み込みパスの調整が必要です。\
※キーをGitに登録しないでください。

```dotenv
OPENAI_API_KEY=your_api_key_here
```

修正版の現在の `BASE_DIR` は `dialogues/` を指しています。このため、プロンプトは `dialogues/prompts/keicyo_binary.md`、モデルは `dialogues/my_custom_bert/` を探します。リポジトリ直下の `prompts/` と `my_custom_bert/` を使うには、コード内の参照先をリポジトリ直下に合わせる必要があります。カレントディレクトリの変更だけでは解消しません。

参照先を調整した後、リポジトリ直下で次のコマンドを実行します。

```bash
python dialogues/chat_keicyo_revised.py
```

`User:` に日本語で入力すると返答が表示されます。`exit` で終了します。修正版はスクリプトの配置場所を基準にプロンプトとBERTモデルを探します。元の `chat_keicyo.py` と `chat_kyokan.py` は相対パスを使うため、リポジトリ直下から実行してください。

共感対話を試す場合は、同じ環境で次を実行します。

```bash
python dialogues/chat_kyokan.py
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

ファイル末尾の `if __name__ == "__main__":` は関数ではなく、`python dialogues/chat_keicyo_revised.py` で直接実行したときに `chatbot()` を起動するための条件です。

## ローカルLLM（Ollama）

今後のローカルLLM利用に向けて、`requirements.txt` にPythonクライアントの `ollama` を含めています。既存のELYZA用スクリプトは `llama-cpp-python` を使用しており、Ollamaとの連携はまだ実装していません。

Ollamaを利用するには、Pythonライブラリに加えてOllama本体を別途インストール・起動し、使用するモデルを `ollama pull <モデル名>` で取得します。詳しくは[Ollama Pythonライブラリの公式手順](https://github.com/ollama/ollama-python#prerequisites)を参照してください。

## 開発環境とデータセット

元の開発環境は Python 3.13.5、NumPy 2.4.6、Ubuntu 24.04.1 LTS です。

モデルの学習と評価には、[日本語日常対話コーパス（Japanese-Daily-Dialogue）](https://github.com/jqk09a/japanese-daily-dialogue) を利用しています。データセットはライセンスの都合でこのリポジトリには含めていません。必要な場合は公式リポジトリを参照してください。
