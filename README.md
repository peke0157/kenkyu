# kenkyu

日本語の対話生成と、発話の自己開示判定を試す研究・学習用リポジトリです。対話用スクリプトはリポジトリ直下にあり、応答方針は `prompts/`、ローカルの分類モデルは `my_custom_bert/` に置いています。

## 対話プログラム

| ファイル | 内容 | 主な参照先・実行上の注意 |
| --- | --- | --- |
| [`chat_keicyo_revised.py`](chat_keicyo_revised.py) | 傾聴を重視する対話の修正版。直近の発話をBERTで自己開示判定し、その結果と会話履歴をOpenAI APIへ渡します。 | `prompts/keicyo_binary.md`、`my_custom_bert/`。最初に試す場合はこちら。詳しくは[解説書](chat_keicyo_revised_guide.md)。 |
| [`chat_keicyo.py`](chat_keicyo.py) | 傾聴を重視する元の実験用スクリプト。BERTの判定結果を応答プロンプトに追加します。 | `prompts/keicyo.md`、`my_custom_bert/`。クラス0と1の解釈が確認用コードと一致せず、5段階の指示に対して判定結果は2クラスです。 |
| [`chat_kyokan.py`](chat_kyokan.py) | 共感を重視する対話。BERTによる自己開示の有無も応答に反映します。 | `prompts/kyokan.md`、`my_custom_bert/`。起動時にこれらを読み込みます。 |
| [`chat_normal.py`](chat_normal.py) | 通常の対話を試すスクリプト。 | `prompts/chatprompt.md`。現在は起動時に `outputs/` のJSONを読みますが、このリポジトリに `outputs/` はなく、文字コード指定にも誤記があります。現状のままでは起動できません。 |
| [`elyza_chat.py`](elyza_chat.py) | ELYZAのローカルLLMを使う対話の実験用スクリプト。 | `llama_cpp` と別途用意するモデルが必要です。データ読み込みの呼び出しなどが未完成で、現状のままでは対話を実行できません。 |

ファイル名は `chat_keicyo.py` です（`chat_kecyo.py` ではありません）。`chat_keicyo_revised.py` は元のスクリプトを置き換えず、別ファイルとして保存しています。

## ディレクトリ構成

```text
kenkyu/
├── README.md
├── chat_keicyo.py                 # 傾聴対話の元スクリプト
├── chat_keicyo_revised.py         # 傾聴対話の修正版
├── chat_keicyo_revised_guide.md   # 修正版の詳しい解説
├── chat_kyokan.py                 # 共感対話
├── chat_normal.py                 # 通常対話の実験用コード
├── elyza_chat.py                  # ELYZAを使う実験用コード
├── prompts/                       # 対話・ラベル付け用プロンプト
│   ├── keicyo.md
│   ├── keicyo_binary.md
│   ├── kyokan.md
│   ├── chatprompt.md
│   ├── labelprompt.md
│   └── その他のプロンプト
├── my_custom_bert/                # 学習済み自己開示分類モデルとトークナイザー
├── label/                         # 発話ラベルの生成・変換用スクリプト
├── test/                          # 分類・ラベル付けなどの確認用スクリプト
├── ch07/                        # 学習・試作用のコード
├── ch8/                         # 学習・試作用のコード
├── practice/                    # 学習・試作用のコード
└── その他のルート直下のPythonファイル  # APIやデータ処理の試作
```

`outputs/` は現在のディレクトリ構成にはありません。一部の実験用スクリプトは、そこに生成したJSONファイルがあることを前提にしています。

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

## 開発環境とデータセット

元の開発環境は Python 3.13.5、NumPy 2.4.6、Ubuntu 24.04.1 LTS です。

モデルの学習と評価には、[日本語日常対話コーパス（Japanese-Daily-Dialogue）](https://github.com/jqk09a/japanese-daily-dialogue) を利用しています。データセットはライセンスの都合でこのリポジトリには含めていません。必要な場合は公式リポジトリを参照してください。
