from llama_cpp import Llama
import time


def main():
    llm = Llama.from_pretrained(
        repo_id="elyza/Llama-3-ELYZA-JP-8B-GGUF",
        filename="Llama-3-ELYZA-JP-8B-q4_k_m.gguf",
        n_ctx=2048,
        chat_format="llama-3",
    )

    start_time = time.perf_counter()

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "あなたは誠実で優秀な日本人のアシスタントです。特に指示がなければ日本語で回答してください",
            },
            {
                "role": "user",
                "content": "大谷翔平について教えて",
            },
        ],
        max_tokens=1024,
    )
    fin_time = time.perf_counter() - start_time
    print(response["choices"][0]["message"]["content"])
    print(f"応答時間: {fin_time: .2f}s")


if __name__ == "__main__":
    main()
