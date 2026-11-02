"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

--chat を指定すると、同じチャットで会話履歴を保持して質問できる。
    uv run python -m app.ask --chat "LoRAとは何ですか？"
ツールや講座固有の知識は持たない。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)

    print_usage(response)

    return response.text


def print_usage(response) -> None:
    """APIが使用量を返さない場合は、0と区別して表示する。"""
    usage = response.usage_metadata
    prompt = getattr(usage, "prompt_token_count", None)
    output = getattr(usage, "candidates_token_count", None)
    print(f"[tokens] 入力: {prompt if prompt is not None else '不明'} / "
          f"出力: {output if output is not None else '不明'}")


def chat_loop(first_question: str) -> None:
    """プロセス内で同じChatを再利用し、SDKに履歴の管理を任せる。"""
    with genai.Client(api_key=get_api_key()) as client:
        chat = client.chats.create(model=MODEL)
        question = first_question
        while True:
            try:
                response = chat.send_message(question)
            except errors.APIError as e:
                print(api_error_message(e.code, e.message))
            else:
                print_usage(response)
                print(response.text or "（テキストの回答がありません）")
            while True:
                try:
                    question = input("\n次の質問（/exit で終了）> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n会話を終了します。")
                    return
                if question == "/exit":
                    return
                if question:
                    break


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--chat", action="store_true", help="履歴を保って続けて質問する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    try:
        if args.chat:
            chat_loop(args.question)
        else:
            print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
