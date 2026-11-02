"""実際のSDKのChatで履歴を確認する。ネットワーク呼び出しのみ置き換える。"""
from unittest.mock import patch
from google import genai
from google.genai import types
import pytest
from app import ask


def response(text="LoRAの説明"):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(
            role="model", parts=[types.Part(text=text)]))],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=20))


def test_chat_sends_previous_user_and_model_turns(monkeypatch, capsys):
    client = genai.Client(api_key="test-not-a-real-key")
    chat_manager = client.chats
    original_create = type(chat_manager).create
    with patch.object(client.models, "generate_content", return_value=response()) as generate:
        with patch.object(type(chat_manager), "create", autospec=True, side_effect=original_create) as create:
            monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: client)
            monkeypatch.setattr(ask, "get_api_key", lambda: "test-not-a-real-key")
            inputs = iter(["", "もっと短く", "/exit"])
            monkeypatch.setattr("builtins.input", lambda _: next(inputs))
            ask.chat_loop("LoRAとは？")
            assert create.call_count == 1
            assert generate.call_count == 2
            history = generate.call_args.kwargs["contents"]
            assert [item.role for item in history] == ["user", "model", "user"]
            assert [item.parts[0].text for item in history] == [
                "LoRAとは？", "LoRAの説明", "もっと短く"]
    assert "入力: 10 / 出力: 20" in capsys.readouterr().out


@pytest.mark.parametrize("ending", [EOFError, KeyboardInterrupt])
def test_chat_exits_on_end_of_input(monkeypatch, ending):
    from unittest.mock import MagicMock
    client = MagicMock()
    client.__enter__.return_value = client
    client.chats.create.return_value.send_message.return_value = response()
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-not-a-real-key")
    def stop(_):
        raise ending
    monkeypatch.setattr("builtins.input", stop)
    ask.chat_loop("質問")
    client.__exit__.assert_called_once()


def test_missing_usage_is_not_reported_as_zero(capsys):
    ask.print_usage(types.GenerateContentResponse())
    assert "入力: 不明 / 出力: 不明" in capsys.readouterr().out
