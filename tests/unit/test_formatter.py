from unittest.mock import MagicMock

from src.llm.formatter import format_tool_result, format_tool_execution_result


def test_format_tool_result_calls_chat_with_context():
    mock_model = MagicMock()
    mock_model.chat.return_value = "Here's your plan."

    response = format_tool_result(
        mock_model, "schedule my tasks", {"plan": ["step1", "step2"]}
    )

    assert response == "Here's your plan."
    mock_model.chat.assert_called_once()
    call_args = mock_model.chat.call_args[0][0]
    assert "schedule my tasks" in call_args[0]["content"]
    assert "step1" in call_args[0]["content"]


def test_format_tool_execution_result_summarizes():
    mock_model = MagicMock()
    mock_model.chat.return_value = "The calculation returned 42."

    response = format_tool_execution_result(mock_model, "calculator", {"result": 42})

    assert response == "The calculation returned 42."
    mock_model.chat.assert_called_once()
