from chestct_agent.llm import QwenClient


def test_parse_json_content_accepts_plain_object():
    assert QwenClient._parse_json_content('{"answer": 1}') == {"answer": 1}


def test_parse_json_content_accepts_fenced_or_prefixed_object():
    content = 'Result:\n```json\n{"answer": 2}\n```\nDone.'

    assert QwenClient._parse_json_content(content) == {"answer": 2}
