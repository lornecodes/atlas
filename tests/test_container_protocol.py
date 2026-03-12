"""Tests for container stdin/stdout JSON protocol types."""

import json

import pytest

from atlas.security.protocol import ContainerMessage, ContainerResponse


class TestContainerMessage:
    def test_default_empty(self):
        msg = ContainerMessage()
        assert msg.input == {}
        assert msg.context == {}

    def test_to_json(self):
        msg = ContainerMessage(input={"key": "val"}, context={"job_id": "j-1"})
        parsed = json.loads(msg.to_json())
        assert parsed == {"input": {"key": "val"}, "context": {"job_id": "j-1"}}

    def test_to_json_empty(self):
        msg = ContainerMessage()
        parsed = json.loads(msg.to_json())
        assert parsed == {"input": {}, "context": {}}

    def test_with_data(self):
        msg = ContainerMessage(input={"a": 1, "b": [2, 3]})
        parsed = json.loads(msg.to_json())
        assert parsed["input"]["b"] == [2, 3]


class TestContainerResponse:
    def test_success(self):
        resp = ContainerResponse.from_json('{"output": {"result": 42}}')
        assert resp.success is True
        assert resp.output == {"result": 42}
        assert resp.error == ""

    def test_error(self):
        resp = ContainerResponse.from_json('{"error": "something broke"}')
        assert resp.success is False
        assert resp.error == "something broke"

    def test_empty_output(self):
        resp = ContainerResponse.from_json('{"output": {}}')
        assert resp.success is True
        assert resp.output == {}

    def test_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            ContainerResponse.from_json("not json")
