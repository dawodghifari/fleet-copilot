"""The agent's tool loop, driven by a scripted fake Anthropic client."""

import json
import sys
import types
from types import SimpleNamespace as NS

import pytest
from test_agent_tools import make_tools

from fleet_copilot import agent


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


@pytest.fixture
def fake_anthropic(monkeypatch):
    """Install a fake `anthropic` module and return a function that scripts it."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    holder = {}

    def script(*responses):
        holder["messages"] = FakeMessages(responses)
        module = types.ModuleType("anthropic")
        module.Anthropic = lambda api_key: NS(messages=holder["messages"])
        monkeypatch.setitem(sys.modules, "anthropic", module)
        return holder["messages"]

    return script


def tool_call(name, args, call_id="t1"):
    return NS(stop_reason="tool_use",
              content=[NS(type="tool_use", name=name, input=args, id=call_id)])


def final(text):
    return NS(stop_reason="end_turn", content=[NS(type="text", text=text)])


def test_reuses_the_tools_it_is_given(fake_anthropic, monkeypatch):
    def no_reload():
        raise AssertionError("run_agent rebuilt FleetTools instead of reusing it")

    monkeypatch.setattr(agent, "FleetTools", no_reload)
    fake_anthropic(final("No vehicles needed."))
    assert agent.run_agent("hello", tools=make_tools())["answer"] == "No vehicles needed."


def test_model_comes_from_the_environment(fake_anthropic, monkeypatch):
    monkeypatch.setenv("FLEET_AGENT_MODEL", "some-other-model")
    messages = fake_anthropic(final("ok"))
    agent.run_agent("hello", tools=make_tools())
    assert messages.calls[0]["model"] == "some-other-model"


def test_default_model_without_the_variable(fake_anthropic, monkeypatch):
    monkeypatch.delenv("FLEET_AGENT_MODEL", raising=False)
    messages = fake_anthropic(final("ok"))
    agent.run_agent("hello", tools=make_tools())
    assert messages.calls[0]["model"] == agent.DEFAULT_AGENT_MODEL


def test_tool_results_go_back_to_the_model(fake_anthropic):
    messages = fake_anthropic(tool_call("predict_vehicle", {"vehicle_id": 0}),
                              final("Vehicle 0 is flagged."))
    result = agent.run_agent("Is vehicle 0 at risk?", tools=make_tools())
    assert result["answer"] == "Vehicle 0 is flagged."
    assert result["trace"][0]["output"]["flagged"] is True
    # The conversation list is shared and keeps growing, so find the tool result
    # by shape rather than position.
    tool_result = next(
        block
        for msg in messages.calls[1]["messages"]
        if msg["role"] == "user" and isinstance(msg["content"], list)
        for block in msg["content"]
        if block.get("type") == "tool_result"
    )
    assert tool_result["tool_use_id"] == "t1"
    assert json.loads(tool_result["content"])["probability_aps_failure"] == 0.9


def test_a_failing_tool_is_reported_not_raised(fake_anthropic):
    fake_anthropic(tool_call("predict_vehicle", {"vehicle_id": 99}), final("No such vehicle."))
    result = agent.run_agent("vehicle 99?", tools=make_tools())
    assert "error" in result["trace"][0]["output"]
    assert result["answer"] == "No such vehicle."
