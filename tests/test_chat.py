"""Chat and agent layer: grounded tools, the offline agent, grounding, approval, message formats."""
import asyncio
import json
import warnings

import numpy as np
import pandas as pd
import pytest

from pycausalsim import CausalSimulator
from pycausalsim.ai import ToolBox, brief
from pycausalsim.chat import (TASKS, AnthropicLLM, Chat, OfflineAgent, OpenAILLM, Reply, check_grounding,
                              main, make_llm)


def make_data(n=800, seed=42, load_effect=-3.0):
    """Visitors: high-intent traffic converts better and gets slower pages (the confounder)."""
    rng = np.random.default_rng(seed)
    traffic = rng.choice([0, 1], n, p=[0.6, 0.4])
    load = 3.0 + 0.5 * traffic + rng.exponential(0.5, n)
    price = 20 + rng.normal(0, 5, n)
    device = rng.choice([0, 1, 2], n)
    conv = 15 + load_effect * load + 2.0 * traffic - 0.5 * (price - 20) + rng.normal(0, 2, n)
    return pd.DataFrame({"traffic_source": traffic, "page_load_time": load, "price": price,
                         "device_type": device, "conversion_rate": conv})


def simulator():
    sim = CausalSimulator(make_data(), "conversion_rate", treatment_vars=["page_load_time", "price"],
                          confounders=["traffic_source", "device_type"], random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.discover_graph()
    return sim


@pytest.fixture
def box():
    return ToolBox(simulator())


# ----------------------------------------------------------------- shared core

class FakeBox:
    system_prompt = "fake"
    mutating = frozenset({"change"})

    def __init__(self):
        self.calls = []

    def specs(self, fmt="tool_use"):
        return [{"name": "read", "description": "", "input_schema": {"type": "object"}},
                {"name": "change", "description": "", "input_schema": {"type": "object"}}]

    def call(self, name, args=None):
        self.calls.append(name)
        return {"tool": name, "result": {"value": 4.25, "share": 0.317}, "caveats": []}


class Scripted:
    name, tool_format = "scripted", "tool_use"

    def __init__(self, *outs):
        self.outs, self.seen = list(outs), []

    def complete(self, system, history, tools):
        self.seen.append(list(history))
        return self.outs.pop(0)


def call(name, i=0, **args):
    return {"text": "", "tool_calls": [{"id": f"c{i}", "name": name, "input": args}]}


def test_grounding_traces_numbers_and_flags_invented_ones():
    src = [("t", {"roas": 1.2345, "share": 0.0937, "change": -12.3, "note": "Pareto k 0.41"})]
    g = check_grounding("ROAS 1.23 (about 1.2), 9.4% of sales, cut 12.3, k 0.41.", src)
    assert g["ok"] and g["checked"] == 5
    g = check_grounding("ROAS 1.23, or about 2.47 if doubled.", src)
    assert not g["ok"] and [u["text"] for u in g["unsupported"]] == ["2.47"]
    # labels are not findings: small counts, interval levels, years, dates
    assert check_grounding("The 90% range over 3 channels in 2025-12-29.", []) == \
        {"ok": True, "checked": 0, "supported": [], "unsupported": []}


def test_loop_runs_tools_then_grounds_the_answer():
    box = FakeBox()
    llm = Scripted(call("read"), {"text": "Value 4.25, share 31.7%, and 7.77 made up.", "tool_calls": []})
    r = Chat(box, llm=llm).ask("go")
    assert box.calls == ["read"] and [s["tool"] for s in r.steps] == ["read"]
    assert [u["text"] for u in r.grounding["unsupported"]] == ["7.77"] and r.backend == "scripted"
    assert llm.seen[1][-1]["role"] == "tool"          # the model saw the tool result
    assert isinstance(r, Reply) and json.loads(r.to_json())["text"] == r.text


def test_actions_that_change_the_model_wait_for_approval():
    box = FakeBox()
    chat = Chat(box, llm=Scripted(call("change", x=1), {"text": "done", "tool_calls": []}))
    r = chat.ask("change it")
    assert r.pending == {"tool": "change", "input": {"x": 1}} and box.calls == []
    with pytest.raises(RuntimeError):
        chat.ask("something else")
    r = chat.approve()
    assert box.calls == ["change"] and r.text == "done" and r.pending is None

    box = FakeBox()
    chat = Chat(box, llm=Scripted(call("change"), {"text": "skipped", "tool_calls": []}))
    chat.ask("change it")
    r = chat.decline()
    assert box.calls == [] and "declined" in r.steps[0]["result"]["error"]

    box = FakeBox()
    r = Chat(box, llm=Scripted(call("change"), {"text": "no", "tool_calls": []}),
             confirm=lambda name, args: False).ask("change it")
    assert box.calls == [] and r.pending is None and r.text == "no"


def test_step_cap_stops_a_runaway_agent():
    box = FakeBox()
    llm = Scripted(*[call("read", i) for i in range(5)])
    r = Chat(box, llm=llm, max_steps=3).ask("loop")
    assert len(box.calls) == 3 and "stopped after 3 tool calls" in r.text


def test_message_formats_for_each_api():
    hist = [{"role": "user", "text": "q"},
            {"role": "assistant", "text": "", "tool_calls": [{"id": "a", "name": "read", "input": {}},
                                                             {"id": "b", "name": "read", "input": {"k": 1}}]},
            {"role": "tool", "id": "a", "name": "read", "input": {}, "result": {"x": 1}},
            {"role": "tool", "id": "b", "name": "read", "input": {"k": 1}, "result": {"x": 2}}]
    am = AnthropicLLM.messages(hist)
    assert [m["role"] for m in am] == ["user", "assistant", "user"]
    assert [b["type"] for b in am[2]["content"]] == ["tool_result", "tool_result"]
    om = OpenAILLM.messages("sys", hist)
    assert [m["role"] for m in om] == ["system", "user", "assistant", "tool", "tool"]
    assert json.loads(om[2]["tool_calls"][1]["function"]["arguments"]) == {"k": 1}


def test_backend_specs(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError):
        make_llm("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert make_llm("anthropic:claude-haiku-5-5").model == "claude-haiku-5-5"
    assert make_llm("ollama:llama3.1").base_url.endswith(":11434/v1")
    assert isinstance(make_llm(None, FakeBox()), OfflineAgent)
    with pytest.raises(ValueError):
        make_llm("nope")


def test_stepping_shows_each_tool_call_as_it_happens():
    box = FakeBox()
    chat = Chat(box, llm=Scripted(call("read"), call("change", 1), {"text": "4.25", "tool_calls": []}))
    chat.start("go")
    out = [chat.step(), chat.step(), chat.step(), chat.step()]
    assert [o["type"] for o in out[:3]] == ["plan", "tool", "plan"]
    assert out[1]["tool"] == "read" and out[1]["result"]["result"]["value"] == 4.25
    assert isinstance(out[3], Reply) and out[3].pending["tool"] == "change"
    assert chat.approve(run=False) is None and box.calls == ["read"]
    assert chat.step()["tool"] == "change"
    final = chat.step()
    assert isinstance(final, Reply) and final.text == "4.25" and final.grounding["ok"]


# ---------------------------------------------------------------- grounded tools

def test_toolbox_returns_json_with_caveats(box):
    names = [t["name"] for t in box.specs()]
    assert {"brief", "simulate_intervention", "naive_effect", "validate", "set_graph"} <= set(names)
    assert all(t["type"] == "function" for t in box.specs("function"))
    assert all("inputSchema" in t for t in box.specs("mcp"))
    e = box.call("simulate_intervention", {"variable": "page_load_time", "change": -1.0})
    json.dumps(e)
    r = e["result"]
    assert r["value"] == pytest.approx(r["current_mean"] - 1.0)
    assert r["ci_lower"] <= 3.0 <= r["ci_upper"]                 # true effect of one second faster: +3
    assert any("Backdoor" in c for c in e["caveats"])
    n = box.call("naive_effect", {"variable": "page_load_time", "change": -1.0})["result"]
    assert n["effect"] < r["effect"]                             # confounding hides part of the payoff
    assert "error" in box.call("simulate_intervention", {"variable": "nope", "value": 1})
    assert "error" in box.call("simulate_intervention", {"variable": "conversion_rate", "value": 1})
    assert "error" in box.call("simulate_intervention", {"variable": "price"})
    assert "error" in box.call("nope")
    far = box.call("simulate_intervention", {"variable": "price", "value": 500, "method": "scm", "n_bootstrap": 5})
    assert any("outside its observed range" in c for c in far["caveats"])
    b = brief(box.sim)
    assert b["n_edges"] == len(b["edges"]) and any("observational" in c for c in b["caveats"])
    opt = box.call("optimize_policy", {})["result"]
    assert set(opt["settings"]) == {"page_load_time", "price"}


def test_set_graph_is_marked_as_changing_the_model(box):
    assert box.mutating == {"set_graph"}
    bad = box.call("set_graph", {"graph": {"price": ["conversion_rate"], "conversion_rate": ["price"]}})
    assert "cycle" in bad["error"]
    ok = box.call("set_graph", {"graph": {"conversion_rate": ["page_load_time", "price", "traffic_source"],
                                          "page_load_time": ["traffic_source"]}})
    assert "error" not in ok and box.graph_source == "set"
    assert not any("observational" in c for c in box.call("brief")["caveats"])


# ------------------------------------------------------- pycausalsim's offline agent

QUESTIONS = {
    "What causes conversion?": ["get_graph"],
    "What if we cut page load time by 1 second?": ["simulate_intervention", "naive_effect"],
    "Set price to 25": ["simulate_intervention", "naive_effect"],
    "What drives conversion most?": ["rank_drivers"],
    "What are the best settings?": ["optimize_policy"],
    TASKS["overview"]: ["brief", "data_summary"],
    TASKS["effect_audit"]: ["simulate_intervention", "naive_effect", "validate"],
    TASKS["act_first"]: ["rank_drivers", "validate", "optimize_policy"],
}


@pytest.mark.parametrize("question", list(QUESTIONS))
def test_offline_agent_plans_and_every_number_is_grounded(box, question):
    r = Chat(box).ask(question)
    assert [s["tool"] for s in r.steps] == QUESTIONS[question]
    assert r.grounding["ok"], r.grounding["unsupported"]
    assert r.text and r.backend == "offline"
    assert "\u2014" not in r.text and "\u2013" not in r.text


def test_offline_agent_reads_the_question(box):
    chat = Chat(box)
    assert chat.ask("Speed up page load to 2.5 seconds").steps[0]["input"] == {"variable": "page_load_time",
                                                                               "value": 2.5}
    assert chat.ask("cut page load by 1").steps[0]["input"]["change"] == -1.0
    inp = chat.ask("raise price by 10%").steps[0]["input"]
    assert inp["variable"] == "price" and inp["change"] == pytest.approx(0.1 * box.sim.data.price.mean(), rel=1e-4)
    assert chat.ask("tell me a joke").text.startswith("I am the built-in offline agent")


def test_offline_agent_asks_before_changing_the_graph(box):
    chat = Chat(box)
    r = chat.ask("Assume device type causes page load time")
    assert r.pending["tool"] == "set_graph" and "device_type" not in box.sim.graph["page_load_time"]
    r = chat.approve()
    assert "device_type" in box.sim.graph["page_load_time"] and r.grounding["ok"]
    r = chat.ask("Device type does not cause page load time")
    r = chat.decline()
    assert "device_type" in box.sim.graph["page_load_time"] and "left the graph" in r.text


def test_terminal_chat(tmp_path, capsys):
    path = tmp_path / "visitors.csv"
    make_data().to_csv(path, index=False)
    main(["--task", "overview", "--data", str(path), "--target", "conversion_rate",
          "--treatments", "page_load_time,price", "--confounders", "traffic_source,device_type"])
    out = capsys.readouterr().out
    assert "[tool] brief" in out and "numbers traced to a tool result" in out


def test_mcp_server_serves_the_tools(box):
    pytest.importorskip("mcp")
    from pycausalsim.mcp_server import build_server
    names = {t.name for t in asyncio.run(build_server(box).list_tools())}
    assert {"brief", "simulate_intervention", "validate"} <= names and "set_graph" not in names
    names = {t.name for t in asyncio.run(build_server(box, allow_graph_changes=True).list_tools())}
    assert "set_graph" in names
