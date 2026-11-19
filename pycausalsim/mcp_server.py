#   Copyright 2026 Brian Curry / Vector1 Research
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Serve a causal simulator over the Model Context Protocol.

    python -m pycausalsim.mcp_server --data visitors.csv --target conversion_rate \\
        --treatments page_load_time,price --confounders traffic_source,device_type

The graph is discovered once at start-up; every tool then answers through
:class:`pycausalsim.ai.ToolBox`, so an assistant connected to this server reads
the same numbers and the same caveats as the Python API. The data never leaves
the machine the server runs on. ``set_graph`` is not served unless
``--allow-graph-changes`` is given, because MCP clients differ in whether they
ask the user before running a tool. Needs the optional ``mcp`` package
(``pip install "pycausalsim[mcp]"``).
"""

from __future__ import annotations

from .ai import SYSTEM_PROMPT, ToolBox

__all__ = ["build_server", "main"]


def build_server(box: ToolBox, name: str = "pycausalsim", allow_graph_changes: bool = False):
    """An MCP server exposing ``box``'s tools. Returns the server object; call ``.run()``."""
    try:
        from mcp.server.mcpserver import MCPServer as Server      # mcp >= 2
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as Server      # mcp 1.x
        except ImportError as e:  # pragma: no cover
            raise ImportError('the MCP server needs the "mcp" package: pip install "pycausalsim[mcp]"') from e
    server = Server(name, instructions=SYSTEM_PROMPT)
    desc = {t["name"]: t["description"] for t in box.specs("mcp")}

    def brief() -> str:
        return box.call_json("brief")

    def data_summary() -> str:
        return box.call_json("data_summary")

    def get_graph() -> str:
        return box.call_json("get_graph")

    def simulate_intervention(variable: str, value: float | None = None, change: float | None = None,
                              method: str = "backdoor", n_bootstrap: int = 50) -> str:
        return box.call_json("simulate_intervention", {"variable": variable, "value": value, "change": change,
                                                       "method": method, "n_bootstrap": n_bootstrap})

    def naive_effect(variable: str, value: float | None = None, change: float | None = None) -> str:
        return box.call_json("naive_effect", {"variable": variable, "value": value, "change": change})

    def rank_drivers(n_bootstrap: int = 20) -> str:
        return box.call_json("rank_drivers", {"n_bootstrap": n_bootstrap})

    def validate(variable: str, n_bootstrap: int = 20) -> str:
        return box.call_json("validate", {"variable": variable, "n_bootstrap": n_bootstrap})

    def optimize_policy(objective: str = "maximize", constraints: dict[str, list[float]] | None = None,
                        grid_points: int = 5) -> str:
        return box.call_json("optimize_policy", {"objective": objective, "constraints": constraints or {},
                                                 "grid_points": grid_points})

    def set_graph(graph: dict[str, list[str]]) -> str:
        return box.call_json("set_graph", {"graph": graph})

    tools = [brief, data_summary, get_graph, simulate_intervention, naive_effect, rank_drivers, validate,
             optimize_policy] + ([set_graph] if allow_graph_changes else [])
    for fn in tools:
        server.add_tool(fn, name=fn.__name__, description=desc[fn.__name__])
    return server


def main(argv: list[str] | None = None) -> None:
    import argparse

    from ._planner import cli_toolbox

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    ap.add_argument("--allow-graph-changes", action="store_true")
    a, rest = ap.parse_known_args(argv)
    build_server(cli_toolbox(rest), allow_graph_changes=a.allow_graph_changes).run(a.transport)


if __name__ == "__main__":  # pragma: no cover
    main()
