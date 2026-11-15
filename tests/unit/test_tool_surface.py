"""Tool list and server instructions for per-request-credential deployments."""

import pytest


@pytest.fixture
def server_module(monkeypatch):
    from postgres_mcp import server

    tools = dict(server.mcp._tool_manager._tools)
    yield server
    server.mcp._tool_manager._tools.clear()
    server.mcp._tool_manager._tools.update(tools)


def test_shared_pool_tools_are_hidden(server_module):
    removed = server_module.hide_shared_pool_tools()
    left = set(server_module.mcp._tool_manager._tools)
    assert set(removed) == set(server_module.SHARED_POOL_TOOLS)
    assert not left & set(server_module.SHARED_POOL_TOOLS)
    assert {"list_objects", "get_object_details"} <= left


def test_hiding_twice_is_harmless(server_module):
    server_module.hide_shared_pool_tools()
    assert server_module.hide_shared_pool_tools() == []


def test_every_hidden_tool_really_needs_the_shared_pool(server_module):
    """Guard against hiding a tool that works per request, or missing a new one."""
    import inspect

    for name, tool in server_module.mcp._tool_manager._tools.items():
        src = inspect.getsource(tool.fn)
        uses_pool = "get_sql_driver()" in src
        assert uses_pool == (name in server_module.SHARED_POOL_TOOLS), name


def test_instructions_file(tmp_path):
    from postgres_mcp.server import load_instructions

    assert load_instructions(None) is None
    f = tmp_path / "instructions.md"
    f.write_text("  Antworte auf Deutsch.\n")
    assert load_instructions(str(f)) == "Antworte auf Deutsch."
    f.write_text("   \n")
    with pytest.raises(ValueError, match="empty"):
        load_instructions(str(f))
