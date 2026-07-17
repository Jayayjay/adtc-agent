from src.tools.registry import ToolRegistry


def test_full_tool_execution_cycle(tmp_path):
    registry = ToolRegistry(sandbox_root=tmp_path)

    calc_result = registry.execute("calculator", expression="5 * (3 + 1)")
    assert calc_result["result"] == 20

    registry.execute("filesystem", action="write", path="out.txt", content="result: 20")
    read_result = registry.execute("filesystem", action="read", path="out.txt")
    assert "20" in read_result["content"]
