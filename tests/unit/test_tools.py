import pytest

from src.core.exceptions import ToolExecutionError
from src.tools.calculator import CalculatorTool
from src.tools.datetime import DateTimeTool
from src.tools.filesystem import FilesystemTool
from src.tools.registry import ToolRegistry


class TestCalculatorTool:
    def setup_method(self):
        self.tool = CalculatorTool()

    def test_basic_arithmetic(self):
        assert self.tool.run(expression="2 + 2")["result"] == 4

    def test_order_of_operations(self):
        assert self.tool.run(expression="3 * (4 + 2) / 2")["result"] == 9.0

    def test_rejects_unsafe_expression(self):
        with pytest.raises(ToolExecutionError):
            self.tool.run(expression="__import__('os').system('echo hi')")


class TestDateTimeTool:
    def setup_method(self):
        self.tool = DateTimeTool()

    def test_now_returns_iso_string(self):
        result = self.tool.run(action="now")
        assert "iso" in result

    def test_offset_requires_days(self):
        with pytest.raises(ToolExecutionError):
            self.tool.run(action="offset")


class TestFilesystemTool:
    def setup_method(self, sandbox_tmp_path=None):
        pass  # uses fixture directly in test methods below

    def test_write_and_read(self, sandbox_tmp_path):
        tool = FilesystemTool(sandbox_root=sandbox_tmp_path)
        tool.run(action="write", path="note.txt", content="hello")
        result = tool.run(action="read", path="note.txt")
        assert result["content"] == "hello"

    def test_blocks_path_traversal(self, sandbox_tmp_path):
        tool = FilesystemTool(sandbox_root=sandbox_tmp_path)
        with pytest.raises(ToolExecutionError):
            tool.run(action="read", path="../../etc/passwd")

    def test_list_directory(self, sandbox_tmp_path):
        tool = FilesystemTool(sandbox_root=sandbox_tmp_path)
        tool.run(action="write", path="a.txt", content="a")
        tool.run(action="write", path="b.txt", content="b")
        result = tool.run(action="list", path=".")
        assert set(result["entries"]) == {"a.txt", "b.txt"}


class TestToolRegistry:
    def test_default_tools_registered(self, sandbox_tmp_path):
        registry = ToolRegistry(sandbox_root=sandbox_tmp_path)
        names = {t["name"] for t in registry.manifest()}
        assert "calculator" in names
        assert "datetime" in names
        assert "filesystem" in names

    def test_detect_mentions(self, sandbox_tmp_path):
        registry = ToolRegistry(sandbox_root=sandbox_tmp_path)
        mentions = registry.detect_mentions("use the calculator to compute this")
        assert "calculator" in mentions
