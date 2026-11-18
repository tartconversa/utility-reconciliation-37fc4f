"""资料文件和通用检查器的已有测试。"""
from pathlib import Path
import tempfile
import unittest
from sourcepack.catalog import inspect, read_table

class SourceTests(unittest.TestCase):
    def test_pack(self):
        result = inspect(Path(__file__).resolve().parents[1])
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["tables"])
        self.assertTrue(all(t["rows"] > 0 for t in result["tables"]))

    def check_bad(self, data, message):
        table = {"columns": [{"name": "id", "type": "string"}, {"name": "amount", "type": "number"}], "key": ["id"]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.csv"
            path.write_text(data, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, message):
                read_table(path, table)

    def test_duplicate(self):
        self.check_bad("id,amount\na,1\na,2\n", "记录键")

    def test_type(self):
        self.check_bad("id,amount\na,unknown\n", "类型")

    def test_columns(self):
        self.check_bad("id,wrong\na,1\n", "字段")

    def test_nonfinite(self):
        self.check_bad("id,amount\na,nan\n", "类型")

if __name__ == "__main__":
    unittest.main()
