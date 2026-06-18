"""CSV 解析器"""
import csv, logging
from typing import List
from src.parser.base import BaseParser, ParseResult
logger = logging.getLogger("course_assistant.parser.csv")
MAX_SAMPLE = 100

class CSVParser(BaseParser):
    @property
    def supported_extensions(self) -> List[str]: return [".csv", ".CSV"]

    def parse(self, file_path):
        result = ParseResult()
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                rows = list(csv.reader(f))
            if not rows: result.errors.append("empty CSV"); return result
            header, data = rows[0], rows[1:]
            result.metadata = {"columns": header, "num_rows": len(data)}
            sample = data[:MAX_SAMPLE]
            lines = [f"CSV: {len(header)} 列, {len(data)} 行", f"列名: {', '.join(header)}", "数据预览:"]
            for i, row in enumerate(sample[:5]): lines.append(f"  行{i+1}: {', '.join(row)}")
            if sample:
                md = self._table_to_markdown([header] + sample)
                result.tables.append(md); result.text = "\n".join(lines) + "\n\n" + md
            else: result.text = "\n".join(lines)
        except Exception as e: result.errors.append(f"CSV error: {e}")
        return result
