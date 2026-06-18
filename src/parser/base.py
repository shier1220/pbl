"""解析器基类 + ParseResult"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

@dataclass
class ParseResult:
    text: str = ""
    tables: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    chunks: List[dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    @property
    def success(self): return bool(self.text.strip())

class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> ParseResult: ...
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]: ...

    @staticmethod
    def _table_to_markdown(data):
        if not data: return ""
        lines = ["| " + " | ".join(str(c or "") for c in data[0]) + " |",
                  "|" + "|".join("---" for _ in data[0]) + "|"]
        for row in data[1:]:
            padded = row + [""] * (len(data[0]) - len(row))
            lines.append("| " + " | ".join(str(c or "") for c in padded) + " |")
        return "\n".join(lines)

class SafeParser:
    def __init__(self, registry): self.registry = registry
    def parse(self, file_path):
        result = ParseResult(text="")
        try: result = self.registry.get_parser(file_path).parse(file_path)
        except Exception as e:
            result.errors.append(f"Parser error: {e}")
            try:
                with open(file_path, "rb") as f: result.text = f.read().decode("utf-8", errors="replace")
            except Exception as fb: result.errors.append(f"Fallback failed: {fb}")
        return result
