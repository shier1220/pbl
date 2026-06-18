"""解析器注册表"""
import os, logging
logger = logging.getLogger("course_assistant.parser")

class ParserRegistry:
    def __init__(self): self._parsers = {}
    def register(self, parser):
        for ext in parser.supported_extensions: self._parsers[ext.lower()] = parser
    def get_parser(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self._parsers: raise ValueError(f"不支持: {ext}")
        return self._parsers[ext]
    def supports(self, file_path): return os.path.splitext(file_path)[1].lower() in self._parsers
    @property
    def supported_extensions(self): return sorted(self._parsers.keys())
    @property
    def parser_count(self): return len({type(p) for p in self._parsers.values()})
