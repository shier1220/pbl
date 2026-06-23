"""共享 fixtures"""
import sys, os, pytest, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def tmp_db_path():
    """临时 SQLite 数据库路径"""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)
