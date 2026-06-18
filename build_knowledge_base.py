"""
知识库构建脚本
读取 aigc_knowledge_base.txt → 分块 → 向量化 → 写入 ChromaDB

用法:
    python build_knowledge_base.py                          # 默认读取 aigc_knowledge_base.txt
    python build_knowledge_base.py --file knowledge.txt     # 指定其他文件
    python build_knowledge_base.py --clear                  # 只清空知识库
"""

import argparse
import sys
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from chromadb import PersistentClient

from src.embedding import InstructorEmbedding
from src.config import MODEL_PATH, CHROMA_PATH, COLLECTION_NAME, COLLECTION_METADATA, CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS


# ── 本地配置 ──
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = str(BASE_DIR / "aigc_knowledge_base.txt")


def clear_collection():
    """清空 ChromaDB 中的 course_knowledge 集合"""
    try:
        client = PersistentClient(path=CHROMA_PATH)
        client.delete_collection(COLLECTION_NAME)
        print("已清空 collection: %s" % COLLECTION_NAME)
    except Exception:
        print("collection 不存在或已清空，跳过")


def build_knowledge_base(source_file: str, clear: bool = True):
    # 1. 读取源文件
    src_path = Path(source_file)
    if not src_path.exists():
        print("错误: 文件不存在: %s" % source_file)
        sys.exit(1)

    with open(src_path, "r", encoding="utf-8") as f:
        text = f.read()
    print("源文件: %s (%d 字符)" % (src_path.name, len(text)))

    # 2. 分块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
    )
    chunks = splitter.split_text(text)
    print("分块完成: %d 个片段 (chunk_size=%d, overlap=%d)" % (
        len(chunks), CHUNK_SIZE, CHUNK_OVERLAP
    ))

    if not chunks:
        print("错误: 分块结果为空")
        sys.exit(1)

    # 3. 加载嵌入模型
    print("加载嵌入模型: %s ..." % MODEL_PATH)
    embedding_fn = InstructorEmbedding(MODEL_PATH)

    # 4. 清空旧数据 + 创建新 collection
    chroma_client = PersistentClient(path=CHROMA_PATH)
    if clear:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    col = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata=COLLECTION_METADATA,
    )

    vectorstore = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        persist_directory=CHROMA_PATH,
    )

    # 5. 分批写入（每批 10 条，避免超时）
    metas = [{"source": src_path.name, "chunk": i} for i in range(len(chunks))]
    batch_size = 10
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_metas = metas[i : i + batch_size]
        vectorstore.add_texts(texts=batch_chunks, metadatas=batch_metas)
        end = min(i + batch_size, len(chunks))
        print("  写入进度: %d/%d" % (end, len(chunks)))

    # 6. 验证
    final_count = col.count()
    print("\n知识库构建完成！")
    print("  存储路径: %s" % CHROMA_PATH)
    print("  集合名:   %s" % COLLECTION_NAME)
    print("  片段数:   %d" % final_count)

    return final_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIGC 课程助手知识库构建脚本")             # 命令行参数解析
    parser.add_argument(                    
        "--file", "-f",
        default=DEFAULT_SOURCE,
        help="知识库源文件路径 (默认: aigc_knowledge_base.txt)",
    )                   # 指定知识库源文件路径
    parser.add_argument(
        "--clear", "-c",
        action="store_true",
        help="只清空知识库，不写入新数据",
    )
    parser.add_argument(
        "--append", "-a",
        action="store_true",
        help="不清空旧数据，追加写入",
    )
    args = parser.parse_args()                  # 解析命令行参数

    if args.clear:
        clear_collection()
    else:
        build_knowledge_base(args.file, clear=not args.append)
