from pathlib import Path

from rich import print

from rag_pipeline import RAGPipeline


if __name__ == "__main__":
    data_dir = Path("data/books")
    if not data_dir.exists():
        raise FileNotFoundError("未找到 data/books 目录，请先放入书籍文件。")

    rag = RAGPipeline()
    for result in rag.ingest_dir(data_dir):
        print(f"[green]{result}[/green]")
