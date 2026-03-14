from rag_pipeline import RAGPipeline


if __name__ == "__main__":
    rag = RAGPipeline()
    print("RAG 问答已启动，输入 q 退出。")
    while True:
        q = input("\n问题 > ").strip()
        if q.lower() in {"q", "quit", "exit"}:
            print("已退出。")
            break
        answer = rag.answer(q)
        print(f"\n回答:\n{answer}")
