from pathlib import Path

import streamlit as st

from rag_pipeline import RAGPipeline


@st.cache_resource
def get_rag() -> RAGPipeline:
    return RAGPipeline()


st.set_page_config(page_title="Books RAG Chat", page_icon="📚", layout="wide")
st.title("📚 书籍 RAG 聊天助手")
st.caption("Hybrid Retrieval: 向量 + BM25 + Rerank")

if "history" not in st.session_state:
    st.session_state.history = []

rag = None
init_error = None
try:
    rag = get_rag()
except Exception as exc:  # noqa: BLE001
    init_error = str(exc)

if init_error:
    st.warning(f"RAG 初始化失败：{init_error}。请先配置 .env 后再使用问答与入库功能。")

with st.sidebar:
    st.subheader("知识库操作")
    if st.button("重新执行增量入库 (data/books)", disabled=rag is None):
        results = rag.ingest_dir(Path("data/books"))
        for result in results:
            st.write(result)

question = st.chat_input("请输入你的问题…", disabled=rag is None)
if question and rag is not None:
    st.session_state.history.append(("user", question))
    with st.spinner("检索 + 生成中..."):
        answer = rag.answer(question)
        hits = rag.retrieve_hybrid(question)
    st.session_state.history.append(("assistant", answer))

    with st.expander("查看检索到的上下文", expanded=False):
        for i, hit in enumerate(hits, 1):
            st.markdown(
                f"**{i}. {hit['source']}#{hit['chunk_idx']}**  \\\n"
                f"hybrid={hit.get('hybrid_score', 0):.4f}, rerank={hit.get('rerank_score', 0):.4f}"
            )
            st.write(hit["text"])

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.write(content)
