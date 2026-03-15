# 基于 DeepSeek + Chroma 的书籍 RAG（支持增量更新）

这个项目是一个可落地的书籍问答 RAG 模板，包含你要求的增强能力：

- ✅ DeepSeek API 生成回答
- ✅ Chroma 本地向量数据库（免费、离线可用、无需云账号）
- ✅ 增量更新（新增/改动文件只重建对应分块）
- ✅ BM25 + 向量混合检索
- ✅ Rerank 重排模型
- ✅ 自动去重，减少重复上下文，查询更快
- ✅ Web UI 聊天界面（Streamlit）
- ✅ Docker 沙箱环境（便于移植部署）

## 1. 架构

- Embedding：`BAAI/bge-small-zh-v1.5`
- Vector DB：Chroma（`data/chroma` 持久化）
- Hybrid Retrieval：
  1) Chroma 向量召回
  2) 本地 BM25 召回
  3) 分数融合（hybrid_score）
  4) 文本去重
  5) Rerank（`BAAI/bge-reranker-base`）
- 生成：DeepSeek `deepseek-chat`

## 2. 快速开始（本机 Python）

### 2.1 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2.2 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 填入：

- `DEEPSEEK_API_KEY`

### 2.3 放入书籍

把书籍文件放到 `data/books/`，支持：`.pdf` / `.txt` / `.md`

### 2.4 执行增量入库

```bash
python src/ingest.py
```

### 2.5 命令行问答

```bash
python src/chat.py
```

### 2.6 Web UI 聊天

```bash
streamlit run src/web_ui.py
```

浏览器打开后即可问答，侧边栏支持手动触发增量入库。

## 3. Docker 沙箱（推荐）

### 3.1 构建并启动

```bash
docker compose up --build -d
```

### 3.2 执行一次增量入库

```bash
docker compose exec app python src/ingest.py
```

### 3.3 访问 Web UI

打开：`http://localhost:8501`

### 3.4 停止

```bash
docker compose down
```

> 向量库数据会落盘到 `./data/chroma`，容器重建后仍可复用。

## 4. 目录

```text
.
├── data/books/
├── data/chroma/
├── docs/
│   └── 使用说明.md
├── src/
│   ├── rag_pipeline.py
│   ├── ingest.py
│   ├── chat.py
│   └── web_ui.py
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── requirements.txt
```
