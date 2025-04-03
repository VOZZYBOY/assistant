# app_standalone.py
# Простой RAG-ассистент с FastAPI в одном файле

import os
import shutil
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import List, Dict

# --- Основные зависимости ---
# FastAPI & Uvicorn
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel

# LangChain компоненты
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import GigaChat
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# --- Конфигурация ---
# Определяем пути относительно этого файла app_standalone.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR # В этом простом варианте корень = папка со скриптом

# Папка с исходными документами (ПОЛОЖИТЕ ПАПКУ matrixcrmdoca РЯДОМ С ЭТИМ ФАЙЛОМ)
DOCS_PATH = os.path.join(PROJECT_ROOT, "matrixcrmdoca")
# Папка для хранения векторной базы данных Chroma
VECTOR_STORE_PATH = os.path.join(PROJECT_ROOT, "chroma_db_store_standalone")

# Модель для создания эмбеддингов
EMBEDDING_MODEL_NAME = 'ai-forever/sbert_large_nlu_ru'

# Параметры разделения текста на части (чанки)
CHUNK_SIZE = 700
CHUNK_OVERLAP = 150
SPLITTER_TYPE = 'recursive'


GIGACHAT_CREDENTIALS="OTkyYTgyNGYtMjRlNC00MWYyLTg3M2UtYWRkYWVhM2QxNTM1OmZkZmFmNTFlLTFlYjUtNDlkMC04NGI5LTc5Zjg3OTBlZjQyZg=="

# --- Настройка логирования ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Уменьшаем шум от библиотек
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("unstructured").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
# ---------------------------

# --- Глобальные переменные для инициализированных компонентов ---
LLM = None
RETRIEVER = None
INITIALIZATION_COMPLETE = False
INITIALIZATION_ERROR = None



def get_device():
    try:
        import torch
        if torch.cuda.is_available(): return "cuda"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built(): return "mps"
    except ImportError: pass
    except Exception as e: logger.warning(f"Ошибка проверки Torch: {e}.")
    return "cpu"

def get_embeddings(model_name: str):
    device = get_device()
    logger.info(f"Инициализация эмбеддингов: '{model_name}' на '{device}'...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs={'device': device}, encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"Эмбеддинги '{model_name}' загружены.")
        return embeddings
    except Exception as e:
        logger.error(f"Сбой загрузки эмбеддингов: {e}", exc_info=True); return None

def load_local_documents(docs_path: str) -> list:
    logger.info(f"Загрузка документов из: {docs_path}")
    if not os.path.exists(docs_path) or not os.path.isdir(docs_path):
         logger.error(f"Папка с документами не найдена: {docs_path}"); return []
    loader_map = {".md": UnstructuredMarkdownLoader, ".txt": TextLoader, ".pdf": PyPDFLoader}
    all_docs = []
    start_time = time.time()
    try:
        for ext, loader_cls in loader_map.items():
            loader = DirectoryLoader(
                docs_path, glob=f"**/*{ext}", loader_cls=loader_cls, use_multithreading=True,
                show_progress=False, recursive=True, silent_errors=True,
                loader_kwargs={'encoding': 'utf-8'} if loader_cls == TextLoader else {}
            )
            docs = loader.load()
            if docs: all_docs.extend(docs)
        loaded_count = len(all_docs)
        all_docs = [doc for doc in all_docs if doc and hasattr(doc, 'page_content') and doc.page_content.strip()]
        filtered_count = len(all_docs)
        duration = time.time() - start_time
        logger.info(f"Загружено {filtered_count} док-ов из ~{loaded_count} за {duration:.2f} сек.")
        return all_docs
    except Exception as e:
        logger.error(f"Ошибка загрузки документов: {e}", exc_info=True); return []

def split_documents(docs: list, chunk_size: int, chunk_overlap: int) -> list:
    if not docs: return []
    logger.info(f"Разделение {len(docs)} док-ов (size={chunk_size}, overlap={chunk_overlap})...")
    start_time = time.time()
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len,
            is_separator_regex=False, separators=["\n\n", "\n", ". ", " ", ""] # Добавил точку с пробелом
        )
        splits = text_splitter.split_documents(docs)
        duration = time.time() - start_time
        logger.info(f"Разделено на {len(splits)} чанков за {duration:.2f} сек.")
        splits = [split for split in splits if split.page_content.strip()]
        return splits
    except Exception as e:
        logger.error(f"Ошибка разделения: {e}", exc_info=True); return []

def build_or_load_chroma(embedding_model, force_rebuild: bool = False):
    logger.info(f"Проверка/создание ChromaDB @ {VECTOR_STORE_PATH} (force={force_rebuild})...")
    if vector_store_exists := (os.path.exists(VECTOR_STORE_PATH) and os.path.isdir(VECTOR_STORE_PATH)):
        if not force_rebuild:
            logger.info("Попытка загрузки существующей ChromaDB...")
            try:
                vectorstore = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embedding_model)
                if vectorstore._collection.count() > 0:
                    logger.info("ChromaDB успешно загружена."); return vectorstore
                else:
                    logger.warning("Загруженная ChromaDB пуста. Пересоздание...")
                    shutil.rmtree(VECTOR_STORE_PATH, ignore_errors=True); vector_store_exists = False
            except Exception as e:
                logger.warning(f"Ошибка загрузки ChromaDB: {e}. Пересоздание...", exc_info=False)
                shutil.rmtree(VECTOR_STORE_PATH, ignore_errors=True); vector_store_exists = False
        elif vector_store_exists: 
             logger.info("Принудительное удаление старой ChromaDB...")
             shutil.rmtree(VECTOR_STORE_PATH, ignore_errors=True)

    logger.info("Создание новой ChromaDB...")
    all_documents = load_local_documents(DOCS_PATH)
    if not all_documents: logger.error("Документы не загружены!"); return None
    doc_splits = split_documents(all_documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    if not doc_splits: logger.error("Документы не разделены!"); return None
    logger.info(f"Добавление {len(doc_splits)} чанков в ChromaDB...")
    try:
        start_time = time.time()
        vectorstore = Chroma.from_documents(
            documents=doc_splits, embedding=embedding_model, persist_directory=VECTOR_STORE_PATH
        )
        duration = time.time() - start_time
        logger.info(f"ChromaDB создана за {duration:.2f} сек.")
        return vectorstore
    except Exception as e:
        logger.error(f"Ошибка создания ChromaDB: {e}", exc_info=True)
        if os.path.exists(VECTOR_STORE_PATH): shutil.rmtree(VECTOR_STORE_PATH, ignore_errors=True)
        return None

def get_retriever(vectorstore: Chroma, k_results: int = 5):
    if not vectorstore: logger.error("Векторная база не передана!"); return None
    logger.info("Настройка ретривера...")
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": k_results})
    logger.info(f"Chroma ретривер готов (k={k_results}).")
    bm25_retriever = None
    try:
        all_docs_for_bm25 = load_local_documents(DOCS_PATH)
        if all_docs_for_bm25:
            bm25_retriever = BM25Retriever.from_documents(documents=all_docs_for_bm25, k=k_results)
            logger.info(f"BM25 ретривер готов (k={k_results}).")
    except ImportError: logger.error("BM25 недоступен: pip install rank_bm25")
    except Exception as e_bm25: logger.error(f"Ошибка создания BM25: {e_bm25}")

    if bm25_retriever:
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, chroma_retriever], weights=[0.4, 0.6])
        logger.info("Используется гибридный ретривер (Ensemble).")
        return ensemble_retriever
    else:
        logger.warning("Используется только Chroma ретривер.")
        return chroma_retriever


def initialize_components():
    global LLM, RETRIEVER, INITIALIZATION_COMPLETE, INITIALIZATION_ERROR
    logger.info("--- Начало инициализации компонентов ---")
    start_time = time.time()
    try:
        
        if not GIGACHAT_CREDENTIALS or GIGACHAT_CREDENTIALS == "ЗАМЕНИТЕ_НА_ВАШИ_CREDENTIALS_ЕСЛИ_НЕТ_ENV":
            raise ValueError("Не установлены GIGACHAT_CREDENTIALS.")
        LLM = GigaChat(credentials=GIGACHAT_CREDENTIALS, verify_ssl_certs=False, scope='GIGACHAT_API_PERS')
        logger.info("LLM GigaChat инициализирован.")

        
        embeddings = get_embeddings(EMBEDDING_MODEL_NAME)
        if not embeddings: raise ValueError("Не удалось инициализировать эмбеддинги.")

        
        vectorstore = build_or_load_chroma(embedding_model=embeddings, force_rebuild=False)
        if not vectorstore: raise ValueError("Не удалось создать/загрузить ChromaDB.")

        
        RETRIEVER = get_retriever(vectorstore=vectorstore, k_results=3) 
        if not RETRIEVER: raise ValueError("Не удалось создать ретривер.")

        INITIALIZATION_COMPLETE = True
        duration = time.time() - start_time
        logger.info(f"--- Инициализация УСПЕШНО завершена за {duration:.2f} сек ---")

    except Exception as e:
        logger.error(f"!!! КРИТИЧЕСКАЯ ОШИБКА ИНИЦИАЛИЗАЦИИ: {e} !!!", exc_info=True)
        INITIALIZATION_ERROR = e
        INITIALIZATION_COMPLETE = False
        LLM = None; RETRIEVER = None 


chat_histories: Dict[str, List[BaseMessage]] = defaultdict(list)

# --- FastAPI приложение ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_components() 
    yield
    logger.info("Остановка сервера...")
    chat_histories.clear()

app = FastAPI(title="Standalone Simple RAG Assistant", lifespan=lifespan)


@app.middleware("http")
async def check_initialization_middleware(request: Request, call_next):
    if not INITIALIZATION_COMPLETE:
        error_detail = f"Сервис не готов: {INITIALIZATION_ERROR or 'инициализация...'}"
        logger.error(f"Запрос к {request.url.path} отклонен: {error_detail}")
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": error_detail})
    if not LLM or not RETRIEVER: # Доп. проверка
         error_detail = "Критические компоненты не инициализированы."
         logger.error(f"Запрос к {request.url.path} отклонен: {error_detail}")
         from fastapi.responses import JSONResponse
         return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": error_detail})
    response = await call_next(request)
    return response


class QueryRequest(BaseModel):
    user_id: str
    message: str

class QueryResponse(BaseModel):
    answer: str


async def get_rag_response(user_query: str, history: List[BaseMessage]) -> str:
    logger.info(f"RAG: Запрос '{user_query}'")
    context_str = "Поиск по документации не дал результатов или ретривер недоступен."
    try:
        start_retrieve = time.time()
        results = await RETRIEVER.ainvoke(user_query)
        retrieve_duration = time.time() - start_retrieve
        if results:
            logger.info(f"RAG: Найдено {len(results)} док-ов за {retrieve_duration:.2f} сек.")
            context_parts = []
            for i, doc in enumerate(results): 
                source = doc.metadata.get('source', 'N/A')
                relative_source = os.path.relpath(source, start=PROJECT_ROOT) if source != 'N/A' else 'N/A'
                context_parts.append(f"Источник {i+1}: {relative_source}\nФрагмент: {doc.page_content}")
            context_str = "\n\n---\n\n".join(context_parts)
        else:
            logger.info("RAG: Документы не найдены.")
            context_str = f"В документации не найдено информации по запросу: '{user_query}'."
    except Exception as e_retrieve:
        logger.error(f"RAG: Ошибка поиска: {e_retrieve}", exc_info=True)
        context_str = f"Произошла ошибка при поиске: {e_retrieve}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Ты ИИ-ассистент по документации. Отвечай на вопрос пользователя СТРОГО на основе контекста. Если контекст нерелевантен или пуст, скажи, что не можешь ответить по документации. Будь краток."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "Контекст:\n---\n{context}\n---\n\nВопрос: {query}")
    ])
    chain = prompt | LLM
    ai_response_content = "Ошибка генерации ответа."
    try:
        start_llm = time.time()
        response = await chain.ainvoke({
            "history": history, "context": context_str, "query": user_query
        })
        llm_duration = time.time() - start_llm
        logger.info(f"RAG: Ответ LLM получен за {llm_duration:.2f} сек.")
        if isinstance(response, BaseMessage): ai_response_content = response.content
        elif isinstance(response, str): ai_response_content = response
        else: ai_response_content = str(response); logger.warning(f"Неожиданный тип ответа LLM: {type(response)}")
    except Exception as e_llm:
        logger.error(f"RAG: Ошибка генерации LLM: {e_llm}", exc_info=True)
        ai_response_content = f"Ошибка при обращении к языковой модели: {e_llm}"
    logger.info(f"RAG: Итоговый ответ: {ai_response_content[:150]}...")
    return ai_response_content


@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    user_id = request.user_id
    user_message = request.message.strip()
    if not user_message: raise HTTPException(status_code=400, detail="Сообщение пустое.")
    logger.info(f"API: Запрос от user='{user_id}': '{user_message[:100]}...'")

    history = chat_histories.get(user_id, [])
    final_answer = await get_rag_response(user_message, history)

  
    current_turn = [HumanMessage(content=user_message), AIMessage(content=final_answer)]
    chat_histories[user_id] = history + current_turn
    logger.debug(f"API: История для '{user_id}' обновлена, содержит {len(chat_histories[user_id])} сообщений.")

    return QueryResponse(answer=final_answer)


@app.get("/health")
async def health_check():
    if INITIALIZATION_COMPLETE and LLM and RETRIEVER:
        return {"status": "OK", "message": "Assistant Ready."}
    else:
        return {"status": "INITIALIZING" if not INITIALIZATION_ERROR else "ERROR",
                "message": f"{INITIALIZATION_ERROR or 'Инициализация...'}"}


if __name__ == "__main__":
    logger.info("--- Запуск Standalone Simple RAG Assistant ---")
    
    uvicorn.run("test_version:app", host="127.0.0.1", port=8001, reload=False, log_level="info")

