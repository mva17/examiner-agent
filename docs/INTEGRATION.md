# Как перенести Экзаменатора в командный репозиторий

Командный репозиторий: https://github.com/Kolomkik123/mephilaborant (сейчас там Расчётчик Ильи, коммит «расчетчик» от 6.11). Илья просил: «агентов загружайте потом туда, каждый свою часть».

## Что там сейчас (на 9.11)

```
src/app/
  agents/base.py, calculator.py     ← Расчётчик (Илья)
  api/main.py                       ← FastAPI Расчётчика: /health, /calculate, /index/rebuild
  ui/app.py                         ← Streamlit Расчётчика
  llm/, tools/, data/, config.py, security.py, state.py
tests/ (unittest), eval/, run.py, start.ps1, requirements.txt, .env.example
```
Запуск через `run.py`, `PYTHONPATH=src`, Python 3.12. LangGraph в этом репозитории пока не используется.

## Предложение: отдельная папка, чужие файлы почти не трогаем

Работаем в ветке `feature/examiner` и открываем pull request (так договорились в плане команды).

| Что | Куда | Чужие файлы |
|---|---|---|
| пакет `examiner/` | `src/app/agents/examiner/` (все импорты внутри пакета относительные, переносится как есть) | — |
| маршруты API | `src/app/api/examiner_routes.py` = наш `api/main.py` | **+2 строки** в `src/app/api/main.py` Ильи (см. ниже) |
| чат | `src/app/ui/pages/2_Экзаменатор.py` = наш `ui/app.py` (Streamlit сам добавит страницу в меню, файл Ильи не меняется) | — |
| ДЕМО-тексты | `data/manuals_demo/` (`MANUALS_DIR=data/manuals_demo`) | — |
| тесты | `tests/examiner/` (pytest; заменить `from examiner` → `from app.agents.examiner`) | — |
| бенчмарк | `eval/examiner/` | — |
| зависимости | дописать в `requirements.txt`: `langgraph`, `langchain-core`, `langchain-openai`, `pydantic`, `python-dotenv` | +5 строк |
| настройки | дописать в `.env.example` блок `# Экзаменатор` (`LLM_BASE_URL`, `MAX_QUESTIONS`, `MAX_LLM_CALLS`, `SEARCH_BACKEND`, …) | +10 строк |

Изменение в `src/app/api/main.py` Ильи:
```python
from app.api.examiner_routes import create_router as examiner_router
app.include_router(examiner_router(), prefix="/examiner")
```
Маршруты Экзаменатора станут `/examiner/chat`, `/examiner/exam/start`, … и не пересекутся с `/health` и `/calculate` Ильи (это проверяет тест `test_router_mounts_into_team_app_with_prefix`). Для страницы чата: `API_URL=http://127.0.0.1:8000/examiner`.

**Что согласовать с Ильёй:** `config.py` у него свой (переменные `LLM_PROVIDER=mock|openrouter|local`, `OPENROUTER_API_KEY`). У Экзаменатора `LLM_PROVIDER=fake|openai` и `LLM_API_KEY`. Пока оставить обе схемы (не мешают друг другу), а когда Павел сделает общий `llm.py`, перейти на него: Экзаменатору нужен любой объект с `invoke(messages) -> AIMessage`, это одна строка в `ExaminerDeps`.

## Подключение поиска Оскара (ChromaDB)

1. `pip install chromadb`, в `.env`: `SEARCH_BACKEND=chroma`, `CHROMA_DIR=<папка индекса>`, `CHROMA_COLLECTION=<имя коллекции>`.
2. **Сверить метаданные кусков** с Оскаром. Экзаменатору нужны:
   - `work` — код или номер работы (одинаковый у всех кусков одной методички, по нему фильтр `where={"work": ...}`);
   - `page` — номер страницы (int);
   - `section` — заголовок раздела; раздел с контрольными вопросами должен начинаться с «Контрольные вопросы» (или текст куска начинается с этих слов), вопросы — нумерованным списком `1. …`;
   - желательно `title` — название работы.

   Если поля называются иначе (`lab`, `page_number`, `heading`), они уже поддержаны; другие названия дописываются в `ChromaManualSearch.FIELD_MAP`.
3. **Эмбеддинги запроса.** Если Оскар строит индекс через Qwen3-Embedding по API (как на практике 30.09), запрос надо векторизовать той же моделью:
   ```python
   from langchain_openai import OpenAIEmbeddings
   emb = OpenAIEmbeddings(model="Qwen/Qwen3-Embedding-0.6B", base_url=..., api_key=..., check_embedding_ctx_length=False)
   search = ChromaManualSearch.from_settings(settings, embed_query=emb.embed_query)
   ```
4. Когда у Оскара будет функция `search_manuals(query, work=None, k=5) -> list[Source]`, можно вместо Chroma-адаптера сделать обёртку из 10 строк с тем же интерфейсом `ManualSearch`. Граф Экзаменатора менять не нужно.

## Подключение Проверяющего (Ибрагим)

Передать объект с методом `verify(question, run) -> VerifierVerdict(ok, reason)`:
```python
deps = ExaminerDeps.from_settings()
deps.verifier = IbragimVerifierAdapter()   # вызывает Проверяющего для вопроса и эталона
```

## Подключение к Диспетчеру (Павел)

Вариант 1 (проще): Диспетчер при `intent == "exam"` вызывает `POST /examiner/chat` с теми же `user_id`, `chat_id`.
Вариант 2: граф Экзаменатора как подграф. `build_graph(deps)` возвращает скомпилированный граф. Формат `thread_id = "<user_id>:<chat_id>"` совпадает с планом Павла.

## Порядок действий (после подтверждения Вячеслава)

1. Илья добавляет `mva17` в коллабораторы, Вячеслав принимает приглашение.
2. `git clone https://github.com/Kolomkik123/mephilaborant && cd mephilaborant && git checkout -b feature/examiner`.
3. Скопировать файлы по таблице выше, поправить импорты в тестах, дописать `requirements.txt` и `.env.example`.
4. `python run.py test` (тесты Ильи) и `python -m pytest tests/examiner` — оба зелёные.
5. Push ветки и pull request в `main` с описанием; Илья смотрит и мёржит.
