"""Чат Экзаменатора на Streamlit. Ходит в FastAPI (API_URL), ключей модели не знает.

Запуск:  streamlit run ui/app.py   (сначала: uvicorn api.main:app)
"""

from __future__ import annotations

import os
import uuid

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")

NODE_NAMES = {
    "guard_input": "проверка запроса",
    "extract_topic": "выделение темы",
    "retrieve": "поиск по методичкам",
    "check_relevance": "проверка релевантности найденного",
    "rewrite_query": "переформулировка запроса",
    "plan_manual": "вопросы по методичке",
    "plan_general": "генерация и проверка вопросов",
    "pick": "выбор следующего вопроса",
    "grade": "оценка ответа",
    "followup": "уточняющий вопрос",
    "summarize": "итог",
    "refuse": "отказ",
}
VERDICT_STYLE = {"верно": ("✅", "green"), "частично": ("🟡", "orange"), "неверно": ("❌", "red")}

st.set_page_config(page_title="Экзаменатор — допуск к лабам", page_icon="🎓")


def api(method: str, path: str, **kwargs) -> dict | None:
    try:
        response = httpx.request(method, f"{API_URL}{path}", timeout=120, **kwargs)
    except httpx.HTTPError:
        st.error(f"API недоступен по адресу {API_URL}. Запустите: `uvicorn api.main:app`")
        return None
    if response.status_code >= 400:
        detail = response.json().get("detail", response.text) if response.headers.get("content-type", "").startswith("application/json") else response.text
        st.error(f"Ошибка {response.status_code}: {detail}")
        return None
    return response.json()


def new_chat() -> None:
    st.session_state.chat_id = uuid.uuid4().hex[:12]
    st.session_state.messages = []


if "chat_id" not in st.session_state:
    new_chat()

# --- Боковая панель ---------------------------------------------------------
with st.sidebar:
    st.header("🎓 Экзаменатор")
    st.caption("Тренажёр допуска к лабораторным по физике. Команда 43, НИЯУ МИФИ.")
    user_id = st.text_input("Ваш id", value=st.session_state.get("user_id", "student1"), max_chars=64)
    st.session_state.user_id = user_id
    if st.button("🆕 Новая сессия", use_container_width=True):
        new_chat()
        st.rerun()
    health = api("GET", "/health")
    if health:
        st.caption(f"Модель: {health['model']} · поиск: {health['search_backend']}")
        if health["llm_provider"] == "fake":
            st.info("Работает демо-модель без ключа: оценки упрощённые. Для настоящей модели впишите ключ в .env.")
    memory = api("GET", f"/memory/{user_id}") if user_id else None
    if memory and memory["weak_topics"]:
        st.subheader("Слабые темы")
        for w in memory["weak_topics"]:
            st.write(f"• {w['topic']} — средний балл {w['avg_score']}")
    st.divider()
    st.caption("Примеры: «Проверь меня по теме микрометр», «маятник», «закон Ома». Чтобы закончить — «стоп».")
    st.caption("Тексты в базе — ДЕМО, не методички МИФИ.")


def render(turn: dict) -> None:
    grade = turn.get("grade")
    if grade and (turn["status"] == "question" or turn["status"] == "finished"):
        icon, color = VERDICT_STYLE.get(grade["verdict"], ("•", "gray"))
        page = f" · с. {grade['page']}" if grade.get("page") else ""
        st.markdown(f"{icon} **:{color}[{grade['verdict'].capitalize()}]**{page}  \n{grade['explanation']}")
    if turn["status"] == "question":
        q = turn["question"]
        if q["number"] == 1 and q["kind"] != "followup" and turn.get("notice"):
            st.caption(turn["notice"])
        title = "Уточнение" if q["kind"] == "followup" else f"Вопрос {q['number']} из {q['max_questions']}"
        badge = ":orange[не из методички]" if q["origin"] == "general" else f":blue[{q['label']}]"
        st.markdown(f"**{title}** {badge}\n\n{q['question']}")
    elif turn["status"] == "finished":
        st.markdown(turn["message"].replace("\n", "  \n"))
    elif turn["status"] == "refused":
        st.warning(turn["message"])
    else:
        st.error(turn["message"] or "Что-то пошло не так.")
    q = turn.get("question") or {}
    if turn.get("sources") and turn["status"] == "question" and q.get("origin") == "manual" and q.get("number") == 1:
        with st.expander("Источники (найдено поиском)"):
            for s in turn["sources"]:
                st.write(f"{s['work']}, с. {s['page']} — {s['section']}")


def feedback_buttons(i: int, msg: dict) -> None:
    if msg.get("feedback_sent"):
        st.caption("Спасибо за отзыв!")
        return
    c1, c2, _ = st.columns([1, 1, 6])
    for col, useful, label in ((c1, True, "👍"), (c2, False, "👎")):
        if col.button(label, key=f"fb{useful}{i}", help="полезно" if useful else "не полезно"):
            body = {"user_id": user_id, "chat_id": st.session_state.chat_id, "useful": useful, "message": msg["turn"]["message"][:4000]}
            if api("POST", "/feedback", json=body):
                msg["feedback_sent"] = True
                st.rerun()


# --- Чат --------------------------------------------------------------------
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            render(msg["turn"])
            feedback_buttons(i, msg)

if prompt := st.chat_input("Тема для проверки или ваш ответ"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.status("Экзаменатор работает…", expanded=False) as status:
            turn = api("POST", "/chat", json={"user_id": user_id, "chat_id": st.session_state.chat_id, "message": prompt})
            if turn:
                for e in turn.get("events", [])[-8:]:
                    mark = "✓" if e["ok"] else "✗"
                    st.write(f"{mark} {NODE_NAMES.get(e['node'], e['node'])} — {e['elapsed_s']} с")
                status.update(label=f"Готово · вызовов модели: {turn['llm_calls']}", state="complete")
            else:
                status.update(label="Ошибка", state="error")
    if turn:
        st.session_state.messages.append({"role": "assistant", "turn": turn})
        st.rerun()
