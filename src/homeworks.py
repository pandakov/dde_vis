import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text, inspect

# ---------- Streamlit setup ----------
st.set_page_config(page_title="PostgreSQL Explorer", page_icon="🗄️", layout="wide")
st.title("Домашние задания")

# ---------- Connect to PostgreSQL ----------
DB_USER = st.secrets["postgres"]["user"]
DB_PASS = st.secrets["postgres"]["password"]
DB_HOST = st.secrets["postgres"]["host"]
DB_PORT = st.secrets["postgres"]["port"]
DB_NAME = "homeworks"


@st.cache_resource
def get_engine():
    conn_str = (
        f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    return create_engine(conn_str)


engine = get_engine()
SCHEMA = "public"


@st.cache_data(ttl=300)
def list_tables(_engine) -> list[str]:
    insp = inspect(_engine)
    return insp.get_table_names(schema=SCHEMA)


@st.cache_data(ttl=300, show_spinner=False)
def load_table_df(_engine, table: str, limit: int = 1000) -> pd.DataFrame:
    q = text(f'SELECT * FROM "{SCHEMA}"."{table}" LIMIT :lim')
    with _engine.begin() as conn:
        return pd.read_sql(q, conn, params={"lim": limit})


@st.cache_data(ttl=300, show_spinner=False)
def table_rowcount(_engine, table: str) -> int:
    q = text(f'SELECT COUNT(*) AS n FROM "{SCHEMA}"."{table}"')
    with _engine.begin() as conn:
        return pd.read_sql(q, conn).iloc[0, 0]


# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Настройки подключения")

    tables = list_tables(engine)
    if not tables:
        st.error("В схеме public нет таблиц.")
        st.stop()

    default_table_idx = tables.index("titanic") if "titanic" in tables else 0
    table_sel = st.selectbox("Таблица", tables, index=default_table_idx)
    limit_rows = st.slider("Лимит строк (предпросмотр)", 10, 100, 20, step=10)

    refresh_data = st.button("♻️ Обновить данные", use_container_width=True)
if refresh_data:
    load_table_df.clear()
    table_rowcount.clear()
    list_tables.clear()
    st.toast("Кэш обновлён")

# ---------- Load data ----------
with st.spinner(f'Загрузка таблицы "{SCHEMA}.{table_sel}"...'):
    df = load_table_df(engine, table_sel, limit=limit_rows)

df.columns = [c.lower() for c in df.columns]
st.subheader(f"Студент: {table_sel.capitalize()}")
st.dataframe(df, use_container_width=True)

# ---------- Stats ----------
try:
    total_rows = table_rowcount(engine, table_sel)
    st.caption(f"Всего строк: **{total_rows:,}**")
except Exception:
    st.caption("Не удалось получить COUNT(*) — проверь права SELECT.")

col1, col2 = st.columns(2)
with col1:
    st.metric("Колонки", len(df.columns))
with col2:
    st.metric("Строк (предпросмотр)", len(df))
