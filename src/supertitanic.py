from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------- Page config ----------
st.set_page_config(
    page_title="Titanic Data Visualization",
    page_icon="🚢",
    layout="wide",
)

st.title("🚢 Titanic Data Visualization — Advanced")


# ---------- Data loading with cache ----------
@st.cache_data(ttl=600, show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(ttl=600, show_spinner=False)
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    return df


with st.sidebar:
    st.header("📁 Источник данных")
    uploaded = st.file_uploader("Загрузить .parquet (опционально)", type=["parquet"])
    default_path = "data/titanic.parquet"
    data_status = "uploaded" if uploaded else "default"
    st.caption(
        f"Используется: **{ 'загруженный файл' if uploaded else default_path }**"
    )

with st.spinner("Загрузка данных..."):
    if uploaded:
        df_raw = pd.read_parquet(uploaded)
    else:
        if not Path(default_path).exists():
            st.error(
                f"Файл {default_path} не найден. Загрузите .parquet через сайдбар."
            )
            st.stop()
        df_raw = load_parquet(default_path)

df = standardize_columns(df_raw)

# ---------- Basic hygiene ----------
# Заполним некоторые NaN для удобства фильтрации
for col in ["sex", "class", "embarked"]:
    if col in df.columns:
        df[col] = df[col].astype("category")

# Вычислим полезные признаки (если столбцы есть)
if {"sibsp", "parch"}.issubset(df.columns):
    df["family_size"] = df["sibsp"].fillna(0) + df["parch"].fillna(0) + 1
if "age" in df.columns:
    df["is_child"] = (df["age"] < 16).astype("Int64")


# ---------- Query params helpers ----------
def get_param_list(name, fallback):
    qp = st.query_params
    if name not in qp:
        return fallback
    vals = qp.get_all(name)
    # если было одно значение – всё равно список
    return [type(fallback[0])(v) for v in vals] if fallback else vals


def set_query_params(**kwargs):
    # только непустые параметры записываем
    qp = {}
    for k, v in kwargs.items():
        if isinstance(v, list):
            if len(v) > 0:
                qp[k] = [str(x) for x in v]
        elif v is not None:
            qp[k] = str(v)
    st.query_params.clear()
    st.query_params.update(qp)


# ---------- Sidebar filters ----------
with st.sidebar:
    st.header("🎛️ Фильтры")
    sex_options = (
        sorted(df["sex"].dropna().unique().tolist()) if "sex" in df.columns else []
    )
    class_options = (
        sorted(df["pclass"].dropna().unique().tolist())
        if "pclass" in df.columns
        else []
    )
    survived_options = (
        sorted(df["survived"].dropna().unique().tolist())
        if "survived" in df.columns
        else []
    )

    # Диапазоны
    age_min, age_max = (
        (int(df["age"].min()), int(df["age"].max()))
        if "age" in df.columns
        else (0, 100)
    )
    fare_min, fare_max = (
        (float(df["fare"].min()), float(df["fare"].max()))
        if "fare" in df.columns
        else (0.0, 600.0)
    )

    # Восстановление из query params
    sex_sel = st.multiselect(
        "Пол", options=sex_options, default=get_param_list("sex", sex_options[:])
    )
    pclass_sel = st.multiselect(
        "Класс (pclass)",
        options=class_options,
        default=get_param_list("pclass", class_options[:]),
    )
    survived_sel = st.multiselect(
        "Выжил (0/1)",
        options=survived_options,
        default=get_param_list("survived", survived_options[:]),
    )

    age_range = st.slider(
        "Возраст",
        min_value=age_min,
        max_value=age_max,
        value=(age_min, age_max),
        step=1,
        disabled="age" not in df.columns,
    )
    fare_range = st.slider(
        "Тариф (fare)",
        min_value=float(fare_min),
        max_value=float(fare_max),
        value=(float(fare_min), float(fare_max)),
        step=1.0,
        disabled="fare" not in df.columns,
    )

    log_y_fare = st.toggle("Лог-ось Y (для fare в 2D графике)", value=False)

    # Обновляем query params
    set_query_params(
        sex=sex_sel,
        pclass=pclass_sel,
        survived=survived_sel,
    )

# ---------- Apply filters ----------
filtered = df.copy()


def between_num(s, lo, hi):
    return (
        s.between(lo, hi)
        if pd.api.types.is_numeric_dtype(s)
        else pd.Series([True] * len(s), index=s.index)
    )


if "sex" in filtered.columns and sex_sel:
    filtered = filtered[filtered["sex"].isin(sex_sel)]
if "pclass" in filtered.columns and pclass_sel:
    filtered = filtered[filtered["pclass"].isin(pclass_sel)]
if "survived" in filtered.columns and survived_sel:
    filtered = filtered[filtered["survived"].isin(survived_sel)]
if "age" in filtered.columns:
    filtered = filtered[between_num(filtered["age"], age_range[0], age_range[1])]
if "fare" in filtered.columns:
    filtered = filtered[between_num(filtered["fare"], fare_range[0], fare_range[1])]

st.toast(f"Данных после фильтрации: {len(filtered)}", icon="✅")

# ---------- KPIs ----------
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
total_all = len(df)
total_f = len(filtered)
surv_rate_all = df["survived"].mean() if "survived" in df.columns else None
surv_rate_f = filtered["survived"].mean() if "survived" in filtered.columns else None
avg_age_all = df["age"].mean() if "age" in df.columns else None
avg_age_f = filtered["age"].mean() if "age" in filtered.columns else None
avg_fare_all = df["fare"].mean() if "fare" in df.columns else None
avg_fare_f = filtered["fare"].mean() if "fare" in filtered.columns else None

with col_kpi1:
    st.metric("Пассажиров (фильтр)", total_f, delta=total_f - total_all)
with col_kpi2:
    if surv_rate_f is not None:
        st.metric(
            "Доля выживших",
            f"{surv_rate_f:,.1%}",
            delta=(
                None
                if surv_rate_all is None
                else f"{(surv_rate_f - surv_rate_all):.1%}"
            ),
        )
with col_kpi3:
    if avg_age_f is not None:
        st.metric(
            "Средний возраст",
            f"{avg_age_f:,.1f}",
            delta=None if avg_age_all is None else f"{(avg_age_f - avg_age_all):.1f}",
        )
with col_kpi4:
    if avg_fare_f is not None:
        st.metric(
            "Средний тариф",
            f"{avg_fare_f:,.2f}",
            delta=(
                None if avg_fare_all is None else f"{(avg_fare_f - avg_fare_all):.2f}"
            ),
        )

# ---------- Data preview with column_config ----------
st.subheader("👀 Предпросмотр данных (фильтр)")
col_cfg = {}
if "survived" in filtered.columns:
    col_cfg["survived"] = st.column_config.SelectboxColumn(
        "survived", help="0 — не выжил, 1 — выжил", options=[0, 1]
    )
if "sex" in filtered.columns:
    col_cfg["sex"] = st.column_config.SelectboxColumn(
        "sex", options=sorted(filtered["sex"].dropna().unique().tolist())
    )
if "pclass" in filtered.columns:
    col_cfg["pclass"] = st.column_config.NumberColumn(
        "pclass", help="Passenger class (1/2/3)"
    )

st.dataframe(filtered.head(50), use_container_width=True, column_config=col_cfg or None)

# ---------- Tabs for visuals ----------
tab2d, tab3d, tabdist, tabpivot = st.tabs(
    ["✨ 2D Scatter", "📐 3D Scatter", "📊 Распределения", "🧮 Сводные/Теплокарта"]
)

with tab2d:
    x_axis = st.selectbox(
        "X-axis",
        filtered.columns,
        index=(filtered.columns.get_loc("age") if "age" in filtered.columns else 0),
    )
    y_axis = st.selectbox(
        "Y-axis",
        filtered.columns,
        index=(filtered.columns.get_loc("fare") if "fare" in filtered.columns else 1),
    )
    color = st.selectbox(
        "Color by",
        filtered.columns,
        index=(filtered.columns.get_loc("sex") if "sex" in filtered.columns else 0),
    )
    size = st.selectbox(
        "Size by (опционально)", ["—"] + filtered.columns.tolist(), index=0
    )

    size_kw = {} if size == "—" else {"size": size}
    fig2d = px.scatter(
        filtered,
        x=x_axis,
        y=y_axis,
        color=color,
        hover_data=[c for c in ["survived", "pclass"] if c in filtered.columns],
        title="2D Scatter Plot (filtered)",
        **size_kw,
    )
    if log_y_fare and y_axis == "fare":
        fig2d.update_layout(yaxis_type="log", title="2D Scatter Plot (filtered, log Y)")
    st.plotly_chart(fig2d, use_container_width=True)

    # Download current figure as standalone HTML
    html_bytes = fig2d.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")
    st.download_button(
        "⬇️ Скачать график (HTML)",
        data=html_bytes,
        file_name="scatter_2d.html",
        mime="text/html",
    )

with tab3d:
    st.caption("Совет: для 3D лучше выбирать числовые оси.")
    x3 = st.selectbox(
        "X",
        [c for c in filtered.columns if pd.api.types.is_numeric_dtype(filtered[c])],
        index=0,
    )
    y3 = st.selectbox(
        "Y",
        [c for c in filtered.columns if pd.api.types.is_numeric_dtype(filtered[c])],
        index=(
            1
            if len(
                [
                    c
                    for c in filtered.columns
                    if pd.api.types.is_numeric_dtype(filtered[c])
                ]
            )
            > 1
            else 0
        ),
    )
    z3 = st.selectbox(
        "Z",
        [c for c in filtered.columns if pd.api.types.is_numeric_dtype(filtered[c])],
        index=(
            2
            if len(
                [
                    c
                    for c in filtered.columns
                    if pd.api.types.is_numeric_dtype(filtered[c])
                ]
            )
            > 2
            else 0
        ),
    )
    color3 = st.selectbox(
        "Color by",
        filtered.columns,
        index=(filtered.columns.get_loc("sex") if "sex" in filtered.columns else 0),
    )
    fig3d = px.scatter_3d(
        filtered,
        x=x3,
        y=y3,
        z=z3,
        color=color3,
        title="Titanic 3D Visualization (filtered)",
    )
    st.plotly_chart(fig3d, use_container_width=True)

with tabdist:
    c1, c2 = st.columns(2)
    with c1:
        if "sex" in filtered.columns and "survived" in filtered.columns:
            fig_hist = px.histogram(
                filtered,
                x="sex",
                color="survived",
                barmode="group",
                title="Survival by Sex",
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        elif "sex" in filtered.columns:
            st.info("Нет столбца 'survived' — показываю распределение по полу.")
            fig_hist = px.histogram(filtered, x="sex", title="Distribution by Sex")
            st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        if "age" in filtered.columns:
            fig_age = px.histogram(
                filtered, x="age", nbins=30, marginal="box", title="Age Distribution"
            )
            st.plotly_chart(fig_age, use_container_width=True)
        if "fare" in filtered.columns:
            fig_fare = px.histogram(
                filtered,
                x="fare",
                nbins=40,
                marginal="violin",
                title="Fare Distribution",
            )
            st.plotly_chart(fig_fare, use_container_width=True)

with tabpivot:
    st.caption("Сводная по средней выживаемости (можно менять оси).")
    rows = st.selectbox(
        "Строки",
        options=[
            c
            for c in filtered.columns
            if filtered[c].dtype.name not in ("float64", "float32")
        ],
        index=0 if "sex" in filtered.columns else 0,
    )
    cols = st.selectbox(
        "Столбцы",
        options=[
            c
            for c in filtered.columns
            if filtered[c].dtype.name not in ("float64", "float32")
        ],
        index=0 if "pclass" in filtered.columns else 0,
    )
    if "survived" in filtered.columns:
        pv = filtered.pivot_table(
            index=rows, columns=cols, values="survived", aggfunc="mean"
        )
        pv = pv.sort_index().sort_index(axis=1)
        st.dataframe((pv * 100).round(1).astype(str) + " %", use_container_width=True)
        fig_heat = px.imshow(
            pv, text_auto=".1%", aspect="auto", title="Heatmap: Survival Rate"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.warning("Столбец 'survived' не найден — теплокарта недоступна.")

# ---------- Download filtered data ----------
csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Скачать отфильтрованные данные (CSV)",
    data=csv_bytes,
    file_name="titanic_filtered.csv",
    mime="text/csv",
)

# ---------- Data summary in expander ----------
with st.expander("ℹ️ Справка/описание данных"):
    st.markdown(
        """
        - **pclass** — класс билета (1/2/3)  
        - **sex** — пол  
        - **age** — возраст  
        - **sibsp, parch** — родственники на борту (сиблинги/супруги, родители/дети)  
        - **fare** — стоимость билета  
        - **embarked** — порт посадки  
        - **survived** — 0/1 (не выжил/выжил)  
        - **family_size** — вычисляемый признак (sibsp + parch + 1)  
        - **is_child** — возраст < 16  
        """
    )
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

# Final toast
st.toast(f"Источник данных: {data_status}", icon="📦")
