import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Titanic Data Visualization")

st.set_page_config(
    page_title="Titanic Data Visualization",
    page_icon="🚢",
    layout="wide",
)



@st.cache_data
def load_data():
    return pd.read_parquet("data/titanic.parquet")


df = load_data()

new_col_names = {col: col.lower() for col in df.columns}
df.rename(columns=new_col_names, inplace=True)

col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
total_all = len(df)
surv_rate_all = df["survived"].mean() if "survived" in df.columns else None
surv_rate_f = df["survived"].mean() if "survived" in df.columns else None
avg_age_all = df["age"].mean() if "age" in df.columns else None
avg_age_f = df["age"].mean() if "age" in df.columns else None
avg_fare_all = df["fare"].mean() if "fare" in df.columns else None
avg_fare_f = df["fare"].mean() if "fare" in df.columns else None

with col_kpi1:
    st.metric("Пассажиров", total_all, delta=total_all)
    
with col_kpi2:
    if surv_rate_f is not None:
        st.metric("Доля выживших", f"{surv_rate_f:,.1%}",
                  delta=None if surv_rate_all is None else f"{(surv_rate_f - surv_rate_all):.1%}")
with col_kpi3:
    if avg_age_f is not None:
        st.metric("Средний возраст", f"{avg_age_f:,.1f}",
                  delta=None if avg_age_all is None else f"{(avg_age_f - avg_age_all):.1f}")
with col_kpi4:
    if avg_fare_f is not None:
        st.metric("Средний тариф", f"{avg_fare_f:,.2f}",
                  delta=None if avg_fare_all is None else f"{(avg_fare_f - avg_fare_all):.2f}")


st.subheader("Preview of Dataset")
st.dataframe(df.head(20))

x_axis = st.selectbox("Select X-axis:", df.columns, index=df.columns.get_loc("age"))
y_axis = st.selectbox("Select Y-axis:", df.columns, index=df.columns.get_loc("fare"))
color = st.selectbox("Color by:", df.columns, index=df.columns.get_loc("sex"))
size = st.selectbox("Size by:", df.columns, index=df.columns.get_loc("pclass"))

fig2d = px.scatter(
    df,
    x=x_axis,
    y=y_axis,
    color=color,
    size=size,
    hover_data=["survived", "pclass"],
    title="2D Scatter Plot of Titanic Data",
)
st.plotly_chart(fig2d, use_container_width=True)

st.subheader("3D Scatter Plot")
fig3d = px.scatter_3d(
    df,
    x="age",
    y="fare",
    z="pclass",
    color="sex",
    # size="survived",
    title="Titanic 3D Visualization",
)
st.plotly_chart(fig3d, use_container_width=True)

st.subheader("Survival Distribution by Sex")
fig_hist = px.histogram(df, x="sex", color="survived", barmode="group", title="Survival by Gender")
st.plotly_chart(fig_hist, use_container_width=True)
