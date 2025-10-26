# Код к практическому занятию №7 "Визуализация"

Ноутбук - [notebooks/vis_test.ipynb](./notebooks/vis_test.ipynb)

## 1. Настройка окружения

### Установка зависимостей

Для настрйоки окружения используется poetry.

Зависимости описаны в файле [pyproject.toml](./pyproject.toml).

В своем переменном окружении выполняем
```bash
poetry install --no-root
```

### Переменные для подключения к базе данных

Прописываем в `.streamlit/secrets.toml`:

```toml
[postgres]
user = "user"
password = "password"
host = "localhost"
port = "5432"
dbname = "db_name"
```

## 2. Запуск

Скрипты Streamlit ([docs](https://docs.streamlit.io/get-started/tutorials/create-an-app)) лежат в [src/](./src/).

Запуск сервиса выполняется командой
```bash
streamlit run <имя скрипта>.py
```
