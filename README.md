# Код к практическому занятию №7 "Визуализация"

## Настройка окружения

### Установка зависимостей

Для настрйоки окружения используется poetry.

Зависимости описаны в файле [pyproject.toml](./pyproject.toml).

В своем переменном окружении выполняем
```bash
poetry install --no-root
```

### Переменные для подключения к базе данных

Прописываем в `.streamlit/secrets.toml`.

Шаблон в [.streamlit/_secrets.toml](./.streamlit/_secrets.toml):

```toml
[postgres]
user = "user"
password = "password"
host = "localhost"
port = "5432"
dbname = "db_name"
```

## Запуск

Скрипты Streamlit лежат в [src/](./src/).

Запуск сервиса выполняется командой
```bash
streamlit run <имя скрипта>.py
```
