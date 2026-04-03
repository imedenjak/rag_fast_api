FROM python:3.12-slim

WORKDIR /app

ENV PYTHONPATH=/app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
