FROM python:3.12-slim

WORKDIR /app
COPY frontend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/streamlit_app.py ./streamlit_app.py
COPY data /data

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
