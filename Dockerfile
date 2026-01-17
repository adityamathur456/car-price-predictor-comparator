FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir streamlit pandas numpy==1.26.4 scikit-learn==1.3.2 catboost==1.2.5

EXPOSE 8501

CMD ["streamlit","run","app/app.py","--server.port=8501","--server.address=0.0.0.0"]
