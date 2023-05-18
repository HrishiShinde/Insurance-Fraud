FROM python:3.11.1

WORKDIR /fraud_app

EXPOSE 8501

COPY . /fraud_app

RUN pip install -r requirements.txt

CMD streamlit run server.py