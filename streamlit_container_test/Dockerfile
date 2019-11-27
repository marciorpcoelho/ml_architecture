FROM python:3.7
EXPOSE 8501
WORKDIR /streamlit_container_test
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD streamlit run streamlit_docker_test.py