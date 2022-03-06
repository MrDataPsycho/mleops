FROM huggingface/transformers-pytorch-cpu:latest

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

COPY ./ /app
WORKDIR /app


RUN pip install -r deploy-requirements.txt
#RUN pipenv install --deploy --ignore-pipfile
RUN python3 python_version.py
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
