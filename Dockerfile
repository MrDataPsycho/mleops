FROM amazon/aws-lambda-python

RUN yum -y install gcc-c++

ARG MODEL_DIR=./models

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

COPY ./ /app
WORKDIR /app


RUN pip install -r deploy-requirements.txt --no-cache-dir
#RUN pipenv install --deploy --ignore-pipfile
# RUN python3 python_version.py
# EXPOSE 8000
RUN python lambda_handler.py
RUN chmod -R 0755 $MODEL_DIR

# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
CMD [ "lambda_handler.lambda_handler"]