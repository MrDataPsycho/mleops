FROM public.ecr.aws/lambda/python:3.7

RUN yum -y install gcc-c++

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR ${LAMBDA_TASK_ROOT}
COPY . .

RUN pip install -r deploy-requirements.txt --no-cache-dir
#RUN pipenv install --deploy --ignore-pipfile
# RUN python3 python_version.py
# EXPOSE 8000
RUN python lambda_handler.py

# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
CMD [ "lambda_handler.handler"]