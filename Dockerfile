FROM public.ecr.aws/lambda/python:3.7

RUN yum -y install gcc-c++

WORKDIR ${LAMBDA_TASK_ROOT}
COPY . .

RUN pip install -r deploy-requirements.txt --no-cache-dir
RUN python lambda_handler.py

CMD [ "lambda_handler.handler"]