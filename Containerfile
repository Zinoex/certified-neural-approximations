FROM python:3.7-slim
COPY . /app
WORKDIR /app
RUN pip install .
CMD ["certified_neural_approximations"]
