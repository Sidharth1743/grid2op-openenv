FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r server/requirements.txt
RUN pip install -e .

# Pre-download the required Grid2Op dataset at build time.
RUN python -c "import grid2op; env = grid2op.make('l2rpn_case14_sandbox'); env.close()"

EXPOSE 7860

CMD ["uvicorn", "grid2op_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
