FROM python:3.10
WORKDIR /SpaceshipTitanic
COPY poetry.lock pyproject.toml main.py LiidClassifierModel.py .
RUN pip install poetry
RUN POETRY_VIRTUALENVS_CREATE=false poetry install --no-interaction --no-ansi
ENTRYPOINT ["python", "main.py"]