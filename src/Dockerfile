FROM python:3.11

WORKDIR /app

COPY requirements.txt .

# Actualizar pip
RUN python -m pip install --upgrade pip

# Instalar dependencias (pywin32 será ignorado en Linux)
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "index.py"]
