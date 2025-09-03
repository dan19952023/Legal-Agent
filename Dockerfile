from python:3.12-slim

copy requirements.txt requirements.txt
run pip install --no-cache-dir -r requirements.txt

workdir /workspace
copy app app
copy assets assets
copy server.py server.py

cmd ["python", "server.py", "--db-path", "assets/data/uscis_sample_output_20250806_150543_chunked.json"]