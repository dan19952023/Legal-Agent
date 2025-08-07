find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf
rm -rf agent.zip
zip -r agent.zip app assets server.py Dockerfile requirements.txt