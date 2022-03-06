run:
	TOKENIZERS_PARALLELISM=false pipenv run python train.py training.max_epochs=5

serve:
	uvicorn app:app --host 0.0.0.0 --port 8000 --reload

make_req:
	pipenv lock -r > deploy-requirements.txt

docker_run:
	docker run -p 8000:8000 --name inference_container inference:latest