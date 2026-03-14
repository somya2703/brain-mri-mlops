build:
	docker compose build

run:
	docker compose up

train:
	docker compose run training

batch:
	docker compose run batch_inference