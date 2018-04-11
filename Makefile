clean:
	find . -name '*.pyc' -exec rm --force {} +
install:
	pip install -r requirements.txt

