unittests:
	@echo "Unittesting NER..."
	@nosetests --nocapture --nologcapture  --all-modules --verbose --exe tagger/tests --cover-package=tagger --with-coverage --cover-inclusive --cover-erase --cover-html --cover-html-dir=testing/coverage

docs-local:
	@echo "Making local documentation..."
	@mkdocs serve -a 0.0.0.0:8000

docs-publish:
	@echo "Publishing documentation..."
	@mkdocs gh-deploy