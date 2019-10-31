unittests:
	@echo "Unittesting NER..."
	@nosetests --nocapture --nologcapture  --all-modules --verbose --exe tagger/tests --cover-package=tagger --with-coverage --cover-inclusive --cover-erase --cover-html --cover-html-dir=testing/coverage
