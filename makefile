JFILE =
PFILE =
FUNCTION =
SFUNCTION = "$(FUNCTION)"

all: 
	make convert-directory
	make testing-directory
convert:
	jupyter nbconvert --to python notebooks/exercises/$(JFILE).ipynb --output-dir pythonfiles/files
testing:
	pytest pythonfiles/tests/$(PFILE).py
conv-test:
	make convert
	make testing
convert-directory:
	jupyter nbconvert --to python notebooks/exercises/*.ipynb --output-dir pythonfiles/files
testing-directory:
	chmod u+x pythonfiles/tests/searchfile.sh
	bash pythonfiles/tests/searchfile.sh
testing-function:
	pytest pythonfiles/tests/$(PFILE).py -k $(SFUNCTION)
conv-test-func:
	make convert
	make testing-function