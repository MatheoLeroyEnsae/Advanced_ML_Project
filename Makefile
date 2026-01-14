.PHONY: install run-main run-all test clean

install:
	@echo " Install libraries"
	python install_requirements.py

run: 
	@echo " Launch the program with the defined config"
	python -m src