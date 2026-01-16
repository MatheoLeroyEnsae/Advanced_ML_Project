# Advanced_ML_Project
Projet_Advanced_ML 

# Introduction 

This project have been created by Matheo LEROY, ENSAE Paris.
The goal of this project is to quantify the uncertainty of LLMs on the QA task.
Please refer to the write-up for more details.

# Reproduction of the results

The code runs with the service VSCODE-python-gpu from onixya. 
It uses the NVIDIA GPU :  NVIDIA A2 – 16 Go VRAM

## Clone the repo from onixya 

add the following commands to the terminal :

ssh-keygen -t ed25519 -C “email@ensae.fr” 
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
// copy the key and add it to github in settings SSH key
ssh -T git@github.com
git clone git@github.com:MatheoLeroyEnsae/Advanced_ML_Project.git

## To run the code

- In the terminal, ensure you are in the folder of the project, to do so run:

    cd Advanced_ML

- Use the makefile commands to run the code. 

    make install ": install the dependencies used in this project" 
    make run ": run the main file __main__.py of the src package. 
