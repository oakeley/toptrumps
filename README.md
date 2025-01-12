# toptrumps
HMM model for learning how to play top trumps with Ollama Phi4 for language comprehension
## Train and save the best model
python toptrump.py train
## Load the best model and play the game on the screen with the text prompts and cards from each play displayed live so we can watch the game
python toptrump.py demo Top_Trumps_Dinosaurs.csv
## Allow a human being to substitute for player 1 so that it can play directly against the LLM using the best model and a selected card deck
python toptrump.py human Top_Trumps_Dr_Who_45_Years_of_Time_Travel.csv

# Installation instructions
## Get card data
git clone https://github.com/APStats/Top-Trumps-data
## Rename cards to remove illegal characters
ls Top-Trumps-data/ | grep Top > start; ls Top-Trumps-data/ | grep Top | sed 's/ - /_/g' | tr " " "_" | tr -d "()" | tr -s "_" > stop; paste start stop | sed 's/ /\\ /g' | sed 's/(/\\(/g' | sed 's/)/\\)/g' | awk -F"\t" '{print "cp Top-Trumps-data/"$1" "$2}' > ren.sh

bash ren.sh

## Create conda environment
conda create -n trump

conda activate trump

conda install pomegranate  -y

conda install tenacity -y

conda install pandas -y

#conda install hmmlearn -y

conda install httpx  -y

conda install typing -y

pip install argparse
