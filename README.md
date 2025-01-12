# toptrumps
HMM model for learning how to play top trumps with Ollama Phi4 for language comprehension
# Installation instructions
## Get card data
git clone https://github.com/APStats/Top-Trumps-data
## Rename cards to remove illegal characters
ls Top-Trumps-data/ | grep Top > start; ls Top-Trumps-data/ | grep Top | sed 's/ - /_/g' | tr " " "_" | tr -d "()" | tr -s "_" > stop; paste start stop | sed 's/ /\\ /g' | sed 's/(/\\(/g' | sed 's/)/\\)/g' | awk -F"\t" '{print "cp Top-Trumps-data/"$1" "$2}' > ren.sh

bash ren.sh
## Create conda environment
conda create -n trump

conda activate trump

conda install numpy pandas hmmlearn random json pickle httpx argparse typing tenacity pathlib
