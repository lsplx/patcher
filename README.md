# Description
Repairing Catastrophic-Neglect in Text-to-Image Diffusion Models via Attention-Guided Feature Enhancement
patcher is an attention-guided feature enhancement repair method for fixing catastrophic-neglect issues in Text-to-Image diffusion models.

- main.py is a combination of the Attention Explanational Tool and the Text-to-Image diffusion models.
- Pathcer.py is the repair approach for the T2I DMs.


RADIATION Architecture:  
%![Image text]()

# Packages
-opencv-python
-Pillow
-torch
-flask
-flask_restful
-flask_cors
-diffusers
-transformers
-nltk
-openai

# Environment
git clone -b penguin https://github.com/paulwong16/ecco.git  
cd ecco 
pip install -e .  

cd ..  
git clone https://github.com/YuhengHuang42/daam.git  
cd daam  
pip install -e .  
cd ..  

# Run the code
python main.py  
python Patcher.py







