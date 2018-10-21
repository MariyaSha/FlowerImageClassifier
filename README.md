# aipnd-project
---
This Image Classifier project was part of my Nanodegree Program with *Udacity*
This is the first time I've ever created an AI that trains, validates and predicts from scratch.
---
#Authors & Contributors
---
Mariya Shavshyshvili, with the help and instructions of the Udacity Nanodegree Program.
---
**IMPORTANT CREDITS**
---
*Johnny Metz's* video on argparse helped me out a lot with understanding it's uses:
https://youtu.be/cdblJqEUDNo
*LearnCode.academy's* video on GitHub helped me with uploading this project to GitHub:
https://youtu.be/0fKg7e37bQE
---
#File List
---
cat_to_name.json
functions_and_classes.py
LICENSE
predict.py
README.md
train.py
---
#Terminal Instructions
---
Based on project instructions from *Udacity*, there are a few specific details about the terminal use that are best to note;
---
**train.py**

In the terminal, train.py is best called with:
```
cd python train.py ./flowers -g
```
with ```cd``` being the directory in which train.py is in.
with ```-g``` representing the GPU connection so that everything runs faster.
with ```./flowers``` being the directory in which we save our training, validation and testing sets.
the ```./flowers``` directory would have 3 folders: train, valid, test.
These folders would be separated into sub-folders, holding the class number of the flower photos they contain.
---
**predict.py**

In the terminal, predict.py is best called with:
```
cd python predict.py ./flowers/test/11/image_03098.jpg ./save_directory/checkpoint.pth -g
```

with ```cd``` being the directory in which predict.py is in.
with ```./flowers/test/11/image_03098.jpg``` being the path to the image we'd like to predict.
with ```./save_directory/checkpoint.pth``` being the path to the checkpoint we saved in train.py.
with ```-g``` representing the GPU connection so that everything runs faster.
---
#Please note
---
As this project was submitted and grated, to avoid direct duplication of the code by future students I won't include the name of the Nanodegree program or the actual name of the project, as stated in the syllabus.
