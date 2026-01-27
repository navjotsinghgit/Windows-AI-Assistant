# Windows-AI-Assistant
Developed a Windows-based Voice Assistant that performs system automation and user commands using Speech-to-Text (STT) + Natural Language Processing (NLP).  Created a custom dataset of voice commands/intents (like open apps, search web, system control, greetings, etc.) and mapped each intent to specific actions.
Implemented an intent classification algorithm using text preprocessing (tokenization, stopword removal, stemming/lemmatization) to match spoken commands accurately.

Used TF-IDF Vectorizer + Machine Learning classifier (Naive Bayes / SVM / Logistic Regression) for command prediction and smart decision making.

Integrated Text-to-Speech (TTS) to generate human-like responses and improve interactive experience.

Added automation features like opening software, web browsing, file operations, date-time queries, and shutdown/restart commands.

Designed the assistant in a modular way with separate modules for dataset handling, training, prediction, and execution pipeline.

# I have created the model on two different algorithim and different dataset as well

**#SUPERVISED DATA AND LOGISTIC REGRESSION**
In this model i have used the self created dataset and which the voice command manually were recorded to open any software with multiple varation and at different frequencies, further labelled and classified into different claases, then using the librosa library we ahve turned the .wav data to .csv data with there head columns label as 0 to 5 to get the predictiong classification in logistic regression.
the libraries we have used to evalaute the model are :
**pyspark, scikit-learn, numpy, pandas, librosa, pickle, flask**

**#UNSUPERVISED DATA AND XGBOOST**
In this model i have used the kaggle dataset based on the voice assistant data for the iot data and the data preprocessing is different we have applied the audio processing and turn the .wav to .csv with there wavelength classification and  clustered the unsupervised data into different labelled clustered to open different software at different frequency and audio enviroment conditions then we have applied it into the xgboost for furthere evaluation.
the libraries we have used to evaluate the model are :
**scikit-learn, librosa, numpy, pandas, pickle, flask **

AFTER the model binary file i have created two different file one is the control.py which bears the different functions that set the role of different voice command to be execute when it is recongized by the model predicition futher this control.py is inherited to the test.py which is the main program where it takes the input of the voice and by using the model binary file it classifiy the command and execute it to the virtual keyboard of the system and this test.py is having the falsk api medium between the test.py and model binary file to send the data and get the predictions.

More over i'm currently working on this project to make it more refined and improve it by noise reduction in dataset............
