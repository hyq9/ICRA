ICRA: Intelligent Course Review Analysis
This repository implements ICRA (Intelligent Course Review Analysis), which is divided into two main tasks:
Fake Review Filtering: Identify and filter out potential fake reviews based on user text, sentiment analysis, and scoring inconsistency.
Course Rating Prediction: Predict user course ratings based on the reviews and course descriptions using a custom deep learning model.
The model leverages the ERNIE 3.0 pre-trained language model for Chinese text, and the dataset used in this project comes from MOOC platforms.

Dataset
We use a dataset collected from a MOOC platform. The dataset includes:
User Reviews: Text reviews left by users.
Course Information: Course descriptions, including course IDs and overviews.
You can download the original dataset here:
Dataset Link: ：https://pan.baidu.com/s/1VPSc_DiKW6pXvfcSK_54uQ?pwd=eh68 
Extraction Code: eh68

Pre-trained Model
The model relies on the ERNIE 3.0 pre-trained language model for Chinese. The folder includes the following files:
config.json
pytorch_model.bin
vocab.txt
Ensure that the pre-trained model files are correctly placed under the models/ernie-3.0-base-zh/ directory.

Contact
If you have any questions or need further assistance, feel free to contact the author.