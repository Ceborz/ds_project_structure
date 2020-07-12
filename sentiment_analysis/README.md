## Project setup
Install conda
Create enviroment for conda
In the enviroment run
```
conda install python
conda install -c anaconda pandas
conda install -c conda-forge matplotlib
pip install -r requirements.txt
python -m spacy download en_core_web_lg
python -m spacy download es_core_news_lg
```
## To run
Go inside the folder of the project (sentiment_analysis)
```
cd src/data
python prepare_data.py
cd ../../src/features
python tokenize_data.py
cd ../../src/models
python train.py
```