Create a Environment
```bash
conda create -n rainPrediction python=3.7 -y
```
Activate Environment
```bash
conda activate rainPrediction
```
install the requirements
```bash
pip install -r requirements.txt
```
create template.py
create the project structure by running the template.py

```bash
git init
```
```bash
dvc init
```
```bash
dvc add data_given/weatherAUS.csv
```