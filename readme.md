## Steps to run :

## Step-1 : cd <to root directory of project>
## Step-2 : Create the virtual environment using poetry using below command 

poetry install

or create virtual environment using conda for python 3.13.5 and run 

pip install -r requirements.txt

## Step-3 : Run inference

python src/inference.py <path to image file>

(the output images will be saved in output folder)
## Step-4 : To test FastAPI

step-1 : start the FastAPI server

uvicorn app:app --reload --port 8001

step-2 : Test by sending request to the API

curl -X POST http://127.0.0.1:8001/process/   -F image=@<path to image>

or 

Test the api in postman 


## Training Script

Training script is provided as .ipynb file i.e., extra_scripts/yolo_polygon_seg_training.ipynb

## Script to Convert DocXPan_25k Dataset to YOLO format

extra_scripts/DocXPand_25k_to_YOLO_Format.ipynb