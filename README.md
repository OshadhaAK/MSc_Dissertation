# MSc_Dissertation
MSc dissertation to forecast power consumption in industrial facilities

## Folder Structure

MSc_Dissertation/
│
├── app.py
├── model
    └── model.pkl
    └── preprocessing.py
├── requirements.txt
├── static/
│   ├── images/
        └──prediction_image.png
│   └── style.css
├── templates/
│   └── index.html
└── README.md

## Installing from requirements.txt
To install the packages listed in requirements.txt on another machine or environment, you can use:
```bash
pip3 install -r requirements.txt

## Run the Flask app:

python app.py
Test the app by visiting http://127.0.0.1:5000/ in your browser.