# CV Project Summary:


## The project folder have 2 main folders:

  1.models: contain datasets and notebooks for training model:
  
    - we have 2 models and two datasets, each dataset for its model.
    - first model is “Cucumber maturity detection” which detect cucumber objects and classify itsmaturity level ( devolping or maturing).
    - seconf model is “Cucumber leaf disease recognition” which classify the disease of cucmber leaf.

  
  2.app: contain backend and UI files:	
  
    - backend: class for model methods used in UI.
    - UI: simple UI with streamlit library in python.


## How run project:

  1. refactor paths in code to tour paths.

  2.install requirements with:
    ```pip install -r requirements.txt```
    
  3.run UI with 
    a)ensure you are on app folder in terminal then 
    b)run ```streamlit run streamlit_ui.py```
