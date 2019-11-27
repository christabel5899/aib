import os
import pandas as pd
import numpy as np

#Import modul Random forest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


from flask import Flask, render_template, request

app = Flask(__name__, template_folder="templates")
app.jinja_env.cache = None


df = pd.read_excel("fix.xlsx")
test = pd.read_excel("fix_test.xlsx")

@app.route("/", methods=['GET', 'POST'])
def home():
    test_y1 = None
    if request.method == 'POST':
        
        X=df[['STATE_CODE','CBSA_CODE','COUNTY_CODE','CITY_CODE','INSPECTION_YEAR']].values
        Y=df['INSPECTION_SCORE']
        #X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=42)
        
        rnd_clf=RandomForestClassifier(n_estimators=100)
        rnd_clf.fit(X,Y)

        test_x0 = request.form['state']
        test_x1 = request.form['cbsa']
        test_x2 = request.form['county']
        test_x3 = request.form['city']
        test_x4 = request.form['year']

        test_x=np.vstack((test_x0,test_x1,test_x2,test_x3,test_x4)).T
        test_y1=rnd_clf.predict(test_x)

        print(test_y1)    
    else:
        print("haha")
    return render_template(
        "index.html", 
        result=test_y1   
    )

def main():
    port = int(os.environ.get('PORT', 8080))

    app.run(debug=True, port=port, host='0.0.0.0')


if __name__ == "__main__":
    main()