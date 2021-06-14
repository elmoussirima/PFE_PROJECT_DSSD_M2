import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Importation Dataset
df_DSSD = pd.read_csv('code/DSSD/dataset_DSSD.csv')
df_E_Business = pd.read_csv('code/E_Business/dataset_E_Business.csv')
df_WI = pd.read_csv('code/WI/dataset_WI.csv')



####      Models E_Business
Model_M1_E_Business = pickle.load(open('code/E_Business/Model_M1_E_Business.pkl', 'rb'))
Model_M2_E_Business = pickle.load(open('code/E_Business/Model_M2_E_Business.pkl', 'rb'))

#### DSSD
Model_M1_DSSD = pickle.load(open('code/DSSD/Model_M1_DSSD.pkl', 'rb'))
Model_M2_DSSD = pickle.load(open('code/DSSD/Model_M2_DSSD.pkl', 'rb'))

#### WI
Model_M1_WI = pickle.load(open('code/WI/Model_M1_WI.pkl', 'rb'))
Model_M2_WI = pickle.load(open('code/WI/Model_M2_WI.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    pred_output = encodage(final_features)

      

    prediction_M1_E_Business = Model_M1_E_Business.predict([pred_output])
    prediction_M2_E_Business = Model_M1_E_Business.predict([pred_output])

    prediction_M1_DSSD = Model_M1_DSSD.predict([pred_output])
    prediction_M2_DSSD = Model_M2_DSSD.predict([pred_output])
    
    prediction_M1_WI = Model_M1_WI.predict([pred_output])
    prediction_M2_WI = Model_M2_WI.predict([pred_output])

    pred_output[9]=pred_output[9].astype(int)
    pred_output [11] = pred_output[11].astype(int)

    data_z_DSSD = data_z(df_DSSD,pred_output)
    data_z_E_Business = data_z(df_E_Business,pred_output)
    data_z_WI = data_z(df_WI,pred_output)

    DSSD_erreur = False
    E_Business_erreur = False
    WI_erreur = False
    erreur = False

    if ((abs(data_z_DSSD[0]) >= 3) | (abs(data_z_DSSD[1]) >= 3) | (abs(data_z_DSSD[2]) >= 3 ) | (abs(data_z_DSSD[3]) >= 3)| (abs(data_z_DSSD[4]) >= 3)| (abs(data_z_DSSD[5]) >= 3)| (abs(data_z_DSSD[6]) >= 3)| (abs(data_z_DSSD[7]) >= 3 )| (abs(data_z_DSSD[8]) >= 3)| (abs(data_z_DSSD[9]) >= 3)):
        DSSD_erreur = True
        erreur = True
    if ((abs(data_z_E_Business[0]) >= 3) | (abs(data_z_E_Business[1]) >= 3) | (abs(data_z_E_Business[2]) >= 3 ) | (abs(data_z_E_Business[3]) >= 3)| (abs(data_z_E_Business[4]) >= 3)| (abs(data_z_E_Business[5]) >= 3)| (abs(data_z_E_Business[6]) >= 3)| (abs(data_z_E_Business[7]) >= 3 )| (abs(data_z_E_Business[8]) >= 3)| (abs(data_z_E_Business[9]) >= 3)):
        E_Business_erreur = True
        erreur = True
    if  ((abs(data_z_WI[0]) >= 3) | (abs(data_z_WI[1]) >= 3) | (abs(data_z_WI[2]) >= 3 ) | (abs(data_z_WI[3]) >= 3)| (abs(data_z_WI[4]) >= 3)| (abs(data_z_WI[5]) >= 3)| (abs(data_z_WI[6]) >= 3)| (abs(data_z_WI[7]) >= 3 )| (abs(data_z_WI[8]) >= 3)| (abs(data_z_WI[9]) >= 3)):
        WI_erreur = True
        erreur = True


    return render_template('result.html',DSSD_erreur = DSSD_erreur, E_Business_erreur = E_Business_erreur, WI_erreur = WI_erreur, final_features = pred_output,prediction_M1_DSSD = prediction_M1_DSSD, prediction_M2_DSSD = prediction_M2_DSSD, prediction_M1_E_Business = prediction_M1_E_Business, prediction_M2_E_Business = prediction_M2_E_Business, prediction_M1_WI = prediction_M1_WI, prediction_M2_WI = prediction_M2_WI, data_z_DSSD = data_z_DSSD, data_z_E_Business = data_z_E_Business, data_z_WI = data_z_WI, erreur = erreur)

def encodage(final_features):
    pred_output = []
    if (final_features[0][0] == 1):
        pred_output.append(0)
        pred_output.append(1)
        pred_output.append(0)
        pred_output.append(0)

    elif(final_features[0][0]== 2):
        pred_output.append(0)
        pred_output.append(0)
        pred_output.append(1)
        pred_output.append(0)

    elif(final_features[0][0]== 3):
        pred_output.append(1)
        pred_output.append(0)
        pred_output.append(0)
        pred_output.append(0)

    else:
        pred_output.append(0)
        pred_output.append(0)
        pred_output.append(0)
        pred_output.append(1)


    if(final_features[0][1]==3):
        pred_output.append(1)
        pred_output.append(0)
        pred_output.append(0)
        pred_output.append(0)
        pred_output.append(0)

    elif(final_features[0][1]==1):
        pred_output.append(0)
        pred_output.append(1)
        pred_output.append(0)
        pred_output.append(0)
        pred_output.append(0)

    elif(final_features[0][1]==4):
        pred_output.append(0)
        pred_output.append(0)
        pred_output.append(1)
        pred_output.append(0)
        pred_output.append(0)
    elif(final_features[0][1]==5):
        pred_output.append(0)
        pred_output.append(0)
        pred_output.append(0)
        pred_output.append(1)
        pred_output.append(0)
    else:
        pred_output.append(0)
        pred_output.append(0)
        pred_output.append(0)
        pred_output.append(0)
        pred_output.append(1)
   
    for i in range(2,12):
        pred_output.append(final_features[0][i]) 

    
    return (pred_output)


@app.route('/consulter')
def consulter():
    data_z = request.form.values()
    return render_template('new.html', data_z = data_z)


def data_z(df, pred_output):

    df = pd.concat([df['Bac_Année'],df['Bac_Moyenne'],df['Licence_Année'],
               df['Licence_Moy_Informatique'],	df['Licence_Moy_Gestion'],df['Licence_Moy_Mathématiques'],	
               df['Licence_Moy_Langues_et_étiques_de_l\'information'],df['Licence_Moy_L1'],	df['Licence_Moy_L2'],	
               df['Licence_Moy_L3']], axis=1)
    
    data = [pred_output[9],pred_output[10],pred_output[11],pred_output[12],pred_output[13],pred_output[14],pred_output[15],pred_output[16],pred_output[17],pred_output[18]]
    
    data_z = (data-df.mean())/(df.std())
        
    return (data_z)


if __name__ == "__main__":
    app.run(debug=True)