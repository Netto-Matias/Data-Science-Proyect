from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

import pickle
import pandas as pd
from fastapi.encoders import jsonable_encoder
import gradio as gr

app = FastAPI()
#Modelo 
PARAMS_NAME = [
    "orderAmount",
    "orderState",
    "paymentMethodRegistrationFailure",
    "paymentMethodType",
    "paymentMethodProvider",
    "paymentMethodIssuer",
    "transactionAmount",
    "transactionFailed",
    "emailDomain",
    "emailProvider",
    "customerIPAddressSimplified",
    "sameCity"
]


MODEL_PATH = 'data/modelo_proyecto_final.pkl'
with open(MODEL_PATH, "rb") as handle: 
    model = pickle.load(handle)

#Columnas
COLUMNS_PATH = 'data/categories_ohe_without_fraudulent.pickle'
with open(COLUMNS_PATH, "rb") as handle: 
    ohe_tr = pickle.load(handle)

BINS_ORDER = 'data/saved_bins_order.pickle'
with open(BINS_ORDER, 'rb') as handle:
    new_saved_bins_order = pickle.load(handle)

BINS_TRANSACTION = 'data/saved_bins_transaction.pickle'
with open(BINS_TRANSACTION, 'rb') as handle:
    new_saved_bins_transaction = pickle.load(handle)



def predict_fraud_customer(*args):


    answer_dict = {}
    
    for i in range(len(PARAMS_NAME)):
        answer_dict[PARAMS_NAME[i]] = [args[i]]

    single_instance = pd.DataFrame.from_dict(answer_dict)

    # Manejo de puntos de corte
    single_instance["orderAmount"] = single_instance["orderAmount"].astype(float)
    single_instance["orderAmount"] = pd.cut(single_instance["orderAmount"],
                                            bins=new_saved_bins_order,
                                            include_lowest=True)

    single_instance["transactionAmount"] = single_instance["transactionAmount"].astype(int)
    single_instance["transactionAmount"] = pd.cut(single_instance["transactionAmount"],
                                                bins=new_saved_bins_transaction,
                                                include_lowest=True)

    # One hot encoding
    single_instance_ohe = pd.get_dummies(single_instance).reindex(columns=ohe_tr).fillna(0)

    prediction = model.predict(single_instance_ohe)

    # Cast numpy.int64 to just a int
    type_of_fraud = int(prediction[0])

    response = "Error parsing value"
    if type_of_fraud == 0:
        response = "False"
    elif type_of_fraud == 1:
        response = "True"
    elif type_of_fraud == 2:
        response = "Warning"

    return response


with gr.Blocks() as demo: 
    gr.Markdown(
        """
        #Prevención de Fraude
        """
    )

    with gr.Row():
        with gr.Column():

            gr.Markdown(
                """
                ## Predecir si un cliente es fraudulento o no.
                """
            )
            
            orderAmount = gr.Slider(label="Order amount", minimum=0, maximum=353)
            orderState = gr.Radio(
                label = "Order state",
                choices=['failed', 'fulfilled', 'pending']
                )
            paymentMethodRegistrationFailure = gr.Radio(
                label = "Payment method registration failure",
                choices = ['True', 'False']
                )
            paymentMethodType = gr.Radio(
                label = "Payment method type",
                choices = ['apple pay', 'bitcoin', 'card', 'paypal']
            )
            paymentMethodProvider = gr.Dropdown(
                label = "Payment method provider",
                choices = ['American Express',
                            'Diners Club / Carte Blanche',
                            'Discover',
                            'JCB 15 digit',
                            'JCB 16 digit',
                            'Maestro',
                            'Mastercard',
                            'VISA 13 digit',
                            'VISA 16 digit',
                            'Voyager'
                            ],
                multiselect = False
                )
            paymentMethodIssuer = gr.Dropdown(
                label = "Payment method issuer",
                choices = ['Bastion Banks',
                            'Bulwark Trust Corp.',
                            'Citizens First Banks',
                            'Fountain Financial Inc.',
                            'Grand Credit Corporation',
                            'Her Majesty Trust',
                            'His Majesty Bank Corp.',
                            'Rose Bancshares',
                            'Solace Banks',
                            'Vertex Bancorp',
                            'weird'
                            ],
                multiselect = False
                )
            transactionAmount = gr.Slider(label="Transaction amount", minimum=0, maximum=353)
            transactionFailed = gr.Radio(
                label = "Transaction failed",
                choices=['True', 'False']
                )
            emailDomain = gr.Radio(
                label = "Email domain",
                choices=['biz', 'com', 'info', 'net', 'org', 'weird']
                )
            emailProvider = gr.Radio(
                label = "Email provider",
                choices=['gmail', 'hotmail', 'yahoo', 'weird', 'other']
                )
            customerIPAddressSimplified = gr.Radio(
                label = "Customer IP Address",
                choices=['digits_and_letters', 'only_letters']
                )
            sameCity = gr.Radio(
                label = "Same city",
                choices=['no', 'yes', 'unknown']
                )
        
        with gr.Column():

            gr.Markdown(
                """
                ## Predicción
                """
            )

            label = gr.Label(label = "Tipo de fraude")
            predict_btn = gr.Button(value="Evaluar")
            predict_btn.click(
                predict_fraud_customer,
                inputs=[
                    orderAmount,
                    orderState,
                    paymentMethodRegistrationFailure,
                    paymentMethodType,
                    paymentMethodProvider,
                    paymentMethodIssuer,
                    transactionAmount,
                    transactionFailed,
                    emailDomain,
                    emailProvider,
                    customerIPAddressSimplified,
                    sameCity,
                ],
                outputs = [label],
                api_name = "prediccion",
            )
    gr.Markdown(
        """
        <p style='text-align: center'>
            <a href='https://www.escueladedatosvivos.ai/cursos/bootcamp-de-data-science' 
                target='_blank'>Proyecto demo creado en el bootcamp de EDVAI 🤗
            </a>
        </p>
        """
    )

demo.launch()

