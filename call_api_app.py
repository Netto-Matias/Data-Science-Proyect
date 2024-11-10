import requests

# CASO 1 -> Tipo de fraude: 0/False
'''
response = requests.post("http://127.0.0.1:7860/run/prediccion", json={
        "data": [
            18.0,
            "pending",
            "True",
            "card",
            "JCB 16 digit",
            "Citizens First Banks",
            18,
            "False",
            "com",
            "yahoo",
            "only_letters",
            "yes"  
        ]
    }).json()
'''
# CASO 2 -> Tipo de fraude: 1/True   
response = requests.post("http://127.0.0.1:7860/run/prediccion", json={
        "data": [
            26.0,
            "fulfilled",
            "True",
            "bitcoin",
            "VISA 16 digit",
            "Solace Banks",
            26,
            "False",
            "com",
            "yahoo",
            "only_letters",
            "no"  
        ]
    }).json()
    
print(response)