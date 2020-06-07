from io import BytesIO

from django import db
from six.moves import urllib
from keras.preprocessing import image
from tensorflow.keras.models import model_from_json
from keras.preprocessing.image import load_img
from keras import backend as k
import numpy as np

from apiSNN import models


def predecirImg(url_imagen):
    TAM_IMG = (150, 150)

    url_modelo = r'apiSNN/Logica/modelo.json'
    url_pesos = r'apiSNN/Logica/pesos.h5'

    modelo = cargar_modelo(url_modelo, url_pesos)

    img = load_img(url_imagen, target_size=TAM_IMG)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255
    print(img)
    resultado = modelo.predict(img)
    rst = None
    for i in resultado:
        rst = i
    rst*=100

    prd = max(rst)
    inx, = np.where(np.isclose(rst, prd))

    if int(inx[0]) == 0:
        #nombre='Daisy'
        #dbReg=models.Image(1,img,label=nombre,probability=prd)
        #dbReg.save()
        return 'La imagen es: Daisy', prd
    elif int(inx[0] == 1):
        return 'La imagen es: Diente De Leon', prd
    elif int(inx[0] == 2):
        #nombre='Girasol'
        #dbReg=models.Image(1,img.all(),label=nombre,probability=prd)
        #dbReg.save()
        return ' La imagen es: Girasol', prd
    elif int(inx[0] == 3):
        return 'La imagen es: Tulipan', prd # <-- CUIDADO
    elif int(inx[0] == 4):
        return 'La image es: Rosa', prd



def cargar_modelo(url_modelo, url_pesos):
    k.reset_uids()
    with open(url_modelo, 'r') as f:
        print('INTENTA LEER <<___ ' * 5)
        model = model_from_json(f.read())
        print('FINALIZA EL LEER <----' * 10)
    # Cargar Pesos (weights) en el nuevo modelo.json
    model.load_weights(url_pesos)
    print("Red Neuronal Cargada desde Archivo")
    return model
