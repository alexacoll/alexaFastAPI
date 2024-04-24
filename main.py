
from fastapi import FastAPI,status,HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib

app = FastAPI(
    title="Deploy House Pricing",
    version="0.0.1"
)


# -------------------------------------------------
# Load AI Model
# -------------------------------------------------
model = joblib.load("model/lin_regression_V1.pkl")

@app.post("/api/v1/predict-house-pricing", tags=["House-Pricing"])
async def predict(
        LotArea: int,
        age: int,
        firstFlrSF: int,
        secondFlrSF: int,
        BedroomAbvGr: int,
        KitchenAbvGr: int,
        Fireplaces: int,
        GarageArea: int,
        RoofStyle_Flat: bool,
        RoofStyle_Gable: bool,
        RoofStyle_Gambrel: bool,
        RoofStyle_Hip: bool,
        RoofStyle_Mansard: bool,
        RoofStyle_Shed: bool,
        KitchenQual_Ex: bool,
        KitchenQual_Fa: bool,
        KitchenQual_Gd: bool,
        KitchenQual_TA: bool,
        GarageQual_Ex: bool,
        GarageQual_Fa: bool,
        GarageQual_Gd: bool,
        GarageQual_Po: bool,
        GarageQual_TA: bool,
        Heating_Floor: bool,
        Heating_GasA: bool,
        Heating_GasW: bool,
        Heating_Grav: bool,
        Heating_OthW: bool,
        Heating_Wall: bool,
        HeatingQC_Ex: bool,
        HeatingQC_Fa: bool,
        HeatingQC_Gd: bool,
        HeatingQC_Po: bool,
        HeatingQC_TA: bool
):
    dictionary = {
        'LotArea': LotArea,
        'age': age,
        'firstFlrSF': firstFlrSF,
        'secondFlrSF': secondFlrSF,
        'BedroomAbvGr': BedroomAbvGr,
        'KitchenAbvGr': KitchenAbvGr,
        'Fireplaces': Fireplaces,
        'GarageArea': GarageArea,
        'RoofStyle_Flat': RoofStyle_Flat,
        'RoofStyle_Gable': RoofStyle_Gable,
        'RoofStyle_Gambrel': RoofStyle_Gambrel,
        'RoofStyle_Hip': RoofStyle_Hip,
        'RoofStyle_Mansard': RoofStyle_Mansard,
        'RoofStyle_Shed': RoofStyle_Shed,
        'KitchenQual_Ex': KitchenQual_Ex,
        'KitchenQual_Fa': KitchenQual_Fa,
        'KitchenQual_Gd': KitchenQual_Gd,
        'KitchenQual_TA': KitchenQual_TA,
        'GarageQual_Ex': GarageQual_Ex,
        'GarageQual_Fa': GarageQual_Fa,
        'GarageQual_Gd': GarageQual_Gd,
        'GarageQual_Po': GarageQual_Po,
        'GarageQual_TA': GarageQual_TA,
        'Heating_Floor': Heating_Floor,
        'Heating_GasA': Heating_GasA,
        'Heating_GasW': Heating_GasW,
        'Heating_Grav': Heating_Grav,
        'Heating_OthW': Heating_OthW,
        'Heating_Wall': Heating_Wall,
        'HeatingQC_Ex': HeatingQC_Ex,
        'HeatingQC_Fa': HeatingQC_Fa,
        'HeatingQC_Gd': HeatingQC_Gd,
        'HeatingQC_Po': HeatingQC_Po,
        'HeatingQC_TA': HeatingQC_TA
    }

    try:
        df = pd.DataFrame(dictionary, index=[0])
        prediction = model.predict(df)
        return JSONResponse(
            content=prediction[0],
            status_code=status.HTTP_200_OK
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

