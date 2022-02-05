import app

if __name__ == "__main__":
    app.predict(
        n_clicks=1,
        id=2732,
        pregnancies=2,
        glucose=108,
        blood_pressure=80,
        skinthickness=0,
        insulin=0,
        bmi=32.96959829,
        diabetes_pedigree_function=0.241813673,
        age=22,
    )
    app.register(n_clicks=1, index=2732, is_prediction_true="yes")
