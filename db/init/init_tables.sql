CREATE TABLE IF NOT EXISTS diabetes_diagnosis_results(
    id INT(10),
    pregnancies INT(10),
    glucose INT(10),
    blood_pressure FLOAT(10),
    skin_thickness FLOAT(10),
    insulin FLOAT(10),
    bmi FLOAT(10),
    diabetes_pedigree_function FLOAT(10),
    age INT(10),
    outcome INT(10),
    is_trained BOOL,
    primary key (id)
) DEFAULT CHARSET=utf8 COLLATE=utf8_bin;

CREATE TABLE IF NOT EXISTS results_temp(
    id INT(10),
    pregnancies INT(10),
    glucose INT(10),
    blood_pressure INT(10),
    skin_thickness INT(10),
    insulin INT(10),
    bmi FLOAT(10),
    diabetes_pedigree_function FLOAT(10),
    age INT(10),
    predict_result INTEGER,
    predict_probability FLOAT(10),
    primary key (id)
) DEFAULT CHARSET=utf8 COLLATE=utf8_bin;

CREATE TABLE IF NOT EXISTS predict_results(
    id INT(10),
    predict_result INT(10),
    predict_probability FLOAT(10),
    true_result INT(10),
    model_id VARCHAR(64),
    primary key (id)
) DEFAULT CHARSET=utf8 COLLATE=utf8_bin;

LOAD DATA LOCAL
    infile '/docker-entrypoint-initdb.d/init_train_data.csv'
INTO TABLE
    diabetes_diagnosis_results
FIELDS
    terminated BY ','
    enclosed BY '"';

LOAD DATA LOCAL
    infile '/docker-entrypoint-initdb.d/init_input_data.csv'
INTO TABLE
    diabetes_diagnosis_results
FIELDS
    terminated BY ','
    enclosed BY '"';

LOAD DATA LOCAL
    infile '/docker-entrypoint-initdb.d/init_predict_result_data.csv'
INTO TABLE
    predict_results
FIELDS
    terminated BY ','
    enclosed BY '"';
