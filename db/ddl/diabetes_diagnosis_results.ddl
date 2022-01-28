--DROP TABLE IF EXISTS diabetes_diagnosis_results;
CREATE TABLE diabetes_diagnosis_results(
    index INTEGER,
    pregnancies INTEGER,
    glucose INTEGER,
    blood_pressure NUMERIC,
    skin_thickness NUMERIC,
    insulin NUMERIC,
    bmi NUMERIC,
    diabetes_pedigree_function NUMERIC,
    age INTEGER,
    outcome INTEGER,
    is_trained BOOLEAN,
    primary key (index)
);