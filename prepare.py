import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


"this function makes features to numerics and extracts features from other features"
class MakeNumericExtractAndDrop:

    def __init__(self, df, training_df):
        self.df = df
        self.training_df = training_df
    "here are the features we convert to numeric / boolean"
    def sex(self):
        self.df['sex'] = self.df['sex'].replace(['M'], 0)
        self.df['sex'] = self.df['sex'].replace(['F'], 1)

    def blood_type(self):
        self.df['A'] = self.df["blood_type"].isin(["A+", "A-"]).apply(lambda x: int(x))
        self.df['A'] = self.df["blood_type"].isin(["A+", "A-"]).apply(lambda x: int(x))
        self.df['B'] = self.df["blood_type"].isin(["B+", "B-", "AB+", "AB-"]).apply(lambda x: int(x))
        self.df['O'] = self.df["blood_type"].isin(["O+", "O-"]).apply(lambda x: int(x))
        self.df = self.df.drop('blood_type', 1)

    def current_location(self):
        self.df = self.df.drop('current_location', 1)

    def symptoms(self):
        self.df["low_appetite"] = self.df["symptoms"].str.contains("low_appetite").fillna(value = False).apply(lambda x: int(x))
        self.df["cough"] = self.df["symptoms"].str.contains("cough").fillna(value = False).apply(lambda x: int(x))
        self.df["shortness_of_breath"] = self.df["symptoms"].str.contains("shortness_of_breath").fillna(value = False).apply(lambda x: int(x))
        self.df["fever"] = self.df["symptoms"].str.contains("fever").fillna(value = False).apply(lambda x: int(x))
        self.df["sore_throat"] = self.df["symptoms"].str.contains("sore_throat").fillna(value = False).apply(lambda x: int(x))
        self.df = self.df.drop('symptoms', 1)

    def pcr_date(self):
        self.df = self.df.drop('pcr_date', 1)

    def patient_id(self):
        self.df = self.df.drop('patient_id', 1)

    def stdNormal(self, feature):
        df = self.df[[feature]]
        scaler = StandardScaler()
        scaler.fit(self.training_df[[feature]])
        normalized = scaler.transform(df)
        self.df[feature] = normalized

    def minMaxNormal(self, feature):
        df = self.df[[feature]]
        scaler = MinMaxScaler((-1,1))
        scaler.fit(self.training_df[[feature]])
        normalized = scaler.transform(df)
        self.df[feature] = normalized

    def normalization(self):
        self.minMaxNormal("num_of_siblings")
        self.stdNormal("weight")
        self.minMaxNormal("age")
        self.minMaxNormal("happiness_score")
        self.stdNormal("household_income")
        self.stdNormal("conversations_per_day")
        self.stdNormal("sugar_levels")
        self.minMaxNormal("sport_activity")
        for i in range(1, 10):
            if i != 6 and i != 8:
                self.minMaxNormal("PCR_0"+str(i))
        self.stdNormal("PCR_06")
        self.stdNormal("PCR_08")
        self.stdNormal("PCR_10")


    def numeric_change(self):
        self.blood_type()
        self.sex()
        self.pcr_date()
        self.symptoms()
        self.current_location()
        self.patient_id()

    def get_data(self):
        return self.df

def prepare_data(training_data, new_data):
    new_data_pro = MakeNumericExtractAndDrop(new_data.copy(), training_data.copy())
    new_data_pro.numeric_change()
    new_data_pro.normalization()
    new_data_to_return = new_data_pro.get_data()
    return new_data_to_return
