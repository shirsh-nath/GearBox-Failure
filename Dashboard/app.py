import joblib
import streamlit as st
import pandas as pd
from scipy.stats import kurtosis,skew

scaler=joblib.load("scaler.pkl")
final_columns=joblib.load("columns.pkl")
model=joblib.load("model.pkl")

st.title("Fault Diagnosis")

if "readings" not in st.session_state:
    st.session_state.readings=[]
l=st.session_state.readings

st.subheader("Please provide the load and following continuous signals of accelerometer")

load=st.number_input("Enter load percentage:",min_value=0,max_value=95,value=0)

a1=st.number_input("Enter acceleration reading of accelerometer 1:",min_value=-40.0,max_value=40.0,value=0.1)
a2=st.number_input("Enter acceleration reading of accelerometer 2:",min_value=-40.0,max_value=40.0,value=0.1)
a3=st.number_input("Enter acceleration reading of accelerometer 3:",min_value=-40.0,max_value=40.0,value=0.1)
a4=st.number_input("Enter acceleration reading of accelerometer 4:",min_value=-40.0,max_value=40.0,value=0.1)

st.write("Make sure that all data rows are added before submitting.")

add_row=st.button("Add sensor reading")
submit=st.button("Submit")
clear_rows=st.button("Clear Row History")

if clear_rows:
    l.clear()

if add_row:
    l.append({"a1":a1,
              "a2":a2,
              "a3":a3,
              "a4":a4,
              "load":load})
    st.dataframe(pd.DataFrame(l))
    
if submit:

    df_init=pd.DataFrame(l)
    df_init=pd.read_csv("D:\\Applications\\Python\\Practice\\Hello World\\Scratch\\GearBox Failure\\Dashboard\\test_df.csv")
    df=pd.DataFrame()
    acc=["a1","a2","a3","a4"]
    window=100

    for col in acc:
        df[f"{col}_std"]=df_init[col].groupby(df_init.index//window).std()
    for col in acc:
        df[f"{col}_min"]=df_init[col].groupby(df_init.index//window).min()
    for col in acc:
        df[f"{col}_max"]=df_init[col].groupby(df_init.index//window).max()
    for col in acc:
        df[f"{col}_var"]=df_init[col].groupby(df_init.index//window).var()
    for col in acc:
        df[f"{col}_range"]=df_init[col].groupby(df_init.index//window).agg(lambda x:(x.max()-x.min()))
    for col in acc:
        df[f"{col}_kurtosis"]=df_init[col].groupby(df_init.index//window).agg(lambda x:kurtosis(x))
    for col in acc:
        df[f"{col}_skew"]=df_init[col].groupby(df_init.index//window).agg(lambda x:skew(x))
    df["load"]=df_init["load"].groupby(df_init.index//window).agg(lambda x:x.mode().iloc[0])

    df=df.fillna(0)
    df=df.reindex(columns=final_columns,fill_value=0)
    df=scaler.transform(df)

    y_pred=model.predict(df)
    st.write(y_pred)
    if(1 in y_pred):
        st.error("❌ Risk of Failure!")
    else:
        st.success("✅ GearBox is working fine")
    l.clear()