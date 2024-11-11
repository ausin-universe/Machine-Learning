import streamlit as st 
import pandas as pd 


st.write("ML Hello Code")
st.write(pd.DataFrame({
    'first-column':[100.01,200.3,300.2,400.1],
    'second-column':[1,2,3,4]
}))

