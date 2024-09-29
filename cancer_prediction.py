import streamlit as st
import pandas as pd
import numpy as np
import pickle
from model import main  
import plotly.graph_objects as go
from model.main import get_clean_data
from model.main import columns_max_min
from model.main import NeuralNet
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )


def add_sidebar():

    columns = columns_max_min()
    slider_label = []
    categories = []
    # list(set(categories))
    for col, value in columns.items():
        split = col.split("_")
        firstpart = " ".join(split[:-1])
        categories.append(firstpart)
        slider_name = str(firstpart + " ("+split[-1]+")")
        label = (slider_name, col)
        slider_label.append(label)
    input_dict = {}
    for label, key in slider_label:
        
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(columns[key][0]),
            value=float(columns[key][1])
        )
    # st.write(input_dict)    
    # list(categories) = set(categories)
    return input_dict, categories

def get_scaled_values( input_dict):
    data = get_clean_data()
    indep = data.drop(columns=['diagnosis'], axis= 1)

    scaled_dict = {}

    for key,value in input_dict.items():
        max_value = indep[key].max()
        min_value = indep[key].min()
        scaled_value = (value-min_value)/(max_value-min_value)
        scaled_dict[key] = scaled_value

    return scaled_dict

def get_radar_chart(input_data, categories):
    input_data = get_scaled_values(input_data)
    categories = list(set(categories))
    cate = []
    for cat in categories:
        cate.append(cat.title())

    categories = cate

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
           input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
           input_data['concavity_mean'], input_data['concave points_mean'],
           input_data['symmetry_mean'], input_data['fractal_dimension_mean']],
        theta=categories,
        fill='toself',
        name='Mean'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'],
           input_data['area_se'], input_data['smoothness_se'], input_data['compactness_se'],
           input_data['concavity_se'], input_data['concave points_se'],
           input_data['symmetry_se'], input_data['fractal_dimension_se']],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
           input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
           input_data['concavity_worst'], input_data['concave points_worst'],
           input_data['symmetry_worst'], input_data['fractal_dimension_worst']],
        theta=categories,
        fill='toself',
        name='Worst'
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )

    return fig

def add_prediction(input_data):
    model = pickle.load(open("resources/model.pkl", "rb"))
    scaler = pickle.load(open("resources/scaler.pkl", "rb"))
    input_array = np.array(list(input_data.values())).reshape(1,-1)
    X = scaler.transform(input_array)
    X_tran = torch.tensor(X, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(X_tran)
        predicted = outputs.round()
    if predicted.item() == 0.0:
        st.write("<span class='diagnosis benign_cls'>Benign</span>",unsafe_allow_html=True)
        st.write("The probability of being benign: ","\n", round(1.00 - outputs.item(),2))
        st.write("The probability of being malignanat: ","\n", round(outputs.item(),2))
    else:
        st.write("<span class='diagnosis malignant_cls'>Malignant</span>",unsafe_allow_html=True)
        st.write("The probability of being benign: ", round(1.00 - outputs.item(),2))
        st.write("The probability of being malignanat: ", round(outputs.item(),2))
    

def main():

    st.set_page_config(
        page_title= "Breast Cancer Prediction",
        page_icon= ":doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True) 

    input_data, categories = add_sidebar()
    # st.write(input_data)
    # st.write("Categories:")
    # st.write(set(categories))
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("This app predicts using a neural network model whether the breast mass is benign or malignant based on the measurements. \n",
                 "You can update the measurements using the slideers in the sidebar.")

        col1, col2= st.columns([3,1])

        with col1:
            fig_chart = get_radar_chart(input_data, categories)
            st.plotly_chart(fig_chart)
            # st.divider()
        with col2:
            
            st.subheader("Cell Cluster Prediction")
            add_prediction(input_data)

        # with col3:
        #     st.write("this is third col")




if __name__ == '__main__':
    main()