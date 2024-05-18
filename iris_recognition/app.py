import numpy as np
import pandas as pd
import os, cv2
import matplotlib.pyplot as plt
import EI as ei
import streamlit as st
from PIL import Image
import time

# Lien de dataset : https://www.kaggle.com/datasets/anisguechtouli/extraction/data



st.set_page_config("IRIS",layout="wide", page_icon="iris_icon.png")

col1, col2 = st.columns([1,1],gap="large")
with col1.container(height=500,border=True).form("zero_one",border=False):
    with st.container(height=400,border=False):
        img = st.file_uploader("Uplaod an iris image", type='png')
        raw_img = img
        if img is not None:
            #preprocessing
            min_height, min_width = ei.min_HW()
            st.write("Calculating minimum height and width for resizing... :heavy_check_mark:")

            if 'df' not in st.session_state:
                st.session_state["df"] = ei.extract_from_images(min_height,min_width)

            st.write("Extracting features from iris images... :heavy_check_mark:")
            
            img_data = img.read()
            img_array = np.asarray(bytearray(img_data), dtype=np.uint8)
            img_opencv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img_resized = cv2.resize(img_opencv, (min_width, min_height))

            gray_scaled, circle_info = ei.preprocessing(img_resized, min_height, min_width)
            if gray_scaled is not None and circle_info is not None:
                keypoints, features, filtered_keypoints, image = ei.sift_extract(gray_scaled, circle_info)
                st.write("Extracting features from uploaded image... :heavy_check_mark:")

                st.session_state['df'] = st.session_state['df'].sort_values(by=["Person_ID", "Eye","Image_Number"]).reset_index(drop=True)
                df_ni = st.session_state["df"].copy()
                df_ni = df_ni.sort_values(by=["Person_ID", "Eye","Image_Number"]).reset_index(drop=True)

                df_ni = df_ni.drop("img", axis=1)
                comparison_results = ei.compare_features_given_img(st.session_state['df'],features, keypoints,"given")
                comparison_results = comparison_results[comparison_results["Matching_Rate"]<100]
                comparison_results = comparison_results.sort_values(by="Matching_Rate", ascending=False).head()
                st.write("Comparing features and looking for matches... :heavy_check_mark:")

                MR = comparison_results.iloc[0]['Matching_Rate']
                if MR < 44:
                    matches = 0
                else:
                    matches = 1
                    fox = comparison_results['Other_Index'].iloc[0]
                    Pers = df_ni.iloc[fox]['Person_ID']
                    E = df_ni.iloc[fox]['Eye']
                    oi = comparison_results['Other_Index'].iloc[0]
            else:
                st.error('Image uploaded is invalid')
    zo_submit = st.form_submit_button("Start the identification process.",use_container_width=True)

with col2.container(height=500,border=True):
        st.write("Matching results shown here")
        if img is not None and gray_scaled is not None and circle_info is not None:
            col1, col2 = st.columns(2)
            if matches == 1:
                with col1:
                    st.image(image, caption='Uploaded Image', use_column_width=True)
                with col2:
                    st.image(st.session_state.df["img"].iloc[oi], caption='Matching image from dataset', use_column_width=True)

                merged_df = pd.merge(comparison_results, df_ni, left_on='Other_Index', right_index=True, how='left')
                
                merged_df.drop(columns=['Compare_Index','Other_Index', 'Features','Keypoints','fk'], inplace=True)
                merged_df = merged_df.head(1)
                st.dataframe(merged_df,use_container_width=True)

                a = merged_df['Person_ID'].iloc[0]+merged_df['Eye'].iloc[0]+'_'+merged_df['Image_Number'].iloc[0]+'.png'
            else:
                st.error("No matches from the dataset, user does not exist")
