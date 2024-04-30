import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import plotly.express as px
import requests
import certifi
from pycaret.classification import load_model
from sklearn.preprocessing import LabelEncoder

# Function to fetch company description from Financial Modeling Prep API
def get_company_description(ticker):
    try:
        api_key = "5xCs51dRJdSOwpgzM6B4hbEatjvyyJh0"  # API key
        url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}.SI?apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        description = data[0].get('description', "Description not available")
        return description
    except Exception as e:
        print(f"Error fetching description for {ticker}: {e}")
        return "Description not available"

# Function to create donut chart
def make_donut(input_response, input_text, input_color):
    donut_data = pd.DataFrame({
        'Category': ['Low', 'Medium', 'High'],
        'Count': input_response  # Use the input_response directly
    })

    chart = alt.Chart(donut_data).mark_arc().encode(
        color=alt.Color('Category:N', scale=alt.Scale(domain=['Low', 'Medium', 'High'], range=[input_color[0], input_color[1], input_color[2]])),
        theta='Count:Q',
        tooltip=['Category', 'Count']
    ).properties(
        width=200,
        height=200
    ).transform_calculate(
        category='"Low" + datum.Category'  # Adding a prefix to avoid tooltip conflicts
    ).configure_view(
        stroke=None
    ).configure_title(
        fontSize=14
    ).configure_axis(
        labels=False
    )

    st.write(chart)
    
def main():
    # Path to the file containing company names and sectors
    company_data_file_path = "SGX_Screener_Data_24Apr2024.csv"  # Change this to the path of your file

    # Read company data from the file
    company_data = pd.read_csv(company_data_file_path)

    # Sort companies by sector
    sorted_companies = company_data.sort_values(by='Sector')

    # Group companies by sector
    grouped_companies = sorted_companies.groupby('Sector')

    # Get unique company names
    companies = sorted_companies['Company'].unique()  # Define the companies variable

    # Get unique sectors
    sectors = sorted_companies['Sector'].unique()  # Define the sectors variable

    # Requirement 1: Page configuration
    st.set_page_config(
        page_title="SGX Companies Insolvency Risk Assessment",
        page_icon="🏂",
        layout="wide",
        initial_sidebar_state="expanded")

    # Requirement 2: Add a sidebar
    with st.sidebar:
        st.title('🏂 SGX Companies Insolvency Risk Assessment')
        
        # Multiselect for selecting sectors
        selected_sectors = st.multiselect("Select Sectors", sectors)

        # Filter companies based on selected sectors
        if selected_sectors:
            filtered_companies = sorted_companies[sorted_companies['Sector'].isin(selected_sectors)]
            companies = filtered_companies['Company'].unique()

        # Multiselect for selecting companies
        selected_companies = st.multiselect("Select Companies", companies)

    # Requirement 3: Load Data
    data = pd.read_csv("SGX_Screener_Data_24Apr2024.csv")

    # Display selected companies information
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("Basic Financial Information")
        if selected_companies:
            # Initialize an empty list to store dictionaries of company information
            selected_info_list = []

            for index, company in enumerate(selected_companies, start=1):
                selected_company_info = data[data['Company'] == company].reset_index(drop=True)
                if not selected_company_info.empty:
                    # Round the 'Mkt Cap ($M)' column to 2 decimal places
                    selected_company_info['Mkt Cap ($M)'] = selected_company_info['Mkt Cap ($M)'].round(2)
                    # Convert the company information to a dictionary and append to the list
                    selected_info_list.append(selected_company_info[['Company', 'Mkt Cap ($M)', 'Tot. Rev ($M)', 'GTI Score', 
                                                                     'Net Profit %', 'Debt ($M)',
                                                                    'Watchlist Status_Encoded','Suspension Status_Encoded',
                                                                    'Price/CF_Risk_Score','Debt ($M)_Risk_Score','Debt/Equity_Risk_Score',
                                                                    'Mkt Cap ($M)_Risk_Score','Tot. Rev ($M)_Risk_Score']].to_dict('records')[0])
            
            # Convert the list of dictionaries into a DataFrame
            combined_info = pd.DataFrame(selected_info_list)
            
            # Increment index by 1 to start counting from 1 instead of 0
            combined_info.index += 1

            # Display the combined information as a single table
            st.write(combined_info[['Company', 'Mkt Cap ($M)', 'Tot. Rev ($M)', 'Debt ($M)', 'Net Profit %']])
        
            model = load_model("model")
	
            input_df = pd.DataFrame({'GTI Score': combined_info['GTI Score'],'Watchlist Status_Encoded': combined_info['Watchlist Status_Encoded'], 'Suspension Status_Encoded': combined_info['Suspension Status_Encoded'], 'Price/CF_Risk_Score': combined_info['Price/CF_Risk_Score'], 'Debt ($M)_Risk_Score': combined_info['Debt ($M)_Risk_Score'], 'Debt/Equity_Risk_Score': combined_info['Debt/Equity_Risk_Score'], 'Mkt Cap ($M)_Risk_Score': combined_info['Mkt Cap ($M)_Risk_Score'],'Tot. Rev ($M)_Risk_Score': combined_info['Tot. Rev ($M)_Risk_Score']})

            #st.write(input_df)	
            prediction = model.predict(input_df)

            #st.write(prediction)
            
            #For Loop 
            prediction_index = 0
            
            for index, row in combined_info.iterrows():
                company_name = row['Company']
                risk_label = ""
                if prediction[prediction_index] == 0:
                    risk_label = "Low Risk"
                    # Display company name and risk label above the donut
                    st.subheader(f"{company_name}: {risk_label}")
                    make_donut([1, 0, 0], "Risk Category", ["green", "yellow", "red"])
                elif prediction[prediction_index] == 1:
                    risk_label = "Medium Risk"
                    # Display company name and risk label above the donut
                    st.subheader(f"{company_name}: {risk_label}")
                    make_donut([0, 1, 0], "Risk Category", ["green", "yellow", "red"])
                else: 
                    risk_label = "High Risk"
                    # Display company name and risk label above the donut
                    st.subheader(f"{company_name}: {risk_label}")
                    make_donut([0, 0, 1], "Risk Category", ["green", "yellow", "red"])

                prediction_index += 1
                                
             # For Loop         	
            #features = ['Watchlist Status_Encoded','Suspension Status_Encoded','Price/CF_Risk_Score','Debt ($M)_Risk_Score','Debt/Equity_Risk_Score','Mkt Cap ($M)_Risk_Score','Tot. Rev ($M)_Risk_Score']
            #target = 'Insolvency Risk (0-2)'

        else:
            st.write("No companies selected.")
    

    with col2:      
        # Check if any companies are selected
        if selected_companies:
            # Keep track of sectors already processed
            processed_sectors = set()

            # Iterate over selected companies
            for company in selected_companies:
                # Find the sector of the selected company
                sector = data.loc[data['Company'] == company, 'Sector'].iloc[0]

                # Check if the sector has already been processed
                if sector in processed_sectors:
                    continue

                # Mark sector as processed
                processed_sectors.add(sector)

                st.subheader(sector)
                st.write(f"### Top 5 Ranking by {sector} <br><i><span style='font-size:20px; font-weight: normal;'>(Based on Total Revenue $Millions)</span></i>", unsafe_allow_html=True)

                companies_in_sector = grouped_companies.get_group(sector)

                # Get all selected companies within this sector
                selected_companies_in_sector = [c for c in selected_companies if data.loc[data['Company'] == c, 'Sector'].iloc[0] == sector]

                # Remove the selected companies from the sector companies if they're in the top 5
                if selected_companies_in_sector:
                    companies_in_sector = companies_in_sector[~companies_in_sector['Company'].isin(selected_companies_in_sector)]

                # Get the top 5 ranking companies in this sector
                top_5_ranking = companies_in_sector.nlargest(5, 'Tot. Rev ($M)')

                # Combine the selected companies with the top 5 ranking companies
                combined_companies = pd.concat([pd.DataFrame({'Company': [company], 
                                                              'Tot. Rev ($M)': [data.loc[data['Company'] == company, 'Tot. Rev ($M)'].iloc[0]], 
                                                              'Net Profit %': [data.loc[data['Company'] == company, 'Net Profit %'].iloc[0]]}) 
                                                 for company in selected_companies_in_sector] + [top_5_ranking])

                # Rank the combined companies based on 'Tot. Rev ($M)'
                combined_companies['Rank'] = combined_companies['Tot. Rev ($M)'].rank(ascending=False)

                # Sort the combined companies by rank
                combined_companies = combined_companies.sort_values(by='Rank')

                # Drop duplicates from the combined DataFrame
                combined_companies = combined_companies.drop_duplicates(subset=['Company'])

                # Reset index and set the 'Index' column starting from 1
                combined_companies.reset_index(drop=True, inplace=True)

                # Increment index by 1 to start counting from 1 instead of 0
                combined_companies.index += 1

                # Display the combined companies DataFrame
                st.write(combined_companies[['Company', 'Tot. Rev ($M)', 'Net Profit %']], index=False)
                st.write("\n")

        else:
            st.write("No companies selected.")


    with col3:
        st.subheader("Company Description")
        # Display descriptions for selected companies
        for company in selected_companies:
            st.write(f"### {company}")
            ticker = company_data[company_data['Company'] == company]['Ticker'].iloc[0]
            # Call the get_company_description function and display its output
            description = get_company_description(ticker)
            st.write(description)

if __name__ == "__main__":
    main()
