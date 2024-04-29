import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import plotly.express as px
import requests
import certifi
from urllib.request import urlopen
from pycaret.classification import load_model
from sklearn.preprocessing import LabelEncoder

# Function to fetch company description from Financial Modeling Prep API
def get_company_description(ticker):
    try:
        api_key = "5xCs51dRJdSOwpgzM6B4hbEatjvyyJh0"  # API key
        #url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}.SI?apikey={api_key}"
        response = requests.get(url)
        data = response.json()
        description = data[0].get('description', "Description not available")
        return description
    except Exception as e:
        print(f"Error fetching description for {ticker}: {e}")
        return "Description not available"

def main():
    # Path to the file containing company names and sectors
    company_data_file_path = "/content/drive/MyDrive/SGX_Screener_Data_24Apr2024.csv"  # Change this to the path of your file

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
    data = pd.read_csv("/content/drive/MyDrive/SGX_Screener_Data_24Apr2024.csv")

    # Display selected companies information
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("Company Information")
        if selected_companies:
            st.markdown("<span style='font-size:24px;'>Selected Companies</span><br><i><span style='font-size:12px;'>(Based on Total Revenue $Millions)</span></i>", unsafe_allow_html=True)

            # Initialize an empty list to store dictionaries of company information
            selected_info_list = []

            for index, company in enumerate(selected_companies, start=1):
                selected_company_info = data[data['Company'] == company].reset_index(drop=True)
                if not selected_company_info.empty:
                    # Round the 'Mkt Cap ($M)' column to 2 decimal places
                    selected_company_info['Mkt Cap ($M)'] = selected_company_info['Mkt Cap ($M)'].round(2)
                    # Convert the company information to a dictionary and append to the list
                    selected_info_list.append(selected_company_info[['Company', 'Mkt Cap ($M)', 'Tot. Rev ($M)', 'Debt ($M)',
                                                                    'Watchlist Status_Encoded','Suspension Status_Encoded',
                                                                    'Price/CF_Risk_Score','Debt ($M)_Risk_Score','Debt/Equity_Risk_Score',
                                                                    'Mkt Cap ($M)_Risk_Score','Tot. Rev ($M)_Risk_Score']].to_dict('records')[0])
            # Convert the list of dictionaries into a DataFrame
            combined_info = pd.DataFrame(selected_info_list)
            
            # Increment index by 1 to start counting from 1 instead of 0
            combined_info.index += 1

            # Display the combined information as a single table
            st.write(combined_info[['Company', 'Mkt Cap ($M)', 'Tot. Rev ($M)', 'Debt ($M)']])
        else:
            st.write("No companies selected.")

        load_model("/content/drive/MyDrive/model")

        #features = ['Watchlist Status_Encoded','Suspension Status_Encoded','Price/CF_Risk_Score','Debt ($M)_Risk_Score','Debt/Equity_Risk_Score','Mkt Cap ($M)_Risk_Score','Tot. Rev ($M)_Risk_Score']
        #target = 'Insolvency Risk (0-2)'

    with col2:
        st.subheader("Industry Top Ranking")
        
        # Initialize an empty DataFrame to store the top 5 ranking for each sector
        top_5_sector_ranking = pd.DataFrame(columns=['Company', 'Tot. Rev ($M)'])
        
        # Check if any companies are selected
        if selected_companies:
            # Iterate over selected companies
            for company in selected_companies:
                # Find the sector of the selected company
                sector = data.loc[data['Company'] == company, 'Sector'].iloc[0]
                
                # Filter companies in the same sector
                companies_in_sector = data[data['Sector'] == sector]
                
                # Remove the selected company from the sector companies if it's in the top 5
                if company in top_5_sector_ranking['Company'].values:
                    companies_in_sector = companies_in_sector[companies_in_sector['Company'] != company]
                
                # Combine the selected company with the top 5 companies in the sector
                combined_companies = pd.concat([pd.DataFrame({'Company': [company], 'Tot. Rev ($M)': [data.loc[data['Company'] == company, 'Tot. Rev ($M)'].iloc[0]]}), 
                                                companies_in_sector.nlargest(5, 'Tot. Rev ($M)')])
                
                # Rank the combined companies based on 'Tot. Rev ($M)'
                combined_companies['Rank'] = combined_companies['Tot. Rev ($M)'].rank(ascending=False)
                
                # Sort the combined companies by rank
                combined_companies = combined_companies.sort_values(by='Rank')
                
                # Append the top 5 ranking for this sector to the overall top 5 sector ranking DataFrame
                top_5_sector_ranking = pd.concat([top_5_sector_ranking, combined_companies[['Company', 'Tot. Rev ($M)']]])
                
                # Sort the top 5 sector ranking DataFrame based on 'Tot. Rev ($M)'
                top_5_sector_ranking = top_5_sector_ranking.sort_values(by='Tot. Rev ($M)', ascending=False)
            
            # Drop duplicates from the top 5 sector ranking DataFrame
            top_5_sector_ranking = top_5_sector_ranking.drop_duplicates(subset=['Company'])
            
            # Display the top 5 ranking companies for each sector without the index column
            st.write(top_5_sector_ranking[['Company', 'Tot. Rev ($M)']], index=False)
        else:
            st.write("No companies selected.")

    with col3:
        st.subheader("Company Descriptions")
        # Display descriptions for selected companies
        for company in selected_companies:
            st.write(f"### {company}")
            ticker = company_data[company_data['Company'] == company]['Ticker'].iloc[0]
            # Call the get_company_description function and display its output
            description = get_company_description(ticker)
            st.write(description)

if __name__ == "__main__":
    main()
