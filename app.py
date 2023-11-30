import streamlit as st
import main
import asyncio
import redirect as rd
import os

st.set_page_config(page_title="Techvocate")

hide_github_icon = """
GithubIcon {
  visibility: hidden;
}
"""
# st.markdown(hide_github_icon, unsafe_allow_html=True)

def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

def run(query):
    if query:
        with rd.stdout() as out:
            # ox = main.preprocessing().run(query)
            ox = main.to_run(query) 
        output = out.getvalue()
        output = main.remove_formatting(output)
        with st.spinner("Generating..."):
            st.write(ox.response)
        return True


st.title("Legalease\nLegal Documentation Assistant")

option = st.selectbox("Select Type of Legal Document", ["None","Rent Agreement( Only Uttar Pradesh )", "Business Agreement", "Sale Deed", "Other"])

# Function to display page based on selected option
def display_page_0():
    st.header("Generative AI powered solution for generating legal documents related to daily life of a common person. Only related to some legal aspects, just for prototype purpose")

def display_page_1():
    st.subheader("Form for Agreement Details")
    with st.form("Agreement Form"):
        landlord_name = st.text_input("Landlord Name")
        tenant_name = st.text_input("Tenant Name")
        duration = st.text_input("Duration of Agreement (Specify years/months)")
        amount = st.text_input("Amount")
        terms_conditions = st.text_area("Conditions")
        document_date = st.date_input("Agreement Date", value=None)
        query = ""
        if st.form_submit_button("Submit"):
            query = (
                f"Draft a Rent Agreement between {landlord_name} (hereinafter referred to as 'Landlord') "
                f"and {tenant_name} (hereinafter referred to as 'Tenant') of date {document_date}. The agreement's duration "
                f"is {duration} and the monthly rent amount is {amount}. Other conditions for the agreement are: '{terms_conditions}'."
            )
            st.write("Generated Query:")
            st.write(query)
            run(query)


# Function for Page 2
def display_page_2():
    st.subheader("Form for Business Agreement")
    with st.form("Business Agreement Form"):
        company_1_name = st.text_input("Company 1 Name")
        company_2_name = st.text_input("Company 2 Name")
        services_provided = st.text_input("Services Provided")
        duration = st.text_input("Duration of Agreement")
        terms_conditions = st.text_area("Conditions")
        document_date = st.date_input("Agreement Date", value=None)
        query = ""
        if st.form_submit_button("Submit"):
            query = (
                f"Draft a Business Agreement between {company_1_name} and {company_2_name} of date {document_date}. "
                f"The agreement pertains to the provision of {services_provided} for a duration of {duration}. "
                f"The agreement includes the following conditions: '{terms_conditions}'."
            )
            st.write("Generated Query:")
            st.write(query)
            run(query)

# Function for Page 3
def display_page_3():
    st.subheader("Form for Sale Deed")
    with st.form("Sale Deed Form"):
        seller_name = st.text_input("Seller's Name")
        buyer_name = st.text_input("Buyer's Name")
        property_details = st.text_area("Property Details")
        sale_price = st.text_input("Sale Price")
        terms_conditions = st.text_area("Conditions")
        document_date = st.date_input("Date of Sale", value=None)
        query = ""
        if st.form_submit_button("Generate Sale Deed"):
            query = (
                f"Create a Sale Deed between {seller_name} (hereinafter referred to as 'Seller') "
                f"and {buyer_name} (hereinafter referred to as 'Buyer') of date {document_date}. The property being sold has the following details: '{property_details}'. "
                f"The sale price is {sale_price}. The Sale Deed incorporates the following conditions: '{terms_conditions}'."
            )
            st.write("Generated Sale Deed:")
            st.write(query)
            run(query)

# Function for Page 4
def display_page_4():
    st.subheader("Form for Generic Document")
    document_title = st.text_input("Document Title")
    document_type = st.selectbox("Select Document Type", [
        "Agreement",
        "Deed",
        "Contract",
        "Letter",
        "Other"
    ])
    document_details = st.text_area("Enter Document Details")
    document_date = st.date_input("Document Date", value=None)
    
    if st.button("Generate Document"):
        query = f"Draft a {document_type} (document type) titled {document_title} of Date {document_date}. The details about the document are '{document_details}'"
        st.write("Generated Document:")
        st.write(query)
        run(query)

# Conditionally display pages based on selected option
if option=="None":
    display_page_0()
elif option == "Rent Agreement( Only Uttar Pradesh )":
    display_page_1()
elif option == "Business Agreement":
    display_page_2()
elif option == "Sale Deed":
    display_page_3()
elif option == "Other":
    display_page_4()