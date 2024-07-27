import streamlit as st

# this function is used to add custom CSS styles to the web pages
# the styles are defined in the css variable and added to the web page using the st.markdown function
# to use the styles, simply call the css_style() function at the beginning of your script
def css_style():
    css = ""
    css += """
        <!--CSS for modifying the appearance of search results titles-->
        <style>
        .search-results {
            font-size:35px !important;
            font-weight:bold !important;
        }
        </style>
        """
    css += """
        <!--CSS for customizing the appearance of links within search results-->
        <style>
        .custom-link {
            color: #FF9933 !important;
            text-decoration: none;
            font-size: 42px;
        }
        </style>
        """
    css += """
        <!--CSS for highlighting keywords within text-->
        <style>
        .highlight-keyword {
            font-weight: bold;
            font-style: italic;
            <!--color: #0078D7;-->
        }
        </style>
        """
    css += """
        <!--CSS for adjusting the font size of descriptions in search results-->
        <style>
        .description {
            font-size: 18px !important;
        }
        </style>
        """
    css += """
        <!--CSS for styling the button used to extract filters-->
        <style>
        .extract-button {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            justify-content: center;
        }
        """
    css += """
        </style>
        <!--CSS to make any text with the class bold-->
        <style>
        .bold-text {
            font-weight: bold !important;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)