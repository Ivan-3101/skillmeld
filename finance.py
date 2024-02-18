import streamlit as st
import numpy as np
import pickle
from data import finance_prospects_dict

# Load the trained model
loaded_model = pickle.load(open("finance.pkl", "rb"))

# Define the skills and their names
skills_names_finance = [
    "Ethical Conduct",
    "Leadership and Management",
    "Strategic Impact",
    "Communication and Presentation",
    "Analytical Skills",
    "Regulatory Compliance",
    "Risk Management",
    "Decision-Making Skills",
    "Continuous Learning",
    "Adaptability",
    "Financial Reporting",
    "Team Collaboration",
    "Client or Stakeholder Relationship Management",
    "Technology Proficiency",
    "Problem-Solving Skills",
]

# Title and description
st.title("Career Prediction App - Finance")

st.sidebar.info(
    "Use the sliders to input your skill levels (1-10). 1 means low, and 10 means high."
)

# Input for the user to enter skill levels
user_skills = []
for i, skill_name in enumerate(skills_names_finance):
    user_input = st.sidebar.slider(f"{skill_name} (1-10)", 1, 10, 5)
    user_skills.append(user_input)

# Button to make predictions
predict_button_clicked = st.button("Predict Roles")

# Condition to display the steps only when the button is clicked
if predict_button_clicked:
    # Convert user input to NumPy array
    input_data = np.array(user_skills).reshape(1, -1)

    # Make predictions
    predictions = loaded_model.predict(input_data)
    pred_proba = loaded_model.predict_proba(input_data)

    # Display predictions with increased and colorful font
    st.subheader("Predicted Roles:")
    predicted_roles = predictions[0]
    styled_predicted_roles = (
        f'<p style="font-size:24px; color:green;">{predicted_roles}</p>'
    )
    st.markdown(styled_predicted_roles, unsafe_allow_html=True)

    # Display job information for the first selected role
    first_role = predicted_roles
    st.subheader(f"Job Information for {first_role}:")
    job_info = finance_prospects_dict.get(first_role, {})
    if job_info:
        for section, content in job_info.items():
            st.subheader(section)
            if isinstance(content, list):
                for item in content:
                    st.write(item)
            else:
                st.write(content)
    else:
        st.write("Job information not available.")

    # Display prediction probabilities
    with st.expander("Prediction Probabilities"):
        # Display probabilities
        for i, prob in enumerate(pred_proba[0]):
            st.write(f"{loaded_model.classes_[i]}: {prob * 100:.2f}%")

        # Display additional information based on probability threshold
        threshold = 5  # You can adjust this threshold
        selected_roles = [
            loaded_model.classes_[i]
            for i, prob in enumerate(pred_proba[0])
            if prob > threshold / 100
        ]

        st.write("Debug Info:")
        st.write(f"Selected Roles: {selected_roles}")
        st.write(f"Finance Prospects Dict Keys: {finance_prospects_dict.keys()}")

        if selected_roles:
            st.subheader("Potential Roles (Probability > 5%):")
            for role in selected_roles:
                st.write(f"{role}")

# Add an "About" section at the bottom of the main content area
st.markdown("## About This Finance App")
st.info(
    "This web application predicts potential roles in the finance domain based on your inputted skill levels. "
    "Adjust the skill levels using the sliders, click the 'Predict Roles' button, "
    "and explore the predicted roles along with additional information and probabilities."
)

st.markdown("<div style='margin-bottom: 200px;'></div>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align: center; font-size: 125%; color: #777;'>"
    "<i>Made by the <span><strong>Your Team Name</strong></span>ALT+F4 🫅🫅🫅🫅</i></p>",
    unsafe_allow_html=True
)
