import streamlit as st
import numpy as np
import pickle
from business_roles_info import job_info_dict


# Load the trained model
def business_page():
    loaded_model = pickle.load(open("business.pkl", "rb"))

    # Create a Streamlit app
    st.title("Career Prediction App - Business")

    # Sidebar with input sliders
    st.sidebar.title("Business Page - Skill Levels")
    st.sidebar.info("Use the sliders to input your skill levels (1-10).")

    # Input for the user to enter skill levels
    skills_names_business = [
        "Communication Skills",
        "Problem-Solving Skills",
        "Analytical Thinking",
        "Time Management",
        "Team Collaboration",
        "Adaptability",
        "Leadership and Management",
        "Decision-Making Skills",
        "Strategic Thinking",
        "Customer Focus",
        "Attention to Detail",
        "Project Management",
        "Data Analysis and Interpretation",
        "Technology Proficiency",
        "Ethical Conduct",
    ]

    user_skills_business = []
    for i, skill_name in enumerate(skills_names_business):
        user_input = st.sidebar.slider(f"{skill_name} (1-10)", 1, 10, 5)
        user_skills_business.append(user_input)

    # Button to make predictions
    predict_button_clicked_business = st.button("Predict Roles")

    # Condition to display the steps only when the button is not clicked
    if not predict_button_clicked_business:
        # Add steps to use the app
        st.header("How to Use the App:")
        st.write("1. Adjust skill levels using sliders in the sidebar.")
        st.write("2. Click the 'Predict Roles' button below.")
        st.write("3. View the predicted roles.")

    # Condition to display the steps only when the button is clicked
    if predict_button_clicked_business:
        # Convert user input to NumPy array
        input_data_business = np.array(user_skills_business).reshape(1, -1)

        # Make predictions
        predictions_business = loaded_model.predict(input_data_business)
        pred_proba_business = loaded_model.predict_proba(input_data_business)

        # Display predictions with increased and colorful font
        st.subheader("Predicted Roles:")
        predicted_role_business = predictions_business[0]
        styled_predicted_role_business = f'<p style="font-size:34px; color:green;"><b>{predicted_role_business}</b></p>'
        st.markdown(styled_predicted_role_business, unsafe_allow_html=True)

        # Display job information for the predicted role
        st.subheader("Job Information:")
        job_info = job_info_dict.get(predicted_role_business, {})
        if job_info:
            for key, value in job_info.items():
                st.write(f"**{key}:** {value}")
        else:
            st.write("Job information not available.")

        # Display prediction probabilities
        with st.expander("Prediction Probabilities"):
            # Display probabilities
            for i, prob in enumerate(pred_proba_business[0]):
                st.write(f"{loaded_model.classes_[i]}: {prob * 100:.2f}%")

            # Display additional information based on probability threshold
            threshold_business = 5  # You can adjust this threshold
            selected_roles_business = [
                loaded_model.classes_[i]
                for i, prob in enumerate(pred_proba_business[0])
                if prob > threshold_business / 100
            ]
            if selected_roles_business:
                st.subheader("Potential Roles (Probability > 5%):")
                st.write(selected_roles_business)
            else:
                st.subheader("No potential roles found.")


