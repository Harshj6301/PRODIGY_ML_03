import streamlit as st
from fastai.vision.all import *

def is_cat(x): 
    return x[0].isupper()

# Load the pre-trained model
model_path = "model.pkl"  # Replace with the path to your model file
learn = load_learner(model_path)

# Define categories
categories = ['Dog', 'Cat']

# Function to classify an image
def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

# Streamlit app
st.title("Dog vs. Cat Classifier")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "jfif", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Make predictions
    image = PILImage.create(uploaded_image)
    predictions = classify_image(image)

    # Display the prediction
    st.subheader("Prediction:")
    for category, probability in predictions.items():
        st.write(f"{category}: {probability:.2f}")

# Example images
st.sidebar.title("Example Images")
example_images = {
    "Dog": "dog.jfif",
    "Cat": "cat.jfif"
}

selected_example = st.sidebar.selectbox("Select an Example Image", list(example_images.keys()))

if selected_example:
    selected_image_path = example_images[selected_example]
    st.image(selected_image_path, caption=selected_example, use_column_width=True)

    # Make predictions for the example image
    example_image = PILImage.create(selected_image_path)
    example_predictions = classify_image(example_image)

    # Display the prediction
    st.sidebar.subheader("Prediction:")
    for category, probability in example_predictions.items():
        st.sidebar.write(f"{category}: {probability:.2f}")
