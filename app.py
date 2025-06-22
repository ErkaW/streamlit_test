import streamlit as st
from generate_digits import generate_digit_images

st.set_page_config(page_title="MNIST Digit Generator", layout="centered")
st.title("🖍️ Handwritten Digit Generator")
st.markdown("Generate handwritten digit images (0–9) using a CGAN trained from scratch on MNIST.")

digit = st.number_input("Enter a digit (0–9):", min_value=0, max_value=9, step=1)

if st.button("Generate"):
    st.success(f"Generated images for digit {digit}")
    image = generate_digit_images(digit)
    st.image(image, caption=f"Generated Digit {digit}", use_column_width=True)
