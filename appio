import streamlit as st
import cv2
import numpy as np
import sympy as sp
from skimage import measure
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges

def get_contours(edges):
    contours = measure.find_contours(edges, 0.5)
    return contours

def polynomial_fit(x, y, degree=5):
    coeffs = np.polyfit(x, y, degree)
    poly_eq = np.poly1d(coeffs)
    return poly_eq, coeffs

def generate_equation(coeffs):
    x = sp.Symbol('x')
    equation = sum(c * x**i for i, c in enumerate(reversed(coeffs)))
    return equation

def main():
    st.title("Image to Mathematical Equation")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        edges = edge_detection(image)
        st.image(edges, caption="Edge Detection", use_column_width=True, channels="GRAY")
        
        contours = get_contours(edges)
        if contours:
            contour = max(contours, key=len)  # Pick the largest contour
            x, y = contour[:, 1], contour[:, 0]  # Swap axes
            
            poly_eq, coeffs = polynomial_fit(x, y)
            equation = generate_equation(coeffs)
            
            st.write("Generated Mathematical Equation:")
            st.latex(sp.latex(equation))
            
            fig, ax = plt.subplots()
            ax.plot(x, y, label='Original Shape')
            ax.plot(x, poly_eq(x), linestyle='dashed', label='Fitted Curve')
            ax.legend()
            st.pyplot(fig)
            
            st.write("Copy this equation into Desmos for visualization!")

if __name__ == "__main__":
    main()
