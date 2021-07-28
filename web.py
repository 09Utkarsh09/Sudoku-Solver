import streamlit as st
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image as im
import numpy as np
import imutils
import cv2 as cv
from sudoku import Sudoku
import webbrowser
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb

st.set_page_config(page_title="Sudoku Solver", page_icon="memo")
st.sidebar.title("Hello")
st.sidebar.text("Github Handle")
url_uday = 'https://github.com/udaykumarjangra/sudoku'
if st.sidebar.button("Uday's Github Link"):
    webbrowser.open_new_tab(url_uday)

url_utkarsh = 'https://github.com/09Utkarsh09/Sudoku-Solver'
if st.sidebar.button("Utkarsh's Github Link"):
    webbrowser.open_new_tab(url_utkarsh)

st.sidebar.button("Blog Link")

st.markdown("<h1 style='text-align: center; color: Black;'>Sudoku Solver</h1>", unsafe_allow_html=True)
st.text("")
img = st.file_uploader("Please upload an image file of Sudoku here", type = ["jpg","png"])

st.set_option('deprecation.showfileUploaderEncoding', False)


def boundaries(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (7,7), 3)
    threshold = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11,2)
    threshold = cv.bitwise_not(threshold)
    contour = cv.findContours(threshold.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)
    contour = sorted(contour, key = cv.contourArea, reverse=True)
    count = None
    
    for c in contour:
        length = cv.arcLength(c, True)
        
        approx = cv.approxPolyDP(c, 0.02 * length, True)
        
        if len(approx) == 4:
            count = approx
            break
        
    if count is None:
        print("No puzzle found")

    img = image.copy()
    cv.drawContours(img, [count], -1, (0,255,0),2)
    puzzle = four_point_transform(image, count.reshape(4,2))
    warped = four_point_transform(gray, count.reshape(4,2)) 
    return (puzzle, warped)

def extract(cell):
    threshold = cv.threshold(cell, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    threshold = clear_border(threshold)
    contour = cv.findContours(threshold.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour = imutils.grab_contours(contour)
    img = cell.copy()
    cv.drawContours(img, contour, -1, (0,255,0),2)
    if len(contour) == 0:
        return None
    
    c = max(contour, key = cv.contourArea)
    mask = np.zeros(threshold.shape).astype("int8")
    cv.drawContours(mask, [c], -1, 255, -1)
    (height, width) = threshold.shape
    filled = cv.countNonZero(mask)/float(height*width)
    
    if filled < 0.03:
        return None
    
    digit = cv.bitwise_and(threshold, threshold, mask=mask)
    return digit

def import_and_predict(image_data):
    digit_classifier = load_model("digit_classifier")
    (color, grey) = boundaries(img)
    board = np.zeros((9,9), dtype = "int")
    X_size = grey.shape[1] // 9
    y_size = grey.shape[0] // 9

    cells = []

    for y in range(0,9):
        row = []
        for x in range(0,9):
            startX = x * X_size
            startY = y * y_size
            endX = (x+1) * X_size
            endY = (y+1) * y_size
            row.append((startX, startY, endX, endY))
            cell = grey[startY:endY, startX:endX]
            digit = extract(cell)
            
            if digit is not None:
                area = cv.resize(digit, (28,28))
                area = area.astype("float") / 255.0
                area = img_to_array(area)
                area = np.expand_dims(area, axis = 0)
                prediction = digit_classifier.predict(area).argmax(axis=1)[0]
                print(prediction)
                board[y,x] = prediction
        cells.append(row)
    solution = Sudoku(3,3, board = board.tolist())
    solution = solution.solve()
    print(solution.board)
    if (solution.board[0][0]==None):
        return None
    print(board)
    for (cell, sol) in zip(cells, solution.board):
        for(box, digit) in zip(cell, sol):
            X1, Y1, X2, Y2 = box
            X = int((X2-X1) * 0.33)
            Y = int((Y2-Y1) * -0.2)
            X += X1
            Y += Y2
            print(digit)
            cv.putText(color, str(digit), (X,Y), cv.FONT_HERSHEY_SIMPLEX,0.9, (0,142,255),2)
    return color

if img is None:
    st.text("No image uploaded")
else:
    img = im.open(img)
    img = np.asarray(img)
    img = imutils.resize(img, width = 600)
    prediction = import_and_predict(img)
    if prediction is not None:
        st.text("Solution of the Given Sudoku is:")
        st.image(prediction)
    else:
        st.text("No solution found")


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 105px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        "Made by",
        link("https://github.com/udaykumarjangra/sudoku", "@Uday-CO18354"),
        " and ",
        link("https://github.com/09Utkarsh09/Sudoku-Solver", "@Utkarsh-CO18356"),
    ]
    layout(*myargs)


if __name__ == "__main__":
    footer()