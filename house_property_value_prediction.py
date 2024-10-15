import streamlit
import pickle
import sklearn

streamlit.title("House Property Value Prediction")
streamlit.write("Here we are trying to predict the value of house property using input features like land size, "
                "house size, number of rooms, number of bathrooms, large living room, parking space, front garden, swimming pool,"
                "wall fence and room size "
                )
streamlit.image(r"C:\Users\Albin Augustine\Downloads\how-to-add-value-to-your-house.jpg")
with open(r"C:\Users\Albin Augustine\Downloads\property_value_model","rb") as obj1:
    dict1=pickle.load(obj1)
streamlit.subheader('Enter your house property details. We will try to predict the house property value.')
land_size=streamlit.number_input('Land size(in sq m)')
house_size=streamlit.number_input('House size(in sq m)')
rooms=streamlit.selectbox("Number of room",range(1,11))
bathrooms=streamlit.slider('Number of Bathrooms',1,10)
large_living_room=streamlit.slider("Is there a large living room in the house? Select Yes (1) or No (0):", 0, 1, 0)
parking_space=streamlit.slider("Is there any parking space? Select Yes (1) or No (0):", 0, 1, 0)
front_garden=streamlit.slider("Is there any front garden? Select Yes (1) or No (0):", 0, 1, 0)
swimming_pool=streamlit.slider("Is there any swimming pool? Select Yes (1) or No (0):", 0, 1, 0)
wall_fence=streamlit.slider("Is there any wall fence? Select Yes (1) or No (0):", 0, 1, 0)
room_size=streamlit.slider("Select room size Small (0) or Medium (1) or Large(2) or Extra Large(3):", 0, 3, 0)
if streamlit.button('Predict'):
    data=[[land_size,house_size,rooms,bathrooms,large_living_room,parking_space,front_garden,swimming_pool,wall_fence,room_size]]
    scaled=dict1['scaler'].transform(data)
    res=dict1['model'].predict(scaled)[0]
    streamlit.subheader(f'Value predicted is {res}')
