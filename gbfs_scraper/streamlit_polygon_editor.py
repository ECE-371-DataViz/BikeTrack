import os
from typing import List, Tuple

import streamlit as st
from streamlit_folium import st_folium
import folium
from shapely.geometry import Point, Polygon

# Constants
CENTER_LAT = 40.7831  # central Manhattan latitude
CENTER_LON = -73.9712  # central Manhattan longitude

def generate_poly():
    m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=13)
    st_data = st_folium(m)
    if st_data['last_clicked']:
        lat = st_data['last_clicked']['lat']
        lon = st_data['last_clicked']['lng']
        st.write(f"Coordinates {lat:0.2f}, {lon:0.2f}")    
        if st.button("Save Coordinate"):
            if 'poly_coords' in st.session_state:
                st.session_state['poly_coords'].append((lat, lon))
            else:
                st.session_state['poly_coords'] = [(lat, lon)]
    if 'poly_coords' in st.session_state:
        coords: list = st.session_state['poly_coords']
        st.write("Current Polygon Coordinates:")
        for coord in coords:
            st.write(f"{coord[0]:0.2f}, {coord[1]:0.2f}")
            if st.button(f"Remove point", key=coord):
                st.session_state['poly_coords'].remove(coord)
                st.rerun()
                st.success(f"Removed point {coord}")
        if len(coords) >= 3:
            folium.Polygon(locations=coords, color='orange', fill=True, fill_opacity=0.3).add_to(m)
            polygon = Polygon([(lon, lat) for lat, lon in coords])
            st.write(f"Polygon area: {polygon.area}")
            st_folium(m)
        if st.button("Save Coordinates"):
            st.write(coords)
def main():
    st.set_page_config(layout="wide", page_title="Polygon Editor (Manhattan)")

    st.title("Polygon editor for Manhattan stations")
    generate_poly()
    return

if __name__ == '__main__':
    main()
