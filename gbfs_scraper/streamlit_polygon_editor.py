import os
import streamlit as st
from streamlit_folium import st_folium
import folium
from shapely.geometry import Point, Polygon

# Constants
CENTER_LAT = 40.7831  # central Manhattan latitude
CENTER_LON = -73.9712  # central Manhattan longitude


def generate_poly():
    m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=13)
    column1, column2 = st.columns(2)
    with column1:
        # Use the left map as a clicking area for adding points
        st_data = st_folium(m, width=700)
        # Provide a helpful quick debug/inspection view
        if "poly_coords" not in st.session_state:
            st.session_state["poly_coords"] = []

        # Sample polygon coordinates that can be loaded by the user
        SAMPLE_POLY = [
            (40.7845, -73.9750),
            (40.7900, -73.9700),
            (40.7880, -73.9590),
            (40.7780, -73.9570),
            (40.7720, -73.9640),
        ]

        # Add or load coordinates
        last_clicked = st_data.get("last_clicked") if st_data else None
        if last_clicked:
            lat = last_clicked["lat"]
            lon = last_clicked["lng"]
            st.write(f"Coordinates clicked: {lat:0.5f}, {lon:0.5f}")
            if st.button("Add Point"):
                st.session_state["poly_coords"].append((lat, lon))
                st.rerun()

        # Load a sample array directly when user checks the box
        if st.checkbox("Load sample polygon coordinates", value=False):
            st.session_state["poly_coords"] = SAMPLE_POLY.copy()
            st.success("Loaded sample polygon coordinates")
            st.rerun()

        # Convenience buttons: Undo last, Clear all
        cols = st.columns([1, 1])
        with cols[0]:
            if st.button("Undo last") and st.session_state["poly_coords"]:
                removed = st.session_state["poly_coords"].pop()
                st.success(f"Removed last point: {removed}")
                st.rerun()
        with cols[1]:
            if st.button("Clear all") and st.session_state["poly_coords"]:
                st.session_state["poly_coords"] = []
                st.rerun()
    if "poly_coords" in st.session_state:
        coords: list = st.session_state["poly_coords"]
        st.write("Current Polygon Coordinates:")

        # Show the coordinates in a clearer list with index-based delete UI
        if coords:
            formatted = [
                f"{i}: {lat:0.5f}, {lon:0.5f}" for i, (lat, lon) in enumerate(coords)
            ]
            st.write("\n".join(formatted))

            # Dropdown to select index to remove
            rem_idx = st.selectbox(
                "Select index to remove",
                options=list(range(len(coords))),
                format_func=lambda x: f"{x}: {coords[x][0]:0.5f}, {coords[x][1]:0.5f}",
            )
            if st.button("Remove selected point"):
                removed = st.session_state["poly_coords"].pop(rem_idx)
                st.success(f"Removed point {removed}")
                st.rerun()
        # Draw the polygon and markers on the right-hand map
        if column2:
            with column2:
                # Determine center; fallback to constants if event center not provided yet
                center = st_data.get("center") if isinstance(st_data, dict) else None
                if center:
                    map_center = (center["lat"], center["lng"])
                    zoom = st_data.get("zoom", 13)
                else:
                    map_center = (CENTER_LAT, CENTER_LON)
                    zoom = 13

                new_map = folium.Map(
                    location=[map_center[0], map_center[1]], zoom_start=zoom
                )

                # Add markers for each coordinate
                for i, (lat, lon) in enumerate(coords):
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=5,
                        color="blue",
                        fill=True,
                        fill_color="blue",
                        tooltip=f"{i}: {lat:0.5f}, {lon:0.5f}",
                    ).add_to(new_map)

                # Draw polygon if we have at least 3 points
                if len(coords) >= 3:
                    folium.Polygon(
                        locations=coords, color="orange", fill=True, fill_opacity=0.3
                    ).add_to(new_map)
                    polygon = Polygon([(lon, lat) for lat, lon in coords])

                st_folium(new_map, width=700, key="polygon_map")

                if st.button("Save Coordinates"):
                    st.write(coords)


def main():
    st.set_page_config(layout="wide", page_title="Polygon Editor (Manhattan)")

    st.title("Polygon editor for Manhattan stations")
    generate_poly()
    return


if __name__ == "__main__":
    main()
