#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 15:38:55 2026

@author: sandeepbhatt
"""

# %% import modules
import pandas as pd
import numpy as np
from datetime import datetime, timedelta,time
import matplotlib
import matplotlib.pyplot as plt
import joblib
from herbie import Herbie,FastHerbie
from herbie import HerbieLatest, HerbieWait
import xarray as xr
import warnings
import streamlit as st
import plotly.express as px
import folium
from streamlit_folium import st_folium
from streamlit_folium import folium_static
import pytz 
#%% Backend function 
MODEL_PATH = 'frost_model_010626.joblib'

def run_forecast(BOG_LAT, BOG_LON,current_run_time = None):
    points_df = pd.DataFrame({
        "latitude": [BOG_LAT],
        "longitude": [BOG_LON]})
    package = joblib.load(MODEL_PATH)
    model = package['model']
    scaler = package['scaler']
    # FETCH LIVE WEATHER DATA (HRRR Model)
    
    
    # now_utc = datetime.utcnow()
    # target_utc = now_utc.replace(hour=10, minute=0, second=0, microsecond=0)
    # target_date_str = (target_utc - timedelta(hours=6)).strftime('%A, %b %d') # CST Date
    central = pytz.timezone('US/Central')
    now_utc = datetime.now(pytz.utc)
    now_cst = datetime.now(pytz.utc).astimezone(central)
    if current_run_time is None:
        run_time = (datetime.now(pytz.utc) - timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    else:
        run_time = current_run_time
    if run_time.tzinfo is None:
        run_time = run_time.replace(tzinfo=pytz.UTC)
    
    if now_cst.hour >= 3:
        target_date = now_cst.date() + timedelta(days=1)
    else:
        target_date = now_cst.date()
    
    target_utc = pytz.utc.localize(datetime.combine(target_date, time(10, 0)))
    target_local_time = target_utc.astimezone(central)
    display_date_str = (target_local_time - timedelta(days = 1)).strftime('%A, %b %d')
    
    fxx = int((target_utc - run_time).total_seconds() // 3600)
    if fxx > 18:
        # Find the most recent multiple of 6 UTC run
        run_time = run_time.replace(hour=(run_time.hour // 6) * 6)
        fxx = int((target_utc - run_time).total_seconds() // 3600)
        print(f"Long lead time ({fxx}h) detected. Snapping to extended run: {run_time}")
    try:
        H = Herbie(run_time.replace(tzinfo=None), model='hrrr', product='sfc', fxx=fxx)
        # Attempt to load the index to verify it's on the server
        _ = H.index_as_dataframe
    except Exception:
        # If the file is missing, go back 1 more hour
        # If it was an extended run (fxx > 18), go back 6 hours to the previous extended run
        rollback = 6 if fxx > 18 else 1
        run_time = run_time - timedelta(hours=rollback)
        fxx = int((target_utc - run_time).total_seconds() // 3600)
        H = Herbie(run_time.replace(tzinfo=None), model='hrrr', product='sfc', fxx=fxx)
    print (f"\nForecast time: {target_utc} UTC & {target_utc - timedelta(hours=5)} EST\nRun Time found: {run_time} UTC & {run_time - timedelta(hours=5)} EST\nApprox. lead time to forecast minimum temperature: {fxx} hours\n")
    try: #this try-except block is to ensure there is no corrupted file in the local system with index so when the system checks the index is there but the file is corrupted/missing. In that case, the except block redownloads the hrrr data
        # overwrite=True is used to fix the FileNotFoundError
        ds_list = H.xarray(r":TMP:surface|:TCDC:entire atmosphere|:LCDC:low", overwrite=True)
        # merge the different temp/cloud cover ds
        # compat="override" helps when coordinates are off by very small fraction
        ds = xr.merge(ds_list, join="override",compat="override", combine_attrs="drop_conflicts") #consider whether to add combine_attrs / join override
    except FileNotFoundError:
        # If the local file is corrupted/missing, clear the cache and try once more
        st.warning("Local weather file was corrupted. Re-downloading...")
        # Logic to re-run Herbie for new download
        ds_list = H.xarray(r":TMP:surface|:TCDC:entire atmosphere|:LCDC:low", 
        overwrite=True, 
        remove_grib=True)
        ds = xr.merge(ds_list, compat="override", join="override",combine_attrs="drop_conflicts")
    point_data = ds.herbie.pick_points(points_df, method="weighted")
    print(point_data.data_vars)
    safe_DOY = max([datetime.now().timetuple().tm_yday][0], 90)
    inputs = pd.DataFrame({'DOY' : [safe_DOY],
        'FTemp_4AM_Site_HRRR': [(float(point_data.t.values[0]) - 273.15)*(9/5)+32], 
        'ForeLCC_4AM_Site_HRRR': [float(point_data.lcc.values[0])],
        'ForeCC_4AM_Site_HRRR': [float(point_data.tcc.values[0])]
    })
#     inputs = pd.DataFrame({ 
#     'DOY' : [120], 
#     'FTemp_4AM_Site_HRRR': [19.0], 
#     'ForeLCC_4AM_Site_HRRR': [0.0], 
#     'ForeCC_4AM_Site_HRRR': [0.0]
# })
    print(inputs)
    
    inputs_scaled = scaler.transform(inputs)
    print(f"Scaled Inputs (First 3 values): {inputs_scaled}")
    prediction = model.predict(inputs_scaled)[0]
    print(f"Forecast for tonight: {prediction:.2f}°F. Result not logged.")
    return {
        "prediction": round(prediction, 1),
        "hrrr_temp": round(inputs['FTemp_4AM_Site_HRRR'].iloc[0], 1),
        "hrrr_lcc": round(inputs['ForeLCC_4AM_Site_HRRR'].iloc[0], 1),
        "hrrr_tcc": round(inputs['ForeCC_4AM_Site_HRRR'].iloc[0], 1),
        "run_time": run_time.strftime('%H:%M UTC'),
        "fxx": fxx,
        "target_date": display_date_str
        }

def get_hrrr_curve(BOG_LAT, BOG_LON, run_time):
    points_df = pd.DataFrame({"latitude": [BOG_LAT], "longitude": [BOG_LON]}) 
    central = pytz.timezone('US/Central')
    now_utc = datetime.now(pytz.utc)
    now_cst = now_utc.astimezone(central)
    
    if now_cst.hour >= 3:
        target_morning_date = now_cst.date() + timedelta(days=1)
    else:
        target_morning_date = now_cst.date()
        
        
    # 9 PM CST is 03:00 UTC (Next Day); 6 AM CST is 12:00 UTC (Next Day)
    # Logic to find the UTC timestamp for 9 PM Today
    target_start_utc = pytz.utc.localize(datetime.combine(target_morning_date, time(3, 0)))
    # Calculate starting fxx
    start_fxx = int((target_start_utc - run_time).total_seconds() // 3600)
    curve_data = []
    # Fetch 10 points: 9PM, 10PM... through 6AM
    for i in range(10):
        f = start_fxx + i
        # HRRR standard runs go max to 18 hours (longer lead times are extended runs)
        # If we need longer, we have the 00/06/12/18z runs.
        if f > 18 or f < 0:
            continue 
        try:
            point_utc = run_time + timedelta(hours=f)
            point_cst = point_utc.astimezone(central)
            if point_cst <= now_cst:
                continue
            H = Herbie(run_time.replace(tzinfo=None), model='hrrr', product='sfc', fxx=f)
            ds = H.xarray(r":TMP:surface")
            p = ds.herbie.pick_points(points_df, method="weighted")
            temp_f = (float(p.t.values[0]) - 273.15) * 1.8 + 32
            
            # Convert valid time to CST for display
            valid_time_cst = (run_time + timedelta(hours=f)) - timedelta(hours=6)
            
            curve_data.append({
                "Time": point_cst.strftime("%I %p"), 
                "Temp": round(temp_f, 1)
            })
        except:
            break # Stop if file isn't on server yet
            
    return pd.DataFrame(curve_data)

# if __name__ == "__main__":
#     adict = run_forecast(45.20765,-89.86566)

# %% streamlit APP CODE SECTION

st.set_page_config(page_title="Cranberry Frostcast", layout="wide")
# st.set_page_config(page_title="Bog Frost Forecast Tool", layout="wide")

# CACHING THE MODEL - We load the model once and keep it in memory
@st.cache_resource
def load_ml_model(path):
    package = joblib.load(path)
    return package['model'], package['scaler']

#  CACHING THE WEATHER FETCHED BY THE HOUR 
@st.cache_data
def get_prediction(lat, lon,current_run_time):
    # This calls the function defined before that has the try/except rollback logic
    # The function returns a dictionary with metadata
    return run_forecast(lat, lon,current_run_time)

@st.cache_data 
def get_cached_hrrr_curve(lat, lon, current_run_time):
    return get_hrrr_curve(lat, lon, current_run_time)

#reduce white spacing in the page
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 0rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Cranberry Marsh Frostcast ❄️")

# st.info("Current Time: " + datetime.utcnow().strftime('%H:%M') + " UTC (" + (datetime.utcnow()- timedelta(hours=5)).strftime('%H:%M') + " EST)")

col1, col2 = st.columns([1, 2])
with col1:
    # st.header("1. Marsh Location")
    st.header("1. Enter Farm Location")
    site_name = st.text_input("Marsh Name", "Copper River Marsh")
    # site_name = st.text_input("Bog Name", "Copper River Marsh")
    lat = st.number_input("Latitude", value=45.207653, format="%.6f")
    lon = st.number_input("Longitude", value=-89.86566, format="%.6f")
    tol = st.number_input("Tolerance", value=32.0, format="%.1f",step = 0.5)
    predict_btn = st.button("Generate Forecast", type="primary", width = 'stretch')

with col2:
    st.header("2. Location Preview")
    map_df = pd.DataFrame({'lat': [lat], 'lon': [lon]})
    st.map(map_df, zoom=12)
    
st.markdown("---") 
st.write("""
    **Disclaimer:** This forecast is for informational purposes only. 
    Frost protection decisions should be made based on multiple data sources, 
    including on-site thermometers.
""")   
now_utc = datetime.now(pytz.utc)
# We assume the HRRR run from 2 hours ago is the most recent stable one available
latest_run_time = (now_utc - timedelta(hours=2)).replace(minute=0, second=0, microsecond=0)

# .replace(tzinfo=None) with all herbie lines H = Herbie(latest_run_time.replace(tzinfo=None), model='hrrr', product='sfc', fxx=fxx)

if predict_btn:
    st.session_state.show_results = True

if st.session_state.get('show_results'):
    # creating data container to lock data and so it stays with any reruns
    result_container = st.container()
    
    with result_container:
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()
        
        # The map embedded in the window before forecast and hrrr curve in its own window
    with st.expander("Hourly Cloud Movement (Forecast by NOAA NDFD)", expanded=True):
        # creating a Time Slider for the next 12 hours using local time but UTC used for request data from NOAA
        central_tz = pytz.timezone('US/Central')
        central_now_CCmap = datetime.now(pytz.utc).astimezone(central_tz) #In central time for displaying only
        # The user can choose between 0 - 12 hours to look at cloud cover forecast
        hour_offset = st.slider("Forecast Hour Offset", 0, 12, 0, help="Slide to see how clouds move over the next 12 hours.")
         
        # Calculate target times
        target_local_CCmap = central_now_CCmap + timedelta(hours=hour_offset)
        target_utc_CCmap = datetime.utcnow() + timedelta(hours=hour_offset)
        
        # Format for the Map Request (UTC) and the Label (Local)
        vtit_time = target_utc_CCmap.strftime("%Y-%m-%dT%H:00")
        # display_label = target_local.strftime("%I:00 %p %b %d")
        display_label = (target_local_CCmap).strftime("%B %d, %I:00 %p") # This display is according to central or CST/CDT. So there will be one hour lag from NOAA interface if accessed in Mass.
        st.subheader(f"Cloud cover Forecast for {display_label}")
    
        # Setup Map
        m = folium.Map(location=[lat, lon], zoom_start=7, tiles="cartodbpositron")
    
        # Add the NOAA CC Layer synced to the Slider's Time
        folium.WmsTileLayer(
            url="https://digital.weather.gov/ndfd.conus/wms",
            layers="ndfd.conus.sky",
            name="Sky Cover %",
            fmt="image/png",
            transparent=True,
            opacity=0.6,
            version="1.3.0",
            styles="default",
            vtit=vtit_time ,attr="NOAA NDFD"
        ).add_to(m)
    
        # Marker for the Farm
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color="red",
            fill=True,
            popup=f"Marsh: {site_name}"
        ).add_to(m)
        
        # Legend
        # Create 4 equal columns
        leg_col1, leg_col2, leg_col3, leg_col4 = st.columns(4)
        
        with leg_col1:
            st.markdown("**Deep Blue:** \nClear Sky (0-25%)")
        
        with leg_col2:
            st.markdown("**White/Pale Blue:** \nLight (25-50%)")
        
        with leg_col3:
            st.markdown("**Light Gray:** \nModerate (50-75%)")
        
        with leg_col4:
            st.markdown("**Dark Gray:** \nOvercast (75-100%)")
        # Rendering the map
        st_folium(m, height=600, use_container_width=True, key=f"map_{hour_offset}")
        
    with st.spinner(f"Analyzing HRRR data for {site_name}..."):
        res = get_prediction(lat, lon, latest_run_time)
        is_frost = res['prediction'] <= tol
        if is_frost:
            diff = tol - res['prediction']
            aorb = 'below'
        else:
            diff = res['prediction'] - tol
            aorb = 'above'
        # Display Forecast Metrics
        st.subheader(f"Forecast for {site_name} for the night of {res['target_date']}")
        # Adding a caption so the grower knows time when data was refreshed
        st.caption(f"Valid for the night of {res['target_date']} | Based on HRRR {res['run_time']} run (Lead Time: F{res['fxx']})")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Minimum Marsh Temperature", f"{res['prediction']}°F")
        # st.caption(f"Based on Tolerance: {tol:.1f}°F")
        m2.metric("HRRR Surface Temp", f"{res['hrrr_temp']}°F")
        m3.metric("Total Cloud Cover", f"{res['hrrr_tcc']}%")
        if res['prediction'] <= tol - 5:
            st.error(f"⚠️ **HIGH FROST RISK**: Predicted temperature is {res['prediction']:.1f}°F, which is {diff:.1f}°F {aorb} Tolerance ({tol:.1f}°F).")
        elif tol - 5 < res['prediction'] <= tol:
            st.error(f"⚠️ **FROST RISK**: Predicted temperature is {res['prediction']:.1f}°F, which is {diff:.1f}°F {aorb} Tolerance ({tol:.1f}°F).")
        elif tol < res['prediction'] <= tol + 5:
            st.info(f"⚠️ **FROST POSSIBLE**: Predicted temperature is {res['prediction']:.1f}°F, which is {diff:.1f}°F {aorb} Tolerance ({tol:.1f}°F).")
        elif res['prediction'] > tol + 5:
            st.success(f"✅ **LOW RISK**: Conditions currently look safe. Predicted temperature is {res['prediction']:.1f}°F, which is {diff:.1f}°F {aorb} Tolerance ({tol:.1f}°F).")
        # Display the Hourly HRRR Curve
        st.markdown("### Overnight Temperature Trend using NOAA's HRRR regional model")
        df_curve = get_cached_hrrr_curve(lat, lon, latest_run_time)
        if not df_curve.empty:
            plot_col, _ = st.columns([0.8, 0.2])
            with plot_col:
                fig = px.line(df_curve, x="Time", y="Temp", 
                              labels={"Temp": "Temperature (°F)", "Time": "Time"},
                              markers=True)
                # Update the Curve and Markers
                fig.update_traces(line_color='#2E86C1', line_width=4, marker=dict(size=10))
                fig.add_hline(y=tol, line_dash="dash", line_color="#E74C3C", line_width=3,
                              annotation_text=f"TOLERANCE: {tol}°F", 
                              annotation_position="bottom right",
                              annotation_font_size=20,
                              annotation_font_color="#E74C3C")
                fig.update_layout(
                    template="simple_white", 
                    hovermode="x unified",
                    height=450,
                    margin=dict(l=50, r=50, t=20, b=50),
                    xaxis=dict(
                        title_font=dict(size=22),
                        tickfont=dict(size=18),
                        gridcolor='lightgrey'
                    ),
                    yaxis=dict(
                        title_font=dict(size=22),
                        tickfont=dict(size=18),
                        gridcolor='lightgrey',
                        range=[min(df_curve['Temp'].min(), tol) - 2, max(df_curve['Temp'].max(), tol) + 2]
                    ))
                st.plotly_chart(fig, width='stretch')
        else:
            st.warning("Hourly curve data not available for the current HRRR window.")
    
