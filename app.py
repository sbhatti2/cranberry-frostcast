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
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#%% Backend function 
# MODEL_PATH = 'frost_model_010626.joblib'

@st.cache_data(ttl=1800)
def find_5am_value(lat, lon):
    headers = {'User-Agent': '(myweatherapp.com, contact@email.com)'}
    
    # 1. Get Today's date dynamically
    now = datetime.now()
    # If it's already past 5 AM today, we want 5 AM tomorrow
    target_day = now.day if now.hour < 3 else (now + timedelta(days=1)).day
    
    print(f"Searching for 5 AM on Day: {target_day}") # Debug log

    try:
        point_url = f"https://api.weather.gov/points/{lat},{lon}"
        res = requests.get(point_url, headers=headers).json()
        forecast_url = res['properties']['forecastHourly']
        forecast_data = requests.get(forecast_url, headers=headers).json()
        periods = forecast_data['properties']['periods']

        for i, period in enumerate(periods):
            start_dt = datetime.fromisoformat(period['startTime'])
            
            # Check for hour 5 on our dynamic target day
            if start_dt.hour == 5 and start_dt.day == target_day:
                val = period['temperature']
                print(f"Found! Target Time: {period['startTime']}, Value: {val}")
                return val, i

        print("!!! Could not find a 5 AM period in the current forecast.")
        return None # Handled by your new safety checks in app.py

    except Exception as e:
        print(f"!!! API Error: {e}")
        return None
        


@st.cache_data(ttl=1800)
def get_forecast_value_synced(lat, lon, map_choice, hour_offset):
    try:
        # 1. Get the grid endpoint for your coordinates
        # NWS requires a User-Agent header
        headers = {'User-Agent': '(myweatherapp.com, contact@email.com)'}
        point_url = f"https://api.weather.gov/points/{lat},{lon}"
        res = requests.get(point_url, headers=headers).json()
        
        # 2. Get the hourly forecast
        forecast_url = res['properties']['forecastHourly']
        forecast_data = requests.get(forecast_url, headers=headers).json()
        
        # 3. Grab the first period (current/upcoming hour)
        selected_period = forecast_data['properties']['periods'][hour_offset]
        # 2. Validation: Print to console to ensure times match map VTIT
        target_time = selected_period['startTime']
        print(target_time)
        if "Temp" in map_choice:
            return f"{selected_period['temperature']}°{selected_period['temperatureUnit']}"
        elif "Dew" in map_choice:
            dp_c = selected_period.get('dewpoint', {}).get('value')
            if dp_c is not None:
                dp_f = (dp_c * 9/5) + 32
                return f"{int(dp_f)}°F"
        elif "Sky" in map_choice or "Cloud" in map_choice:
            # Hourly forecast doesn't always have a % for sky, 
            # so we use the short description (e.g., "Partly Cloudy")
            return selected_period.get('shortForecast', 'N/A')
            
        return f"{selected_period['temperature']}°F"
    except Exception as e:
        return "--"

def get_synoptic_value(obs_dict, search_term):
    """Returns (value, timestamp) for the first key matching search_term."""
    for key, data in obs_dict.items():
        if search_term in key and 'value' in data:
            val = data['value']
            # Synoptic uses 'date_time' for the observation timestamp
            ts = data.get('date_time', None) 
            return val if val is not None else 0.0, ts
    return 0.0, None

def get_multi_synoptic_observations(token, station_list=["KEWB", "KPYM"], target_time=None):
    # Pass stations as a comma-separated string: "KEWB,KPYM"
    stids = ",".join(station_list)
    # url = f"https://api.synopticdata.com/v2/stations/latest?stid={stids}&token={token}&units=english" # this url was good for latest time
    url = f"https://api.synopticdata.com/v2/stations/nearesttime?stid={stids}&token={token}&units=english" # this url is good to grab a certain time stamp like in our case will be the hrrr run time
    # 2. Add the timestamp if provided (Format: YYYYMMDDHHMM)
    if target_time:
        # Synoptic expects UTC time in this format for the 'attime' parameter
        ts_str = target_time.strftime("%Y%m%d%H%M")
        # 'within=60' looks for the closest observation within 60 minutes of the run_time
        url += f"&attime={ts_str}&within=60"
    results = {}
    try:
        response = requests.get(url).json()
        if response['SUMMARY']['RESPONSE_CODE'] == 1:
            for station in response['STATION']:
                stid = station['STID']
                obs = station['OBSERVATIONS']
                
                # Get values and timestamps
                temp_val, temp_ts = get_synoptic_value(obs, 'air_temp')
                
                results[stid] = {
                    'dp': get_synoptic_value(obs, 'dew_point_temperature')[0],
                    'wind_speed': get_synoptic_value(obs, 'wind_speed')[0],
                    'wind_dir': get_synoptic_value(obs, 'wind_direction')[0],
                    'air_temp': temp_val,
                    'obs_time': temp_ts
                }
            return results
    except Exception as e:
        print(f"Error fetching multi-station data: {e}")
    return None

def predict_quantile(model, X, q):
    """
    Get qth quantile prediction across all RF trees.
    q=0.15 means forecast the 15th percentile — errs cold, reduces misses.
    """
    tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
    # tree_preds shape: (n_estimators, n_samples)
    return np.percentile(tree_preds, q * 100, axis=0)

def run_forecast(BOG_LAT, BOG_LON,model, scaler, time_code,quantile=None, current_run_time = None):
    points_df = pd.DataFrame({
        "latitude": [BOG_LAT],
        "longitude": [BOG_LON]})

    # package = joblib.load(MODEL_PATH)
    # model = package['model']
    # scaler = package['scaler']
    
    ############# FETCH LIVE WEATHER DATA (HRRR Model)
    # now_utc = datetime.utcnow()
    # target_utc = now_utc.replace(hour=10, minute=0, second=0, microsecond=0)
    # target_date_str = (target_utc - timedelta(hours=6)).strftime('%A, %b %d') # CST Date

    local_tz = pytz.timezone(time_code)
    now_utc = datetime.now(pytz.utc)
    now_local = now_utc.astimezone(local_tz)
    tz_suffix = "EST" if "Eastern" in time_code else "CST"
    std_offset_hours = 5 if "Eastern" in time_code else 6

    if current_run_time is None:
        run_time = (now_utc - timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    else:
        run_time = current_run_time if current_run_time.tzinfo else pytz.utc.localize(current_run_time)
    # run_time_local = run_time.astimezone(local_tz)
    run_time_std_local = (run_time.astimezone(pytz.utc) - timedelta(hours=std_offset_hours)).replace(tzinfo=None)
    if now_local.hour >= 3:
        target_date = now_local.date() + timedelta(days=1)
    else:
        target_date = now_local.date()
    
    # Target is 4 AM Local -> Convert to UTC for Herbie
    # target_4am_local = local_tz.localize(datetime.combine(target_date, time(4, 0))) daylight time
    target_4am_std_local = datetime.combine(target_date, time(4, 0))
    target_4am_utc = pytz.utc.localize(target_4am_std_local + timedelta(hours=std_offset_hours))
    # target_4am_utc = target_4am_local.astimezone(pytz.utc)
    display_date_str = (target_4am_std_local - timedelta(days = 1)).strftime('%A, %b %d')
    
    fxx = int((target_4am_utc - run_time).total_seconds() // 3600)
    if fxx > 18:
        # Find the most recent multiple of 6 UTC run
        run_time = run_time.replace(hour=(run_time.hour // 6) * 6)
        fxx = int((target_4am_utc - run_time).total_seconds() // 3600)
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
        fxx = int((target_4am_utc - run_time).total_seconds() // 3600)
        H = Herbie(run_time.replace(tzinfo=None), model='hrrr', product='sfc', fxx=fxx)
    print (f"\nForecast time: {target_4am_utc} UTC & {target_4am_std_local} {tz_suffix}\nRun Time found: {run_time} UTC & {run_time_std_local} {tz_suffix}\nApprox. lead time to forecast minimum temperature: {fxx} hours\n")
    # Universally define the sub-hour Herbie objects
    H_12am = Herbie(run_time.replace(tzinfo=None), model='hrrr', product='sfc', fxx=fxx - 4)
    H_2am = Herbie(run_time.replace(tzinfo=None), model='hrrr', product='sfc', fxx=fxx - 2)
    try: #this try-except block is to ensure there is no corrupted file in the local system with index so when the system checks the index is there but the file is corrupted/missing. In that case, the except block redownloads the hrrr data
        # overwrite=True is used to fix the FileNotFoundError
        ds_list = H.xarray(r":TMP:surface|:TCDC:entire atmosphere|:LCDC:low", overwrite=True)
        # merge the different temp/cloud cover ds
        # compat="override" helps when coordinates are off by very small fraction
        ds = xr.merge(ds_list, join="override",compat="override", combine_attrs="drop_conflicts") #consider whether to add combine_attrs / join override
        # Sub-hour Cloud Cover Data
        ds_12am = H_12am.xarray(r":TCDC:entire atmosphere", overwrite=True)
        ds_2am = H_2am.xarray(r":TCDC:entire atmosphere", overwrite=True)
    except FileNotFoundError:
        # If the local file is corrupted/missing, clear the cache and try once more
        st.warning("Local weather file was corrupted. Re-downloading...")
        # Logic to re-run Herbie for new download
        ds_list = H.xarray(r":TMP:surface|:TCDC:entire atmosphere|:LCDC:low", 
        overwrite=True, 
        remove_grib=True)
        ds = xr.merge(ds_list, compat="override", join="override",combine_attrs="drop_conflicts")
        ds_12am = H_12am.xarray(r":TCDC:entire atmosphere", overwrite=True, remove_grib=True)
        ds_2am = H_2am.xarray(r":TCDC:entire atmosphere", overwrite=True, remove_grib=True)
    point_data = ds.herbie.pick_points(points_df, method="weighted")
    pt_12am = ds_12am.herbie.pick_points(points_df, method="weighted")
    pt_2am = ds_2am.herbie.pick_points(points_df, method="weighted")
    print(point_data.data_vars)
    safe_DOY = max([datetime.now().timetuple().tm_yday][0], 90)
    # Universal Variables
    temp_f = (float(point_data.t.values[0]) - 273.15) * 1.8 + 32
    fcc_4am = float(point_data.tcc.values[0])
    fcc_12am = float(pt_12am.tcc.values[0])
    fcc_2am = float(pt_2am.tcc.values[0])
    flcc_4am = float(point_data.lcc.values[0])
    
    # --- 2. REGION-SPECIFIC DATA FETCHING ---
    if "Eastern" in time_code:  
    ############ Fetch NOAA LCD NB and Ply weather station real time data
        station_data = get_multi_synoptic_observations('d6dc992ce66644d68aa646f7f1e344a6', ["KEWB", "KPYM"], target_time=run_time)
        
        # Extract with safety checks (defaulting to neutral values if a station is down)
        nb = station_data.get('KEWB', {})
        ply = station_data.get('KPYM', {})
        
        # NEW BEDFORD FEATURES (Primary)
        current_dp_nb = nb.get('dp', 0.0)
        current_wind_speed_nb = min(17,nb.get('wind_speed', 0.0)) #capping ws from nb station to 17 mph cause anything above 18mph causes prediction to be 7-8F warmer
        current_wind_dir_nb = nb.get('wind_dir', 0.0)
        current_air_temp_nb = nb.get('air_temp', 32.0)
        
        # PLYMOUTH FEATURES (Secondary for Delta)
        current_air_temp_ply = ply.get('air_temp', 32.0)
        
        # 3. CALCULATE DERIVED FEATURES FOR THE MODEL
        # Invrsn_pot_NB (Temp - Dewpoint)
        inversion_pot = current_air_temp_nb - current_dp_nb
        
        # Delta_AirTemp_NB_Ply (Difference between the two sites)
        delta_temp = current_air_temp_nb - current_air_temp_ply
        
        # Timestamp for the UI
        obs_time_str = nb.get('obs_time', "Unknown")
        dt_obj = datetime.strptime(nb['obs_time'], "%Y-%m-%dT%H:%M:%SZ")
        # Localize to Eastern Time for the user
        local_obs_time = dt_obj.replace(tzinfo=pytz.utc).astimezone(pytz.timezone(time_code))
        readable_time = local_obs_time.strftime("%I:%M %p")
    elif "Central" in time_code:
        readable_time = "N/A"
    
    if "Eastern" in time_code:   
        # MA Feature Set (12 Columns)
        inputs = pd.DataFrame([{
            'FTCC_4AM_HRRR': fcc_4am,
            'FTemp_4AM_HRRR': temp_f,
            'hours_to_target': fxx,
            'DOY': safe_DOY,
            'DewPoint_leadtime_NB': current_dp_nb,
            'WindSpeed_leadtime_NB': current_wind_speed_nb,
            'WindDirec_leadtime_NB': current_wind_dir_nb,
            'Invrsn_pot_NB': inversion_pot,
            'Delta_AirTemp_NB_Ply': delta_temp,
            'ForeCC_12AM_Site_HRRR': fcc_12am,
            'ForeCC_2AM_Site_HRRR': fcc_2am,
            'Lat': BOG_LAT
        }])


    elif "Central" in time_code: 
        inputs = pd.DataFrame({'DOY' : [safe_DOY],
            'FTemp_4AM_Site_HRRR': temp_f, 
            'ForeLCC_4AM_Site_HRRR': flcc_4am,
            'ForeCC_4AM_Site_HRRR': fcc_4am
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
    
    if quantile is not None and hasattr(model, 'estimators_'):
        q_pred = predict_quantile(model, inputs_scaled, quantile)[0]
        mean_pred = model.predict(inputs_scaled)[0]
        # Take the most conservative (lowest) value
        prediction = min(q_pred, mean_pred)
        # log it for debugging
        print(f"Quantile ({quantile}): {q_pred:.2f} | Mean: {mean_pred:.2f} | Final: {prediction:.2f}")
    else:
        prediction = model.predict(inputs_scaled)[0]
    print(f"Forecast for tonight: {prediction:.2f}°F. Result not logged.")
    return {
        "prediction": round(prediction, 1),
        "hrrr_temp": round(temp_f, 1),
        "hrrr_tcc": round(fcc_4am, 1),
        "run_time": run_time.strftime('%H:%M UTC'),
        "fxx": fxx,
        "target_date": display_date_str
        }
    

def get_hrrr_curve(BOG_LAT, BOG_LON, run_time, Time_Code):
    points_df = pd.DataFrame({"latitude": [BOG_LAT], "longitude": [BOG_LON]}) 
    LOCAL_TZ = pytz.timezone(Time_Code)
    now_local = datetime.now(LOCAL_TZ)

    # 1. Determine the target night
    # If it's before 3 AM, we are likely looking at "tonight's" ongoing run.
    # If it's 9 AM, we are looking forward to the upcoming evening.
    if now_local.hour >= 3:
        target_morning_date = now_local.date() + timedelta(days=1)
    else:
        target_morning_date = now_local.date()
    
    # 2. Define the fixed 8 PM (Sunset) to 6 AM (Sunrise) Window
    # Sunset is 8 PM the night before the target morning
    target_start_local = LOCAL_TZ.localize(datetime.combine(target_morning_date - timedelta(days=1), time(20, 0)))
    target_start_utc = target_start_local.astimezone(pytz.utc)

    # 3. Handle Extended Run Snap
    # We check the lead time for the end of the curve (6 AM)
    target_end_utc = target_start_utc + timedelta(hours=10)
    max_fxx = int((target_end_utc - run_time).total_seconds() // 3600)
    
    if max_fxx > 18:
        # Snap to 00, 06, 12, or 18 UTC for long-range data
        run_time = run_time.replace(hour=(run_time.hour // 6) * 6, minute=0, second=0, microsecond=0)

    curve_data = []
    
    # 4. Loop for 11 points (8 PM to 6 AM inclusive)
    for i in range(11):
        point_utc = target_start_utc + timedelta(hours=i)
        f = int((point_utc - run_time).total_seconds() // 3600)
        
        if f < 0: continue 

        try:
            point_time = point_utc.astimezone(LOCAL_TZ)
            H = Herbie(run_time.replace(tzinfo=None), model='hrrr', product='sfc', fxx=f)
            
            # Verify file availability/Rollback
            try:
                _ = H.index_as_dataframe
            except:
                rollback = 6 if max_fxx > 18 else 1
                temp_run = run_time - timedelta(hours=rollback)
                f_new = int((point_utc - temp_run).total_seconds() // 3600)
                H = Herbie(temp_run.replace(tzinfo=None), model='hrrr', product='sfc', fxx=f_new)

            ds_list = H.xarray(r":TMP:surface|:TCDC:entire atmosphere", overwrite=True)
            ds = xr.merge(ds_list, join="override",compat="override", combine_attrs="drop_conflicts") #consider whether to add combine_attrs / join override
            
            
            
            
            p = ds.herbie.pick_points(points_df, method="weighted")
            
            temp_f = (float(p.t.values[0]) - 273.15) * 1.8 + 32
            cloud_pct = float(p.tcc.values[0])
            curve_data.append({
                "Time": point_time.strftime("%I %p"), 
                "Temp": round(temp_f, 1),
                "Cloud": round(cloud_pct, 1),
                "timestamp": point_time # Keep for chronological sorting
            })
        except:
            continue
            
    df = pd.DataFrame(curve_data)
    if not df.empty:
        df = df.sort_values("timestamp") # Ensures 01 AM comes after 11 PM
        
    return df

# if __name__ == "__main__":
#     adict = run_forecast(45.20765,-89.86566)
# --- HELPER FOR CUSTOM LEGENDS ---
def display_custom_legend(layer_id):
    if layer_id == "ndfd.conus.sky":
        # Cloud Cover: Vivid Sky Blue -> Lavender (at 57%) -> Darker Gray
        st.write(
            f'<div style="margin-bottom: 10px; width: 94%;">'
            f'<div style="background: linear-gradient(to right, '
            f'#7BC8F6 0%,    /* 0%: Vivid Light Sky Blue Base */'
            f'#a6dbfb 20%,   /* 20% Blend */'
            f'#cadbfd 40%,   /* 40% Blend */'
            f'#e6e6fa 55%,   /* 57%: Lavender Anchor */'
            f'#c7c7d7 64%,   /* 72% Transition toward gray */'
            f'#b0b0b0 88%,   /* 88% Transition */'
            f'#999999 100%   /* 100%: Darker Gray */'
            f'); height: 25px; width: 100%; margin-left: 0.5%; border-radius: 3px; border: 1px solid #444;"></div>'
            f'<div style="display: flex; justify-content: space-between; width: 100%; padding-top: 8px;">'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">0%</div>'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">20%</div>'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">40%</div>'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">60%</div>'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">80%</div>'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">100%</div>'
            f'</div></div>',
            unsafe_allow_html=True
        )
    elif layer_id == "ndfd.conus.td":
        st.write(
            f'<div style="margin-bottom: 30px; width: 94%;">'
            f'<div style="background: linear-gradient(to right, '
            f'#FFD1DC 0%,    /* -20: Baby Pink */'
            f'#e78ac3 10%,   /* -10: Set2 Pink (Muted/Updated) */'
            f'#9B72AA 20%,   /* 0: Muted Purple */'
            f'#4B0082 30%,   /* 10: Deep Violet */'
            f'#00E5FF 40%,   /* 20: Bright Sky Blue */'
            f'#00FF88 49%,   /* 30: Teal-Green */'
            f'#00DD55 55%,   /* 35: Matplotlib Green Peak */'
            f'#ADFF2F 60%,   /* 40: Green-Yellow */'
            f'#FFFF00 67%,   /* 45: Yellow Starts early */'
            f'#FFFF00 70%,   /* 50: Yellow Peak */'
            f'#FFFF00 72%,   /* 54: Yellow Lingers */'
            f'#FFA500 80%,   /* 60: Orange */'
            f'#FF6347 90%,   /* 70: Muted Red */'
            f'#9E1B32 100%   /* 80: Cranberry Red */'
            f'); height: 25px; width: 100%; margin-left: 0.5%; border-radius: 3px; border: 1px solid #444;"></div>'
            f'<div style="display: flex; justify-content: space-between; width: 100%; padding-top: 8px;">'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">-20°</div>'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">-10°</div>'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">0°</div>'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">10°</div>'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">20°</div>'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">30°</div>'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">40°</div>'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">50°</div>'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">60°</div>'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">70°</div>'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">80°</div>'
            f'</div></div>',
            unsafe_allow_html=True
        )
    elif layer_id == "ndfd.conus.t":
        st.write(
            f'<div style="margin-bottom: 30px; width: 94%;">'
            f'<div style="background: linear-gradient(to right, '
            f'#D3D3D3 0%,    /* -10: Light Gray */'
            f'#FFD1DC 8.3%,  /* 0: Baby Pink */'
            f'#e78ac3 16.6%, /* 10: Muted Pastel Pink */'
            f'#9B72AA 25%,   /* 20: Muted Purple */'
            f'#4B0082 33.3%, /* 30: Deep Violet */'
            f'#00E5FF 41.6%, /* 40: Bright Sky Blue */'
            f'#00FF88 49%,   /* 50: Teal-Green (Added more green) */'
            f'#00DD55 54.5%, /* 55-56: Parrot Green Peak (Shifted left from 60) */'
            f'#ADFF2F 59.9%, /* 60: Muted Green-Yellow transition */'
            f'#FFFF00 66.6%, /* 70: Bright Yellow Peak */'
            f'#FFA500 75%,   /* 80: Orange */'
            f'#FF6347 83.3%, /* 90: Muted Orangish-Red */'
            f'#9E1B32 91.6%, /* 100: Cranberry Red */'
            f'#FFC0CB 100%   /* 110: Pastel Red */'
            f'); height: 25px; width: 100%; margin-left: 0.5%; border-radius: 3px; border: 1px solid #444;"></div>'
            f'<div style="display: flex; justify-content: space-between; width: 100%; padding-top: 8px;">'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">-10°</div>'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">0°</div>'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">10°</div>'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">20°</div>'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">30°</div>'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">40°</div>'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">50°</div>'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">60°</div>'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">70°</div>'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">80°</div>'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">90°</div>'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">100°</div>'
            f'<div style="text-align: center; width: 0; white-space: nowrap; font-size: 16px; color: #444;">110°</div>'
            f'</div></div>',
            unsafe_allow_html=True
            )

# %% streamlit APP CODE SECTION
# 1. SET PAGE CONFIG FIRST (This must be the very first Streamlit command)
st.set_page_config(page_title="Cranberry Frostcast", layout="wide")

# 2. INITIALIZE SESSION STATE
if 'region_selected' not in st.session_state:
    st.session_state.region_selected = None

# --- 3. THE "CHANGE REGION" SIDEBAR BUTTON ---
if st.session_state.region_selected is not None:
    with st.sidebar:
        st.write(f"**Current Region:** {st.session_state.region_selected}")
        if st.button("🔄 Change Region / Reset"):
            # Clear caches and wipe session state to force a clean restart
            st.cache_data.clear()
            st.cache_resource.clear()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
# --- THE SPLASH SCREEN / ENTRY GATE ---
if st.session_state.region_selected is None:
    st.title("❄️ Cranberry Frostcast")
    st.subheader("Safety Disclaimer")
    # st.set_page_config(page_title="Cranberry Frostcast", layout="wide")
    st.warning("""
    **DISCLAIMER:** This tool is intented for informational purposes only and must not be used as the sole basis for spring frost management.
    Frost protection decisions must be made using multiple data sources,
    including on-farm thermometers and local weather observations. The researchers, UMass, and USDA-ARS
    assume no liability for crop loss or damages resulting from the use of this forecast.
    """)
    st.subheader("Select your region")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("**Wisconsin** (Marsh Tool)", width = 'stretch'):
            st.session_state.region_selected = "WI"
            # st.session_state.lat = 45.207653
            # st.session_state.lon = -89.865660
            # st.session_state.site_name = "Copper River Marsh"
            st.rerun()
            
    with col2:
        if st.button("**Massachusetts** (Bog Tool)", width = 'stretch'):
            st.session_state.region_selected = "MA"
            # st.session_state.lat = 41.800299
            # st.session_state.lon = -70.736287
            # st.session_state.site_name = "Rosebrook"
            st.rerun()
            
    st.stop()

if st.session_state.region_selected == "WI":
    MODEL_PATH = 'frost_model_010626.joblib'
    SCALER_PATH = 'frost_model_010626.joblib'
    APP_TITLE = "Cranberry Marsh Frostcast ❄️"
    BOG_TYPE = "Marsh"
    DEFAULT_SITE = "Copper River Marsh"
    DEFAULT_LAT = 45.207653
    DEFAULT_LON = -89.865660
    TOL = 32.0
    LOCAL_TZ = pytz.timezone('US/Central')
    Time_Code = 'US/Central'
    TZ_LABEL = "CDT" if datetime.now(pytz.timezone('US/Central')).dst() else "CST" 
    quantile = 0.5
    
elif st.session_state.region_selected == "MA":
    # Update these to your new filenames
    MODEL_PATH = 'rf_hybrid_under_042726.pkl'
    SCALER_PATH = 'scaler_hybrid_under_042726.pkl' 
    BOG_TYPE = "Bog"
    APP_TITLE = "Cranberry Bog Frostcast ❄️"
    DEFAULT_SITE = "Rosebrook"
    DEFAULT_LAT = 41.800299
    DEFAULT_LON = -70.736287
    TOL = 25
    LOCAL_TZ = pytz.timezone('US/Eastern')
    Time_Code = 'US/Eastern'
    TZ_LABEL = "EDT" if datetime.now(pytz.timezone('US/Eastern')).dst() else "EST"
    quantile = 0.5

# st.set_page_config(page_title = 'Cranberry Frostcast', layout="wide")
st.title(APP_TITLE)

# CACHING THE MODEL - We load the model once and keep it in memory
@st.cache_resource
def load_ml_model(model_path, scaler_path):
    m_obj = joblib.load(model_path)
    # If it's the WI dictionary format
    if isinstance(m_obj, dict) and 'model' in m_obj:
        model = m_obj['model']
    else:
        model = m_obj

    # Load scaler
    s_obj = joblib.load(scaler_path)
    # If it's the WI dictionary format
    if isinstance(s_obj, dict) and 'scaler' in s_obj:
        scaler = s_obj['scaler']
    else:
        scaler = s_obj
    return model, scaler

loaded_model, loaded_scaler = load_ml_model(MODEL_PATH, SCALER_PATH)
#  CACHING THE WEATHER FETCHED BY THE HOUR 
@st.cache_data(ttl=1800)
def get_prediction(lat, lon,current_run_time, _model, _scaler,Time_Code,quantile=None):
    # This calls the function defined before that has the try/except rollback logic
    # The function returns a dictionary with metadata
    return run_forecast(lat, lon,_model, _scaler,Time_Code,quantile, current_run_time)



@st.cache_data(ttl=1800)
def get_cached_hrrr_curve(lat, lon, current_run_time,Time_Code):
    return get_hrrr_curve(lat, lon, current_run_time,Time_Code)

#reduce white spacing in the page
st.markdown("""
    <style>
        .block-container {
            padding-top: 3rem;
            padding-bottom: 0rem;
        }
    </style>
""", unsafe_allow_html=True)

if 'lat' not in st.session_state:
    st.session_state.lat = DEFAULT_LAT
if 'lon' not in st.session_state:
    st.session_state.lon = DEFAULT_LON
if 'site_name' not in st.session_state:
    st.session_state.site_name = DEFAULT_SITE
    
col1, col2 = st.columns([1, 2])

with col1:
    st.header(f"1. Enter {BOG_TYPE} Details")
    st.warning(f"**To generate the forecast,** select location from map below, enter your {BOG_TYPE.lower()}'s tolerance and click on Generate Forecast button")
    # 1. Pull current values from state
    current_lat = st.session_state.get('lat')
    current_lon = st.session_state.get('lon')
    current_site = st.session_state.get('site_name')

    # 2. Create inputs using those values
    site_name = st.text_input(f"{BOG_TYPE} Name", value=current_site)
    lat = st.number_input("Latitude", value=current_lat, format="%.6f")
    lon = st.number_input("Longitude", value=current_lon, format="%.6f")
    
    # 3. Update state immediately if user types something new
    st.session_state.lat = lat
    st.session_state.lon = lon
    st.session_state.site_name = site_name
    
    tol = st.number_input("Tolerance", value=float(TOL), format="%.1f", step=0.5)
    predict_btn = st.button("Generate Forecast", type="primary", use_container_width=True)

with col2:
    st.header("2. Click Map to Select Location")
    st.warning("**To select your farm location,** you can drag the map and click exactly where you want the prediction for. Please confirm your selection below the map.")
    # Create the Folium Map
    m = folium.Map(
        location=[st.session_state.lat, st.session_state.lon], 
        zoom_start=14,
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Google'
    )
    
    # Add a marker for the CURRENTLY SAVED location
    folium.Marker(
        [st.session_state.lat, st.session_state.lon],
        popup=f"Selected Location",
        icon=folium.Icon(color='red')
    ).add_to(m)

    # st_folium will capture the click
    map_output = st_folium(m, width="100%", height=500, key="farm_map")

    # Logic for the "Confirm" step
    if map_output and map_output.get("last_clicked"):
        click_lat = map_output["last_clicked"]["lat"]
        click_lon = map_output["last_clicked"]["lng"]
        
        # Display a bold confirmation button right below the map if they click
        
        # Using a large, bold button that is hard to miss
        if st.button("**Confirm your selected location** ✅ ", type="primary"):
            st.session_state.lat = click_lat
            st.session_state.lon = click_lon
            st.session_state.site_name = f"Selected {BOG_TYPE}"
            st.rerun()
           
        st.markdown(f"**Selected Point: {click_lat:.5f}, {click_lon:.5f}**")
            
st.markdown("---") 
st.write("""
    **Disclaimer:** This forecast is for informational purposes only. 
    Frost protection decisions should be made based on multiple data sources, 
    including on-site temperature sensors. Cranberry buds should be observed in your respective farms to check tolerance.
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
    with st.expander("Weather Forecast Maps (NOAA NDFD)", expanded=True):
        layer_config = {
        "Sky Cover (%)": {"id": "ndfd.conus.sky", "label": "Sky Cover"},
        "Air Temperature (°F)": {"id": "ndfd.conus.t", "label": "Air Temp"}, # Changed to 't'
        "Dew Point (°F)": {"id": "ndfd.conus.td", "label": "Dew Point"}
    }
    
        map_choice = st.selectbox("Select Map Layer", list(layer_config.keys()))
        selected_id = layer_config[map_choice]["id"]
        # creating a Time Slider for the next 12 hours using local time but UTC used for request data from NOAA
        
        
        # The user can choose between 0 - 24 hours to look at cloud cover forecast
        hour_offset = st.slider(
            "Forecast Hour Offset", 
            min_value=0, 
            max_value=24, 
            value=0, 
            step=1, 
            help="Slide to see how conditions move over the next 24 hours (1-hour increments)."
        )
         
        # Calculate target times
        local_now_CCmap = datetime.now(pytz.utc).astimezone(LOCAL_TZ) 
        target_local_CCmap = local_now_CCmap + timedelta(hours=hour_offset)
        target_utc_CCmap = datetime.utcnow() + timedelta(hours=hour_offset)
        
        # Format for the Map Request (UTC) and the Label (Local)
        vtit_time = target_utc_CCmap.strftime("%Y-%m-%dT%H:00")
        # display_label = target_local.strftime("%I:00 %p %b %d")
        display_label = (target_local_CCmap).strftime("%B %d, %I:00 %p") # This display is according to central or CST/CDT. So there will be one hour lag from NOAA interface if accessed in Mass.
        # st.subheader(f"Cloud cover Forecast for {display_label}")

        # Setup Map
        m = folium.Map(location=[lat, lon], zoom_start=12, tiles='cartodbpositron')
        # Add the NOAA CC Layer synced to the Slider's Time
        folium.WmsTileLayer(
            url="https://digital.weather.gov/ndfd.conus/wms",
            layers=selected_id,
            name=map_choice,
            fmt="image/png",
            transparent=True,
            opacity=0.7,
            version="1.3.0",
            vtit=vtit_time ,attr="NOAA NDFD"
        ).add_to(m)
        # 1. Fetch the specific value from NOAA for your Lat/Lon

        # Get the actual number
        display_val = get_forecast_value_synced(lat, lon, map_choice,hour_offset)
        
        # Add the marker with a "Halo" Label
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=f"""
                    <div style="
                        position: relative;
                        left: 20px; 
                        top: -10px;
                        white-space: nowrap;
                        font-weight: 500;
                        font-family: 'Arial Black', Gadget, sans-serif;
                        color: black;
                        font-size: 16px;
                        /* This creates the white border around the black text */
                        text-shadow: 
                            -2px -2px 0 #fff,  
                             2px -2px 0 #fff,
                            -2px  2px 0 #fff,
                             2px  2px 0 #fff,
                             0px  3px 5px rgba(0,0,0,0.5); /* Subtle drop shadow for depth */
                    ">
                        {display_val}
                    </div>
                """
            )
        ).add_to(m)
        
        # #Boundaries
        # folium.TileLayer(
        #     tiles="https://tiles.stadiamaps.com/tiles/stamen_toner_lines/{z}/{x}/{y}{r}.png",
        #     attr="© Stadia Maps © Stamen Design © OpenStreetMap contributors",
        #     name="Boundaries",
        #     opacity=0.9,
        #     overlay=True,
        #     control=False
        # ).add_to(m)
        
        folium.TileLayer(
                    tiles="https://{s}.basemaps.cartocdn.com/light_only_lines/{z}/{x}/{y}{r}.png",
                    attr="© OpenStreetMap contributors © CARTO",
                    name="State & Road Boundaries",
                    overlay=True,
                    control=False,
                    opacity=1.0 # Full opacity so they don't vanish under clouds
                ).add_to(m)
        #Town/State Labels
        folium.TileLayer(
            tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}{r}.png",
            attr="© OpenStreetMap contributors © CARTO",
            name="Towns & Boundaries",
            overlay=True,
            control=False
        ).add_to(m)
        
        # Marker for the Farm
        folium.CircleMarker(
            location=[lat, lon],
            radius=9,
            color="red",
            fill=False,
           popup=f"{BOG_TYPE}: {site_name}"
        ).add_to(m)

        st.subheader(f"{map_choice} Forecast for {display_label}")
        display_custom_legend(selected_id)
        folium.LayerControl().add_to(m)
        # Rendering the map
        st_folium(m, height=600, width = 'stretch', key=f"map_{selected_id}_{hour_offset}")
    
    with st.spinner(f"Analyzing HRRR data for {site_name.lower()}..."):
        res = get_prediction(lat, lon, latest_run_time, loaded_model, loaded_scaler,Time_Code,quantile)
        
        #for this mini block we are dealing with daylight times unless above
        # val, offset = find_5am_value(lat, lon)
        prediction_data = find_5am_value(lat, lon)
        if prediction_data:
            val, offset = prediction_data
        else:
            # Fallback values so the dashboard still renders
            val, offset = 0.0, 0.0
            st.sidebar.warning("Note: NWS comparison data is currently updating. Minimum bog temp prediction is still active.")
        ndfd_5amDL_temp = get_forecast_value_synced(lat, lon, "Temp", offset)
        try:
            ndfd_float = float(ndfd_5amDL_temp.replace("°F", ""))
        except ValueError:
            ndfd_float = None
        hrrr_5amDL_float = float(res['hrrr_temp'])
        
        # Check if we actually have an NDFD number before doing math
        if ndfd_float is not None:
            raw_diff = ndfd_float - hrrr_5amDL_float
            if raw_diff < 0:
                prediction_offset = round(raw_diff, 2)
            else:
                prediction_offset = 0.0
        else:
            # Fallback if NDFD is missing so the app doesn't crash
            prediction_offset = 0.0
            raw_diff = 0.0
    
        print('Comparing ndfd and hrrr for 5 AM DST, lowering the prediction by: ',prediction_offset)
        res['prediction'] = round(res['prediction'] + prediction_offset,1)
        
        
        is_frost = res['prediction'] <= tol
        if is_frost:
            diff = tol - res['prediction']
            aorb = 'below'
        else:
            diff = res['prediction'] - tol
            aorb = 'above'
        # Display Forecast Metrics
        st.subheader(f"Forecast for {site_name.lower()} for the night of {res['target_date']}")
        # Adding a caption so the grower knows time when data was refreshed
        # st.caption(f"Valid for the night of {res['target_date']} | Based on HRRR {res['run_time']} run (Lead Time: F{res['fxx']})")
        
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Predicted Minimum {BOG_TYPE} Temperature", f"{res['prediction']}°F")
        # st.caption(f"Based on Tolerance: {tol:.1f}°F")
        m2.metric(f"HRRR Surface Temp (Lead Time {res['fxx']} hours)", f"{res['hrrr_temp']}°F")
        weather_warning = ""
        if res['hrrr_tcc'] > 25:
            weather_warning = (f"\n\n**Note:** This forecast assumes **{int(res['hrrr_tcc'])}% cloud cover**. If the sky clears more than expected, temperatures will likely drop **below** these predicted values.")
        else:
            weather_warning = ( f"\n\n**Note:** Clear skies (**{int(res['hrrr_tcc'])}% cloud cover**) will lead to rapid radiational cooling. Temperatures may drop quickly if winds also become calm."
    )
        if res['prediction'] <= tol - 5:
            st.error(f"⚠️ **HIGH FROST RISK**: Predicted temperature is {res['prediction']:.1f}°F, which is {diff:.1f}°F {aorb} Tolerance ({tol:.1f}°F).{weather_warning}")
        elif tol - 5 < res['prediction'] <= tol:
            st.error(f"⚠️ **FROST RISK**: Predicted temperature is {res['prediction']:.1f}°F, which is {diff:.1f}°F {aorb} Tolerance ({tol:.1f}°F).{weather_warning}")
        elif tol < res['prediction'] <= tol + 7:
            st.info(f"⚠️ **FROST POSSIBLE**: Predicted temperature is {res['prediction']:.1f}°F, which is {diff:.1f}°F {aorb} Tolerance ({tol:.1f}°F).{weather_warning}")
        elif res['prediction'] > tol + 7:
            st.success(f"✅ **LOW RISK**: Conditions currently look safe. Predicted temperature is {res['prediction']:.1f}°F, which is {diff:.1f}°F {aorb} Tolerance ({tol:.1f}°F).{weather_warning}")
        # Display the Hourly HRRR Curve
        st.markdown("### Overnight Temperature Trend using NOAA's HRRR regional model")
        
        
        
        df_curve = get_cached_hrrr_curve(lat, lon, latest_run_time, Time_Code)
        # 1. Identify the minimum value in the raw HRRR curve
        hrrr_min_val = df_curve['Temp'].min()
        
        # 2. Calculate the offset between the HRRR floor and ML Prediction
        bog_offset = res['prediction'] - hrrr_min_val
        
        # 3. Create the new time series by applying this offset to the HRRR curve
        # We round to 1 decimal place to avoid the floating point tail issues
        df_curve['Bog_Temp_Predicted'] = round(df_curve['Temp'] + bog_offset, 1)
        
        # print(f"HRRR Curve Min: {hrrr_min_val}°F")
        # print(f"ML Predicted Min: {res['prediction']}°F")
        # print(f"Calculated Bog Offset: {bog_offset:+.1f}°F")
        #Curve code from here
        if not df_curve.empty:
            plot_col, _ = st.columns([0.9, 0.1])
            with plot_col:
                # Create figure with secondary y-axis
                fig = make_subplots(specs=[[{"secondary_y": True}]])
        
                # 1. Cloud Cover Bars (Secondary Y)
                fig.add_trace(
                    go.Bar(
                        x=df_curve['Time'], 
                        y=df_curve['Cloud'],
                        name="Cloud Cover %",
                        marker_color='rgba(169, 169, 169, 0.3)', # Transparent gray
                        hoverinfo="x+y" 
                    ),
                    secondary_y=True
                )
                # 2. Temperature Line (Primary Y)
                fig.add_trace(
                    go.Scatter(
                        x=df_curve['Time'], 
                        y=df_curve['Temp'],
                        name="HRRR Regional Temp Forecast",
                        line=dict(color='#2E86C1', width=3, dash='dash'),
                        mode='lines+markers',
        marker=dict(size=8, symbol='diamond'),
        hoverinfo="x+y"
                    ),
                    secondary_y=False
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_curve['Time'], 
                        y=df_curve['Bog_Temp_Predicted'],
                        name=f"Predicted {BOG_TYPE} Temperature",
                        # Thick solid wine line for the primary prediction
                        line=dict(color='#A60000', width=5), 
                        mode='lines+markers',
                        marker=dict(
                            size=10, 
                            color='#A60000',
                            line=dict(color='white', width=1)
                        ),
                        hoverinfo="x+y"
                    ),
                    secondary_y=False
                )
                # 3. Tolerance Line
                fig.add_hline(y=tol, line_dash="dash", line_color="#E74C3C", line_width=3,
                              annotation_text=f"TOLERANCE: {tol}°F", 
                              annotation_position="bottom left",
                              annotation_font_size=20,
                              annotation_font_color="#E74C3C")
        
                # 4. Final Layout
                fig.update_layout(
                    template="simple_white", 
                    hovermode="x unified",
                    height=500,
                    margin=dict(l=1, r=1, t=20, b=50),
                    xaxis=dict(
                        title="Time",
                        title_font=dict(size=20),
                        tickfont=dict(size=18),
                        gridcolor='lightgrey'
                    ),
                    yaxis=dict(
                        title="Temperature (°F)",
                        title_font=dict(size=20),
                        tickfont=dict(size=18),
                        gridcolor='lightgrey',
                        range=[min(df_curve['Bog_Temp_Predicted'].min(), tol) - 2, max(df_curve['Temp'].max(), tol) + 2]
                    ),
                    yaxis2=dict(
                        title="Cloud Cover (%)",
                        title_font=dict(size=20),
                        tickfont=dict(size=18),
                        range=[0, 100],
                        showgrid=False,
                        overlaying='y',
                        side='right'
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1,font=dict(size=20))
                )
                st.plotly_chart(fig, width = 'stretch')
        else:
            st.warning("Hourly curve data not available for the current HRRR window.")