import pandas as pd 
import numpy as np
from pyproj import Proj, transform
import folium
from folium.plugins import FeatureGroupSubGroup, HeatMap, HeatMapWithTime

def adjustDataFrame(src_df, points, coco_class, frame):

    # prepare dataframes
    ind_start = src_df.shape[0]
    ind_end = src_df.shape[0] + len(points)
    ind_range = range(ind_start,ind_end)

    points_df = pd.DataFrame(points, columns=list('XY'), index=ind_range)
    class_df = pd.DataFrame(coco_class, columns=list('t'), index=ind_range)
    f_df = pd.DataFrame(frame, dtype=np.int16, columns=list('f'), index=ind_range)

    # Merge dataframes
    out_df = pd.concat([points_df, class_df, f_df], axis=1, sort=False)
    return out_df


def convertCRS(df, src_epsg, dst_epsg):

    # Extract EPSG code 
    inProj = Proj('epsg:'+str(src_epsg))
    outProj = Proj('epsg:'+str(dst_epsg))

    # Split geometry from data
    data_df = df.drop(["X","Y"], axis=1)

    # Mirror src dataframe 
    points_df = df[['X','Y']]
    points_df.columns = "lat","lng"

    # Convert CRS
    for i in range(0,len(df)):
        p = transform(inProj,outProj,df.iloc[i]["X"],df.iloc[i]["Y"])
        points_df.iloc[i]['lat'] = p[0]
        points_df.iloc[i]['lng'] = p[1]

    # Merge dataframes
    out_df = pd.concat([points_df, data_df], axis=1, sort=False)
    return out_df


def createBaseMap(df):

    m = folium.Map(
        location=[np.mean(df['lat']), np.mean(df['lng'])],
        zoom_start=19,
        max_zoom=21,
        tiles='cartodbdark_matter')
    
    return m


def createDotMapSimple(df, m, clr):

    # draw circle Markers
    for i in range(0,len(df)):
        folium.Circle(
            location = [df.iloc[i]['lat'], df.iloc[i]['lng']],
            radius = 3,
            color = clr
        ).add_to(m)


def createDotMapFrame(df, m, gradient):

    # Adjust gradient palette
    n_groups = len(set(df['t']))
    colorscale = gradient.scale(0,n_groups)

    # Plot dot data
    for t in range(0,n_groups):
        type_i = list(set(df['t']))[t]
        fg = folium.FeatureGroup(type_i)

        # draw circle Marker
        for i in range(0,len(df)):
            if df.iloc[i]['t'] == type_i:
                folium.Circle(
                    location = [df.iloc[i]['lat'], df.iloc[i]['lng']],
                    radius = 0.4,
                    fill = True,
                    opacity = df.iloc[i]['f']/max(df['f']),
                    color = colorscale.rgb_hex_str(t)
                ).add_to(fg)

            # Add group to the map
            m.add_child(fg)

    # Add control to the map
    folium.LayerControl().add_to(m)
