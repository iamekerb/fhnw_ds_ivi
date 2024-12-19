import dash
from dash import dcc, html, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# Load data
flood_data = pd.read_csv('../data/generated/flood_data.csv', sep=',')
rain_data = pd.read_csv('../data/generated/rain_data.csv', sep=',')
regions_data = pd.read_csv('../data/generated/regions_ch.csv', sep=',')

# Print column names to verify them
print("Flood Data Columns:", flood_data.columns)
print("Rain Data Columns:", flood_data.columns)
print("Regions Data Columns:", regions_data.columns)

# Display the first few rows of the flood_data
print(flood_data.head())

# Convert 'DAY' column to datetime
rain_data['DAY'] = pd.to_datetime(rain_data['DAY'], format='%d.%m.%Y')

# Convert 'Start date' and 'End date' columns to datetime
flood_data['Start date'] = pd.to_datetime(flood_data['Start date'], format='%d.%m.%Y')
flood_data['End date'] = pd.to_datetime(flood_data['End date'], format='%d.%m.%Y')

# Extract Latitude and Longitude from the 'regions' column in flood_data
flood_data[['Latitude', 'Longitude']] = flood_data['regions'].str.strip('[]()').str.split(',', expand=True)
flood_data['Latitude'] = flood_data['Latitude'].astype(float)
flood_data['Longitude'] = flood_data['Longitude'].astype(float)

# Extract Latitude and Longitude from the 'Coordinates' column in regions_data
regions_data[['Latitude', 'Longitude']] = regions_data['Coordinates'].str.strip('[]()').str.split(',', expand=True)
regions_data['Latitude'] = regions_data['Latitude'].astype(float)
regions_data['Longitude'] = regions_data['Longitude'].astype(float)

# Merge flood_data with regions_data based on Latitude and Longitude
flood_data = pd.merge(flood_data, regions_data, on=['Latitude', 'Longitude'], how='left', suffixes=('', '_region'))

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    style={'position': 'relative', 'height': '100vh', 'width': '100vw', 'overflow': 'hidden'},
    children=[
        html.Div(
            style={
                'position': 'absolute', 'top': '0', 'left': '0', 'width': '100%', 'height': '100%', 'z-index': '-1',
                'overflow': 'hidden', 'filter': 'grayscale(100%)'
            },
            children=[
                html.Iframe(
                    src="https://www.youtube.com/embed/oDK0FNisfDg?autoplay=1&loop=1&playlist=oDK0FNisfDg&controls=0&mute=1",
                    style={
                        'position': 'absolute', 'top': '0', 'left': '0', 'width': '100vw', 'height': '100vh', 'border': 'none',
                        'transform': 'scale(1.5)', 'transform-origin': 'center'
                    }
                )
            ]
        ),
        dbc.Container([
            dbc.Row([
                dbc.Col(html.H1("Dashboard on Precipitation and Floods", className="text-center mt-3", style={'color': '#fef3c7'}), width=12),
                dbc.Col(html.H3("Interactive Visualization", className="text-center text-muted", style={'color': '#fef3c7'}), width=12),
                dbc.Col(html.H4("Boran", className="text-center text-muted mb-4", style={'color': '#fef3c7'}), width=12)
            ], style={'margin-bottom': '30px'}),
            dbc.Row([
                dbc.Col(
                    dcc.Dropdown(
                        id='country-dropdown',
                        options=[{'label': country, 'value': country} for country in flood_data['Country name'].unique()],
                        value=None,
                        clearable=True,
                        placeholder="Select a country",
                        style={'background-color': '#fef3c7'}
                    ),
                    width=12
                )
            ], style={'margin-bottom': '30px'}),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(id='map-plot', style={'background-color': '#fef3c7', 'border-radius': '10px', 'padding': '10px'}),
                    width=12
                )
            ], style={'margin-bottom': '30px'}),
            dbc.Row([
                dbc.Col(
                    dcc.Slider(
                        id='year-slider',
                        min=flood_data['Year'].min(),
                        max=flood_data['Year'].max(),
                        value=flood_data['Year'].max(),
                        marks={str(year): {'label': str(year), 'style': {'transform': 'rotate(45deg)', 'white-space': 'nowrap', 'color': '#fef3c7'}} for year in flood_data['Year'].unique()},
                        step=None
                    ),
                    width=12
                )
            ], style={'margin-bottom': '30px'}),
            dbc.Row([
                dbc.Col(
                    dcc.Dropdown(
                        id='timeframe-dropdown',
                        options=[
                            {'label': 'Daily', 'value': 'D'},
                            {'label': 'Weekly', 'value': 'W'},
                            {'label': 'Monthly', 'value': 'M'}
                        ],
                        value='D',
                        clearable=False,
                        style={'background-color': '#fef3c7'}
                    ),
                    width=12
                )
            ], style={'margin-bottom': '30px'}),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(id='precipitation-plot', style={'background-color': '#fef3c7', 'border-radius': '10px', 'padding': '10px'}),
                    width=12
                )
            ], style={'margin-bottom': '30px'}),
            dbc.Row([
                dbc.Col([
                    html.H5("Flood Events", style={'color': '#fef3c7'}),
                    dash_table.DataTable(
                        id='flood-table',
                        columns=[
                            {'name': 'Location', 'id': 'location'},
                            {'name': 'Start Date', 'id': 'Start date'},
                            {'name': 'End Date', 'id': 'End date'}
                        ],
                        style_table={'overflowX': 'auto', 'border': '1px solid black'},
                        style_cell={'textAlign': 'left', 'backgroundColor': '#fef3c7', 'color': 'black', 'border': '1px solid black'},
                        style_header={'backgroundColor': '#fef3c7', 'border': '1px solid black'},
                        style_as_list_view=True
                    )
                ], width=12)
            ], style={'margin-bottom': '30px'}),
            dbc.Row([
                dbc.Col([
                    html.H5("Damages and Losses", style={'color': '#fef3c7'}),
                    dash_table.DataTable(
                        id='damage-table',
                        columns=[
                            {'name': 'Location', 'id': 'location'},
                            {'name': 'Fatalities', 'id': 'Fatalities', 'type': 'numeric'},
                            {'name': 'Losses (mln EUR, 2020)', 'id': 'Losses (mln EUR, 2020)', 'type': 'numeric'}
                        ],
                        style_table={'overflowX': 'auto', 'border': '1px solid black'},
                        style_cell={'textAlign': 'center', 'backgroundColor': '#fef3c7', 'color': 'black', 'border': '1px solid black'},
                        style_as_list_view=True
                    )
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col(
                    html.Div(
                        [
                            html.H6("References:", style={'color': '#fef3c7'}),
                            html.Ul([
                                html.Li(html.A("Natural Hazards Europe", href="https://naturalhazards.eu/", target="_blank", style={'color': '#fef3c7'})),
                                html.Li(html.A("European Commission Authentication Service", href="https://ecas.ec.europa.eu/cas/login", target="_blank", style={'color': '#fef3c7'})),
                            ], style={'listStyleType': 'none', 'padding': '0'})
                        ],
                        className="text-center mt-4"
                    ),
                    width=12
                )
            ])
        ], fluid=True, style={'height': '90vh', 'overflowY': 'scroll'})
    ]
)


@app.callback(
    Output('map-plot', 'figure'),
    Output('flood-table', 'data'),
    Output('damage-table', 'data'),
    Input('year-slider', 'value'),
    Input('country-dropdown', 'value')
)
def update_map(year, country):
    # Filter data only by the selected year
    filtered_data = flood_data[flood_data['Year'] == year]

    # Check if a country is selected and filter accordingly
    if country:
        filtered_data = filtered_data[filtered_data['Country name'] == country]

    # Define map center and zoom level based on the selected country
    if country == 'Switzerland':
        map_center = {'lat': 46.8182, 'lon': 8.2275}
        zoom_level = 6
    else:
        # Default center to Europe and zoom out if not Switzerland or no country selected
        map_center = {'lat': 50.1109, 'lon': 8.6821}
        zoom_level = 3 if not country else 6

    # Replace NaN values in 'Losses (mln EUR, 2020)' with a placeholder for losses under 1 million EUR
    filtered_data['Losses (mln EUR, 2020)'] = filtered_data['Losses (mln EUR, 2020)'].fillna('< 1 mln EUR')

    # Determine marker size based on losses
    filtered_data['marker_size'] = filtered_data.apply(lambda row: max(10, row['Losses (mln EUR, 2020)'] / 10) if row['Losses (mln EUR, 2020)'] != '< 1 mln EUR' else 10, axis=1)

    # Check if there is data to plot
    if (filtered_data.empty) or ('Name' not in filtered_data.columns):
        fig = go.Figure()
        fig.update_layout(mapbox_style="carto-darkmatter")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        fig.update_layout(mapbox=dict(center=map_center, zoom=zoom_level))
        fig.add_annotation(
            x=0.5, y=0.5, text="No data available for this selection",
            showarrow=False, font=dict(size=20, color="white"),
            xref="paper", yref="paper"
        )
        flood_table_data = []
        damage_table_data = []
    else:
        # Plot data if available
        fig = px.scatter_mapbox(
            filtered_data,
            lat="Latitude",
            lon="Longitude",
            hover_name="Name",  # Use the region name for hover info
            size='marker_size' if country else None,  # Adjust size only if a country is selected
            size_max=30,
            color_discrete_sequence=["black"],
            zoom=zoom_level,
            height=500
        )
        # Remove latitude and longitude from hover data
        fig.update_traces(marker=dict(opacity=0.5 if country else 1.0))  # Adjust opacity only if a country is selected
        fig.update_traces(hovertemplate='<b>%{hovertext}</b>')
        fig.update_layout(mapbox_style="carto-positron")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        fig.update_layout(mapbox=dict(center=map_center, zoom=zoom_level))

        # Prepare flood table data
        flood_table_data = filtered_data[['Name', 'Start date', 'End date']].copy()
        flood_table_data['Start date'] = flood_table_data['Start date'].dt.strftime('%d.%m.%Y')
        flood_table_data['End date'] = flood_table_data['End date'].dt.strftime('%d.%m.%Y')
        flood_table_data = flood_table_data.rename(columns={'Name': 'location', 'Start date': 'Start date', 'End date': 'End date'}).to_dict('records')

        # Prepare damage table data
        damage_data = filtered_data[['Name', 'Fatalities', 'Losses (mln EUR, 2020)']].copy()
        damage_data['Fatalities'] = damage_data['Fatalities'].fillna('None')
        damage_grouped = damage_data.groupby(['Fatalities', 'Losses (mln EUR, 2020)']).agg({'Name': ', '.join}).reset_index()
        damage_grouped = damage_grouped.rename(columns={'Name': 'location'})
        damage_table_data = damage_grouped.to_dict('records')

    return fig, flood_table_data, damage_table_data

@app.callback(
    Output('precipitation-plot', 'figure'),
    Input('year-slider', 'value'),
    Input('timeframe-dropdown', 'value')
)
def update_charts(year, timeframe):
    filtered_data = rain_data[rain_data['DAY'].dt.year == year]

    if timeframe == 'D':
        filtered_data = filtered_data.groupby('DAY').mean().reset_index()
        y_label = 'Precipitation (mm/Tag)'
    elif timeframe == 'W':
        filtered_data = filtered_data.resample('W-Mon', on='DAY').mean().reset_index()
        y_label = 'Precipitation (mm/Woche)'
    elif timeframe == 'M':
        filtered_data = filtered_data.resample('M', on='DAY').mean().reset_index()
        y_label = 'Precipitation (mm/Monat)'

    # Add a column to indicate if the date falls within any flood event
    flood_periods = flood_data[['Start date', 'End date']].dropna()
    filtered_data['In_Flood_Period'] = filtered_data['DAY'].apply(lambda day: any((day >= start) and (day <= end) for start, end in zip(flood_periods['Start date'], flood_periods['End date'])))

    # Map True/False to 'Überschwemmung'/'Niederschläge'
    filtered_data['In_Flood_Period'] = filtered_data['In_Flood_Period'].map({True: 'Floods', False: 'Precipitation'})

    # Precipitation bar plot
    precipitation_fig = px.bar(
        filtered_data,
        x='DAY',
        y='PRECIPITATION',
        title=f'{timeframe}-Precipitation',
        labels={'PRECIPITATION': y_label},  # Dynamische Achsenbeschriftung
        color='In_Flood_Period',  # Color based on whether it's in a flood period
        color_discrete_map={'Floods': 'red', 'Precipitation': '#000080'}  # Red for flood periods, navy blue otherwise
    )
    precipitation_fig.update_layout(
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#fef3c7',
        xaxis=dict(showgrid=True, gridcolor='grey'),
        yaxis=dict(showgrid=True, gridcolor='grey')
    )

    return precipitation_fig

if __name__ == '__main__':
    app.run_server(debug=True)