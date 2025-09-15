import os
import pandas as pd

def create_city_demand_matrix(airport_codes_path, data_folder, output_path):
    # Load airport mapping
    airport_mapping_df = pd.read_excel(airport_codes_path)
    airport_to_city = dict(zip(airport_mapping_df['AirportC1'], airport_mapping_df['City']))
    
    all_data = []
    for file in os.listdir(data_folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_folder, file))
            if 'airp_pr' not in df.columns or 'OBS_VALUE' not in df.columns:
                print(f"Skipping {file} (missing required columns)")
                continue
            
            # Split origin and destination from Eurostat format: CC_AAAA_CC_AAAA
            parts = df['airp_pr'].str.split('_')
            df['origin_airport'] = parts.str[0] + '_' + parts.str[1]
            df['dest_airport'] = parts.str[2] + '_' + parts.str[3]
            
            # Map airports
            #  to cities
            df['origin_city'] = df['origin_airport'].map(airport_to_city)
            df['dest_city'] = df['dest_airport'].map(airport_to_city)
            
            # Remove rows where mapping failed
            df = df.dropna(subset=['origin_city', 'dest_city'])
            
            # Group by city-to-city and sum passengers
            grouped = df.groupby(['origin_city', 'dest_city'])['OBS_VALUE'].sum().reset_index()
            all_data.append(grouped)
    
    # Combine all country data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Get all unique cities
    cities = sorted(set(combined_df['origin_city']).union(combined_df['dest_city']))
    
    # Create an empty demand matrix with zeros
    demand_matrix = pd.DataFrame(0, index=cities, columns=cities, dtype=int)
    
    # Fill in passenger counts
    for _, row in combined_df.iterrows():
        demand_matrix.loc[row['origin_city'], row['dest_city']] += int(row['OBS_VALUE'])
    
    # Set same-city demand to -1
    for city in cities:
        demand_matrix.loc[city, city] = -1
    
    # Save matrix to Excel
    demand_matrix.to_excel(output_path)
    print(f"City-to-city demand matrix saved to {output_path}")

# Example usage:
# create_city_demand_matrix(
#     airport_codes_path="Airport Codes.xlsx",
#     data_folder="csv_files_folder",
#     output_path="city_demand_matrix.xlsx"
# )

create_city_demand_matrix(
    airport_codes_path="Airport Codes.xlsx",
    data_folder="CountryDemands",
    output_path="city_demand_matrix.xlsx"
)
