# Geospatial

## Geodesic

#### ➡️ Code
- [geodesic.py](geodesic.py)

#### ➡️ Description
- Calculates the geodesic distance in miles between two latitude/longitude points in space.

#### ➡️ Inputs
- 4 columns for source and destination latititude and longitude specified as the variable ***col_names_to_pick*** in the recipe
eg: col_names_to_pick = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']


#### ➡️ Outputs
- Single numeric column with distance in miles. 

#### ➡️ Environment expectation
Dataset has columns which match the column names in this recipe specified by the variable ***col_names_to_pick***

#### ➡️ Dependenencies
- geopy

----

## MyHaversine

#### ➡️ Code
- [myhaversine.py](myhaversine.py)

#### ➡️ Description
- Calculates the haversine distance in miles between two latitude/longitude points in space.

#### ➡️ Inputs
- 4 columns specifying latitude and longitude i.e. two ****_latitude*** and ****_longitude*** named columns in the data set

#### ➡️ Outputs
- Single numeric column with distance in miles. 

#### ➡️ Environment expectation
- Computes miles between first two lat, long columns in the data set. Column names should have strings 'latitude' and 'longitude' in it.
 eg: ***pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude***

#### ➡️ Dependenencies
- None

...

----
