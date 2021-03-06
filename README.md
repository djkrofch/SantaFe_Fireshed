# SantaFe_Fireshed
LANDIS-II based analysis of fuels treatment effects in the Santa Fe Fireshed. This research leverages output of 
initial data products generated for the larger extent across the US Rocky Mountains. Specifically, species paramterizations
for the dynamic landscape model, and raster inputs of the initial community composition across the analysis extent, 
are derivitives of parallel research projects.

### Project description

<b>Objectives:</b> Simulate forest carbon dynamics under projected climate and wildfire both in the presence
and absence of thinning and prescribed burning treatments. <p>

<b>Study Area:</b> The Santa Fe Fireshed (~43,000 ha), plus an additional buffer area
that will include up to an additional 40,000 ha. Simulations will be conducted using the LANDIS-II forest
succession and disturbance model. We will run two management scenarios, no-management and
thinning and prescribed burning. Areas where thinning will be implemented will be determined by a
planned treatment area shp file provided by The Nature Conservancy. Prescribed fire simulations will be
a function of the ecologically appropriate fire return interval as determined from the scientific literature.
All other model parameters (e.g. fire size distribution, fire probability, etc) will be developed from local
empirical data and held constant between the scenarios. We will use either 1/16 degree downscaled or
LOCA 1km downscaled CMIP5 climate projection data (from IPCC AR5) to drive forest growth
simulations. 

<b>Deliverables:</b> Final data products for the response variables and a final
report that presents the results of these simulations for the study area <p>
Estimates of the carbon outcomes of vegetation thinning for the upper Rio Grande watershed in New Mexico
(Sangre de Cristo and Jemez Mountains)<p>
Both will be delivered to The Nature Conservancy by <i>March 31, 2018.</i>

### Project overview
Here we develop the geospatial and tabular inputs required to drive the simulations. Generally speaking,
this is broken up into several core areas: geospatial data prep, ecosystem model parameterization, 
simulation climate preparation, fire system calibration, and management scenario creation.

<b>Geospatial data prep:</b> The SF_Fireshed_Prep notebook walks through the workflow used to create and format
the raster products required for LANDIS-II simulations. Portions of these processing steps were conducted in a GIS,
and portions were conducted in python or R environments. 

<b>Ecosystem model parameterization:</b> The SFF_PnET_Calib notebook walks through the workflow used to parameterize
the PnET extension in LANDIS-II. Generally speaking, publically available and literature derived species parameters
were used where available, and model performance was determined relative to in-situ ecosystem carbon flux measurements
using ameriflux data. 